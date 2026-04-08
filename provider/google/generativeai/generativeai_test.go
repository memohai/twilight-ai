package generativeai_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/memohai/twilight-ai/internal/testutil"
	"github.com/memohai/twilight-ai/provider/google/generativeai"
	"github.com/memohai/twilight-ai/sdk"
)

// ---------- unit tests (mock server) ----------

func TestDoGenerate(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models/gemini-2.0-flash:generateContent" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get("x-goog-api-key") != "test-key" {
			t.Errorf("unexpected api key header: %s", r.Header.Get("x-goog-api-key"))
		}

		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)

		if body["systemInstruction"] == nil {
			t.Error("expected systemInstruction to be set")
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{{
				"content": map[string]any{
					"role":  "model",
					"parts": []map[string]any{{"text": "Hello!"}},
				},
				"finishReason": "STOP",
			}},
			"usageMetadata": map[string]any{
				"promptTokenCount":     5,
				"candidatesTokenCount": 2,
				"totalTokenCount":      7,
			},
		})
	}))
	defer srv.Close()

	p := generativeai.New(
		generativeai.WithAPIKey("test-key"),
		generativeai.WithBaseURL(srv.URL),
	)

	model := p.ChatModel("gemini-2.0-flash")
	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model:  model,
		System: "You are helpful.",
		Messages: []sdk.Message{{
			Role:    sdk.MessageRoleUser,
			Content: []sdk.MessagePart{sdk.TextPart{Text: "Hi"}},
		}},
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}

	if result.Text != "Hello!" {
		t.Errorf("expected 'Hello!', got %q", result.Text)
	}
	if result.FinishReason != sdk.FinishReasonStop {
		t.Errorf("expected finish reason 'stop', got %q", result.FinishReason)
	}
	if result.Usage.InputTokens != 5 {
		t.Errorf("expected 5 input tokens, got %d", result.Usage.InputTokens)
	}
	if result.Usage.OutputTokens != 2 {
		t.Errorf("expected 2 output tokens, got %d", result.Usage.OutputTokens)
	}
}

func TestDoGenerate_ToolCall(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Tools []struct {
				FunctionDeclarations []struct {
					Name        string `json:"name"`
					Description string `json:"description"`
					Parameters  any    `json:"parameters"`
				} `json:"functionDeclarations"`
			} `json:"tools"`
			ToolConfig *struct {
				FunctionCallingConfig struct {
					Mode string `json:"mode"`
				} `json:"functionCallingConfig"`
			} `json:"toolConfig"`
		}
		json.NewDecoder(r.Body).Decode(&body)

		if len(body.Tools) != 1 {
			t.Fatalf("expected 1 tool group, got %d", len(body.Tools))
		}
		if len(body.Tools[0].FunctionDeclarations) != 1 {
			t.Fatalf("expected 1 function declaration, got %d", len(body.Tools[0].FunctionDeclarations))
		}
		if body.Tools[0].FunctionDeclarations[0].Name != "get_weather" {
			t.Errorf("tool name: got %q", body.Tools[0].FunctionDeclarations[0].Name)
		}
		if body.ToolConfig == nil || body.ToolConfig.FunctionCallingConfig.Mode != "AUTO" {
			t.Errorf("expected AUTO tool config mode")
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{{
				"content": map[string]any{
					"role": "model",
					"parts": []map[string]any{{
						"functionCall": map[string]any{
							"name": "get_weather",
							"args": map[string]any{"location": "Beijing"},
						},
					}},
				},
				"finishReason": "STOP",
			}},
			"usageMetadata": map[string]any{
				"promptTokenCount":     20,
				"candidatesTokenCount": 10,
				"totalTokenCount":      30,
			},
		})
	}))
	defer srv.Close()

	p := generativeai.New(generativeai.WithAPIKey("test-key"), generativeai.WithBaseURL(srv.URL))

	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model: p.ChatModel("gemini-2.0-flash"),
		Messages: []sdk.Message{{
			Role:    sdk.MessageRoleUser,
			Content: []sdk.MessagePart{sdk.TextPart{Text: "What's the weather in Beijing?"}},
		}},
		Tools: []sdk.Tool{{
			Name:        "get_weather",
			Description: "Get the weather for a location",
			Parameters: &jsonschema.Schema{
				Type: "object",
				Properties: map[string]*jsonschema.Schema{
					"location": {Type: "string"},
				},
				Required: []string{"location"},
			},
		}},
		ToolChoice: "auto",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}

	if result.FinishReason != sdk.FinishReasonToolCalls {
		t.Errorf("finish: got %q, want %q", result.FinishReason, sdk.FinishReasonToolCalls)
	}
	if len(result.ToolCalls) != 1 {
		t.Fatalf("tool calls: got %d, want 1", len(result.ToolCalls))
	}
	tc := result.ToolCalls[0]
	if tc.ToolName != "get_weather" {
		t.Errorf("tool name: got %q", tc.ToolName)
	}
	input, ok := tc.Input.(map[string]any)
	if !ok {
		t.Fatalf("input type: got %T", tc.Input)
	}
	if input["location"] != "Beijing" {
		t.Errorf("location: got %v", input["location"])
	}
}

func TestDoGenerate_ToolCallMultiTurn(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Contents []json.RawMessage `json:"contents"`
		}
		json.NewDecoder(r.Body).Decode(&body)

		if len(body.Contents) != 3 {
			t.Fatalf("expected 3 contents, got %d", len(body.Contents))
		}

		// verify model message has functionCall
		var modelMsg struct {
			Role  string `json:"role"`
			Parts []struct {
				FunctionCall *struct {
					Name string `json:"name"`
					Args any    `json:"args"`
				} `json:"functionCall,omitempty"`
			} `json:"parts"`
		}
		json.Unmarshal(body.Contents[1], &modelMsg)
		if modelMsg.Role != "model" {
			t.Errorf("msg[1] role: got %q", modelMsg.Role)
		}
		if len(modelMsg.Parts) != 1 || modelMsg.Parts[0].FunctionCall == nil {
			t.Errorf("msg[1] should have functionCall part")
		}

		// verify tool result message has functionResponse
		var toolMsg struct {
			Role  string `json:"role"`
			Parts []struct {
				FunctionResponse *struct {
					Name     string `json:"name"`
					Response struct {
						Name    string `json:"name"`
						Content any    `json:"content"`
					} `json:"response"`
				} `json:"functionResponse,omitempty"`
			} `json:"parts"`
		}
		json.Unmarshal(body.Contents[2], &toolMsg)
		if toolMsg.Role != "user" {
			t.Errorf("msg[2] role: got %q, want 'user'", toolMsg.Role)
		}
		if len(toolMsg.Parts) != 1 || toolMsg.Parts[0].FunctionResponse == nil {
			t.Errorf("msg[2] should have functionResponse part")
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{{
				"content": map[string]any{
					"role":  "model",
					"parts": []map[string]any{{"text": "It's sunny in Beijing."}},
				},
				"finishReason": "STOP",
			}},
			"usageMetadata": map[string]any{
				"promptTokenCount": 30, "candidatesTokenCount": 8, "totalTokenCount": 38,
			},
		})
	}))
	defer srv.Close()

	p := generativeai.New(generativeai.WithAPIKey("test-key"), generativeai.WithBaseURL(srv.URL))

	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model: p.ChatModel("gemini-2.0-flash"),
		Messages: []sdk.Message{
			{
				Role:    sdk.MessageRoleUser,
				Content: []sdk.MessagePart{sdk.TextPart{Text: "Weather?"}},
			},
			{
				Role: sdk.MessageRoleAssistant,
				Content: []sdk.MessagePart{sdk.ToolCallPart{
					ToolCallID: "call_abc",
					ToolName:   "get_weather",
					Input:      map[string]any{"location": "Beijing"},
				}},
			},
			{
				Role: sdk.MessageRoleTool,
				Content: []sdk.MessagePart{sdk.ToolResultPart{
					ToolCallID: "call_abc",
					ToolName:   "get_weather",
					Result:     map[string]any{"temp": 25, "condition": "sunny"},
				}},
			},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}

	if result.Text != "It's sunny in Beijing." {
		t.Errorf("text: got %q", result.Text)
	}
}

func TestDoGenerate_Reasoning(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		thought := true
		_ = thought
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{{
				"content": map[string]any{
					"role": "model",
					"parts": []map[string]any{
						{"text": "Let me think... 2+2=4", "thought": true},
						{"text": "The answer is 4."},
					},
				},
				"finishReason": "STOP",
			}},
			"usageMetadata": map[string]any{
				"promptTokenCount":     5,
				"candidatesTokenCount": 5,
				"totalTokenCount":      20,
				"thoughtsTokenCount":   10,
			},
		})
	}))
	defer srv.Close()

	p := generativeai.New(generativeai.WithAPIKey("k"), generativeai.WithBaseURL(srv.URL))
	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model:    p.ChatModel("gemini-2.5-flash"),
		Messages: []sdk.Message{sdk.UserMessage("2+2?")},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if result.Text != "The answer is 4." {
		t.Errorf("text: got %q", result.Text)
	}
	if result.Reasoning != "Let me think... 2+2=4" {
		t.Errorf("reasoning: got %q", result.Reasoning)
	}
	if result.Usage.ReasoningTokens != 10 {
		t.Errorf("reasoning tokens: got %d", result.Usage.ReasoningTokens)
	}
}

func TestDoGenerate_SystemInstruction(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			SystemInstruction *struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"systemInstruction"`
			Contents []struct {
				Role  string `json:"role"`
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"contents"`
		}
		json.NewDecoder(r.Body).Decode(&body)

		if body.SystemInstruction == nil {
			t.Fatal("expected systemInstruction")
		}
		if len(body.SystemInstruction.Parts) != 1 || body.SystemInstruction.Parts[0].Text != "Be concise." {
			t.Errorf("unexpected systemInstruction: %+v", body.SystemInstruction)
		}

		// system messages should not appear in contents
		for _, c := range body.Contents {
			if c.Role == "system" {
				t.Error("system role should not appear in contents")
			}
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{{
				"content": map[string]any{
					"role":  "model",
					"parts": []map[string]any{{"text": "OK"}},
				},
				"finishReason": "STOP",
			}},
			"usageMetadata": map[string]any{
				"promptTokenCount": 3, "candidatesTokenCount": 1, "totalTokenCount": 4,
			},
		})
	}))
	defer srv.Close()

	p := generativeai.New(generativeai.WithAPIKey("k"), generativeai.WithBaseURL(srv.URL))
	_, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model:    p.ChatModel("gemini-2.0-flash"),
		System:   "Be concise.",
		Messages: []sdk.Message{sdk.UserMessage("Hi")},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
}

func TestDoGenerate_ModelPathWithSlash(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/publishers/google/models/gemini-2.0-flash:generateContent" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{{
				"content":      map[string]any{"role": "model", "parts": []map[string]any{{"text": "OK"}}},
				"finishReason": "STOP",
			}},
			"usageMetadata": map[string]any{"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
		})
	}))
	defer srv.Close()

	p := generativeai.New(generativeai.WithAPIKey("k"), generativeai.WithBaseURL(srv.URL))
	_, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model:    p.ChatModel("publishers/google/models/gemini-2.0-flash"),
		Messages: []sdk.Message{sdk.UserMessage("Hi")},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
}

func TestDoGenerate_FinishReasonMapping(t *testing.T) {
	tests := []struct {
		googleReason string
		hasToolCalls bool
		expected     sdk.FinishReason
	}{
		{"STOP", false, sdk.FinishReasonStop},
		{"STOP", true, sdk.FinishReasonToolCalls},
		{"MAX_TOKENS", false, sdk.FinishReasonLength},
		{"SAFETY", false, sdk.FinishReasonContentFilter},
		{"RECITATION", false, sdk.FinishReasonContentFilter},
		{"IMAGE_SAFETY", false, sdk.FinishReasonContentFilter},
		{"BLOCKLIST", false, sdk.FinishReasonContentFilter},
		{"PROHIBITED_CONTENT", false, sdk.FinishReasonContentFilter},
		{"SPII", false, sdk.FinishReasonContentFilter},
		{"MALFORMED_FUNCTION_CALL", false, sdk.FinishReasonError},
		{"OTHER", false, sdk.FinishReasonOther},
		{"FINISH_REASON_UNSPECIFIED", false, sdk.FinishReasonOther},
	}

	for _, tt := range tests {
		t.Run(tt.googleReason, func(t *testing.T) {
			parts := []map[string]any{}
			if tt.hasToolCalls {
				parts = append(parts, map[string]any{
					"functionCall": map[string]any{"name": "test", "args": map[string]any{}},
				})
			} else {
				parts = append(parts, map[string]any{"text": "ok"})
			}

			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(map[string]any{
					"candidates": []map[string]any{{
						"content":      map[string]any{"role": "model", "parts": parts},
						"finishReason": tt.googleReason,
					}},
					"usageMetadata": map[string]any{
						"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2,
					},
				})
			}))
			defer srv.Close()

			p := generativeai.New(generativeai.WithAPIKey("k"), generativeai.WithBaseURL(srv.URL))
			result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
				Model:    p.ChatModel("gemini-2.0-flash"),
				Messages: []sdk.Message{sdk.UserMessage("test")},
			})
			if err != nil {
				t.Fatalf("DoGenerate: %v", err)
			}
			if result.FinishReason != tt.expected {
				t.Errorf("finish reason: got %q, want %q", result.FinishReason, tt.expected)
			}
			if result.RawFinishReason != tt.googleReason {
				t.Errorf("raw finish reason: got %q, want %q", result.RawFinishReason, tt.googleReason)
			}
		})
	}
}

func TestDoGenerate_NoModel(t *testing.T) {
	p := generativeai.New(generativeai.WithAPIKey("k"))
	_, err := p.DoGenerate(context.Background(), sdk.GenerateParams{})
	if err == nil {
		t.Fatal("expected error for nil model")
	}
}

func TestDoStream_NoModel(t *testing.T) {
	p := generativeai.New(generativeai.WithAPIKey("k"))
	_, err := p.DoStream(context.Background(), sdk.GenerateParams{})
	if err == nil {
		t.Fatal("expected error for nil model")
	}
}

// ---------- streaming tests ----------

func TestDoStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models/gemini-2.0-flash:streamGenerateContent" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.URL.Query().Get("alt") != "sse" {
			t.Errorf("expected alt=sse query param, got %q", r.URL.Query().Get("alt"))
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("server does not support flushing")
		}

		chunks := []string{
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]}}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1,"totalTokenCount":4}}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":" world"}]}}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2,"totalTokenCount":5}}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":""}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2,"totalTokenCount":5}}`,
		}
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", c)
			flusher.Flush()
		}
	}))
	defer srv.Close()

	p := generativeai.New(
		generativeai.WithAPIKey("test-key"),
		generativeai.WithBaseURL(srv.URL),
	)

	sr, err := p.DoStream(context.Background(), sdk.GenerateParams{
		Model: p.ChatModel("gemini-2.0-flash"),
		Messages: []sdk.Message{{
			Role:    sdk.MessageRoleUser,
			Content: []sdk.MessagePart{sdk.TextPart{Text: "Hi"}},
		}},
	})
	if err != nil {
		t.Fatalf("DoStream failed: %v", err)
	}

	var collected string
	var gotStart, gotFinish bool
	for part := range sr.Stream {
		switch p := part.(type) {
		case *sdk.StartPart:
			gotStart = true
		case *sdk.TextDeltaPart:
			collected += p.Text
		case *sdk.FinishPart:
			gotFinish = true
			if p.FinishReason != sdk.FinishReasonStop {
				t.Errorf("expected stop, got %q", p.FinishReason)
			}
		}
	}

	if !gotStart {
		t.Error("missing StartPart")
	}
	if !gotFinish {
		t.Error("missing FinishPart")
	}
	if collected != "Hello world" {
		t.Errorf("expected 'Hello world', got %q", collected)
	}
}

func TestDoStream_ToolCall(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)

		chunks := []string{
			`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"location":"Tokyo"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}`,
		}
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", c)
			flusher.Flush()
		}
	}))
	defer srv.Close()

	p := generativeai.New(generativeai.WithAPIKey("test-key"), generativeai.WithBaseURL(srv.URL))

	sr, err := p.DoStream(context.Background(), sdk.GenerateParams{
		Model: p.ChatModel("gemini-2.0-flash"),
		Messages: []sdk.Message{{
			Role:    sdk.MessageRoleUser,
			Content: []sdk.MessagePart{sdk.TextPart{Text: "Weather in Tokyo?"}},
		}},
		Tools: []sdk.Tool{{Name: "get_weather", Parameters: &jsonschema.Schema{Type: "object"}}},
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}

	var (
		gotInputStart bool
		gotInputEnd   bool
		argsDelta     string
		gotToolCall   *sdk.StreamToolCallPart
		gotFinish     bool
	)

	for part := range sr.Stream {
		switch p := part.(type) {
		case *sdk.ToolInputStartPart:
			gotInputStart = true
			if p.ToolName != "get_weather" {
				t.Errorf("input start tool name: got %q", p.ToolName)
			}
		case *sdk.ToolInputDeltaPart:
			argsDelta += p.Delta
		case *sdk.ToolInputEndPart:
			gotInputEnd = true
		case *sdk.StreamToolCallPart:
			gotToolCall = p
		case *sdk.FinishPart:
			gotFinish = true
			if p.FinishReason != sdk.FinishReasonToolCalls {
				t.Errorf("finish reason: got %q", p.FinishReason)
			}
		case *sdk.ErrorPart:
			t.Fatalf("error: %v", p.Error)
		}
	}

	if !gotInputStart {
		t.Error("missing ToolInputStartPart")
	}
	if !gotInputEnd {
		t.Error("missing ToolInputEndPart")
	}
	if argsDelta != `{"location":"Tokyo"}` {
		t.Errorf("args delta: got %q", argsDelta)
	}
	if gotToolCall == nil {
		t.Fatal("missing StreamToolCallPart")
	} else if gotToolCall.ToolName != "get_weather" {
		t.Errorf("tool call name: got %q", gotToolCall.ToolName)
	}
	input, ok := gotToolCall.Input.(map[string]any)
	if !ok || input["location"] != "Tokyo" {
		t.Errorf("tool call input: %+v", gotToolCall.Input)
	}
	if !gotFinish {
		t.Error("missing FinishPart")
	}
}

func TestDoStream_Reasoning(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)

		chunks := []string{
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"Let me think...","thought":true}]}}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":0,"totalTokenCount":5,"thoughtsTokenCount":3}}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":" 2+2=4","thought":true}]}}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":0,"totalTokenCount":5,"thoughtsTokenCount":5}}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"The answer is 4."}]}}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":4,"totalTokenCount":14,"thoughtsTokenCount":5}}`,
			`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":4,"totalTokenCount":14,"thoughtsTokenCount":5}}`,
		}
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", c)
			flusher.Flush()
		}
	}))
	defer srv.Close()

	p := generativeai.New(generativeai.WithAPIKey("k"), generativeai.WithBaseURL(srv.URL))
	sr, err := p.DoStream(context.Background(), sdk.GenerateParams{
		Model:    p.ChatModel("gemini-2.5-flash"),
		Messages: []sdk.Message{sdk.UserMessage("2+2?")},
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}

	var reasoning, text string
	var gotReasoningStart, gotReasoningEnd, gotTextStart, gotTextEnd bool
	for part := range sr.Stream {
		switch p := part.(type) {
		case *sdk.ReasoningStartPart:
			gotReasoningStart = true
		case *sdk.ReasoningDeltaPart:
			reasoning += p.Text
		case *sdk.ReasoningEndPart:
			gotReasoningEnd = true
		case *sdk.TextStartPart:
			gotTextStart = true
		case *sdk.TextDeltaPart:
			text += p.Text
		case *sdk.TextEndPart:
			gotTextEnd = true
		case *sdk.ErrorPart:
			t.Fatalf("error: %v", p.Error)
		}
	}

	if !gotReasoningStart {
		t.Error("missing ReasoningStartPart")
	}
	if !gotReasoningEnd {
		t.Error("missing ReasoningEndPart")
	}
	if !gotTextStart {
		t.Error("missing TextStartPart")
	}
	if !gotTextEnd {
		t.Error("missing TextEndPart")
	}
	if reasoning != "Let me think... 2+2=4" {
		t.Errorf("reasoning: got %q", reasoning)
	}
	if text != "The answer is 4." {
		t.Errorf("text: got %q", text)
	}
}

func TestDoStream_FlushOnAbruptEnd(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)

		chunks := []string{
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"Thinking...","thought":true}]}}]}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]}}]}`,
		}
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", c)
			flusher.Flush()
		}
	}))
	defer srv.Close()

	p := generativeai.New(generativeai.WithAPIKey("k"), generativeai.WithBaseURL(srv.URL))
	sr, err := p.DoStream(context.Background(), sdk.GenerateParams{
		Model:    p.ChatModel("m"),
		Messages: []sdk.Message{sdk.UserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}

	var gotReasoningEnd, gotTextEnd, gotFinish bool
	for part := range sr.Stream {
		switch part.(type) {
		case *sdk.ReasoningEndPart:
			gotReasoningEnd = true
		case *sdk.TextEndPart:
			gotTextEnd = true
		case *sdk.FinishPart:
			gotFinish = true
		}
	}

	if !gotReasoningEnd {
		t.Error("missing ReasoningEndPart on abrupt stream end")
	}
	if !gotTextEnd {
		t.Error("missing TextEndPart on abrupt stream end")
	}
	if !gotFinish {
		t.Error("missing FinishPart on abrupt stream end")
	}
}

func TestDoGenerate_ToolChoiceNone(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			ToolConfig *struct {
				FunctionCallingConfig struct {
					Mode string `json:"mode"`
				} `json:"functionCallingConfig"`
			} `json:"toolConfig"`
		}
		json.NewDecoder(r.Body).Decode(&body)

		if body.ToolConfig == nil || body.ToolConfig.FunctionCallingConfig.Mode != "NONE" {
			t.Errorf("expected NONE mode, got %+v", body.ToolConfig)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{{
				"content":      map[string]any{"role": "model", "parts": []map[string]any{{"text": "OK"}}},
				"finishReason": "STOP",
			}},
			"usageMetadata": map[string]any{"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
		})
	}))
	defer srv.Close()

	p := generativeai.New(generativeai.WithAPIKey("k"), generativeai.WithBaseURL(srv.URL))
	_, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model:    p.ChatModel("gemini-2.0-flash"),
		Messages: []sdk.Message{sdk.UserMessage("test")},
		Tools: []sdk.Tool{{
			Name:       "tool1",
			Parameters: &jsonschema.Schema{Type: "object"},
		}},
		ToolChoice: "none",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
}

func TestDoGenerate_ToolChoiceRequired(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			ToolConfig *struct {
				FunctionCallingConfig struct {
					Mode string `json:"mode"`
				} `json:"functionCallingConfig"`
			} `json:"toolConfig"`
		}
		json.NewDecoder(r.Body).Decode(&body)

		if body.ToolConfig == nil || body.ToolConfig.FunctionCallingConfig.Mode != "ANY" {
			t.Errorf("expected ANY mode, got %+v", body.ToolConfig)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{{
				"content": map[string]any{
					"role":  "model",
					"parts": []map[string]any{{"functionCall": map[string]any{"name": "tool1", "args": map[string]any{}}}},
				},
				"finishReason": "STOP",
			}},
			"usageMetadata": map[string]any{"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
		})
	}))
	defer srv.Close()

	p := generativeai.New(generativeai.WithAPIKey("k"), generativeai.WithBaseURL(srv.URL))
	_, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model:    p.ChatModel("gemini-2.0-flash"),
		Messages: []sdk.Message{sdk.UserMessage("test")},
		Tools: []sdk.Tool{{
			Name:       "tool1",
			Parameters: &jsonschema.Schema{Type: "object"},
		}},
		ToolChoice: "required",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
}

func TestDoGenerate_ResponseFormatJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			GenerationConfig struct {
				ResponseMimeType string `json:"responseMimeType"`
			} `json:"generationConfig"`
		}
		json.NewDecoder(r.Body).Decode(&body)

		if body.GenerationConfig.ResponseMimeType != "application/json" {
			t.Errorf("expected application/json, got %q", body.GenerationConfig.ResponseMimeType)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{{
				"content":      map[string]any{"role": "model", "parts": []map[string]any{{"text": `{"answer": 42}`}}},
				"finishReason": "STOP",
			}},
			"usageMetadata": map[string]any{"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
		})
	}))
	defer srv.Close()

	p := generativeai.New(generativeai.WithAPIKey("k"), generativeai.WithBaseURL(srv.URL))
	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model:          p.ChatModel("gemini-2.0-flash"),
		Messages:       []sdk.Message{sdk.UserMessage("test")},
		ResponseFormat: &sdk.ResponseFormat{Type: sdk.ResponseFormatJSONObject},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if result.Text != `{"answer": 42}` {
		t.Errorf("text: got %q", result.Text)
	}
}

func TestDoGenerate_Usage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{{
				"content":      map[string]any{"role": "model", "parts": []map[string]any{{"text": "OK"}}},
				"finishReason": "STOP",
			}},
			"usageMetadata": map[string]any{
				"promptTokenCount":        100,
				"candidatesTokenCount":    50,
				"totalTokenCount":         170,
				"cachedContentTokenCount": 20,
				"thoughtsTokenCount":      30,
			},
		})
	}))
	defer srv.Close()

	p := generativeai.New(generativeai.WithAPIKey("k"), generativeai.WithBaseURL(srv.URL))
	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model:    p.ChatModel("m"),
		Messages: []sdk.Message{sdk.UserMessage("test")},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}

	u := result.Usage
	if u.InputTokens != 100 {
		t.Errorf("input tokens: got %d", u.InputTokens)
	}
	if u.OutputTokens != 80 {
		t.Errorf("output tokens: got %d, want 80 (50+30)", u.OutputTokens)
	}
	if u.CachedInputTokens != 20 {
		t.Errorf("cached input tokens: got %d", u.CachedInputTokens)
	}
	if u.ReasoningTokens != 30 {
		t.Errorf("reasoning tokens: got %d", u.ReasoningTokens)
	}
	if u.InputTokenDetails.CacheReadTokens != 20 {
		t.Errorf("input cache read: got %d", u.InputTokenDetails.CacheReadTokens)
	}
	if u.InputTokenDetails.NoCacheTokens != 80 {
		t.Errorf("input no cache: got %d, want 80 (100-20)", u.InputTokenDetails.NoCacheTokens)
	}
	if u.OutputTokenDetails.TextTokens != 50 {
		t.Errorf("output text tokens: got %d", u.OutputTokenDetails.TextTokens)
	}
	if u.OutputTokenDetails.ReasoningTokens != 30 {
		t.Errorf("output reasoning tokens: got %d", u.OutputTokenDetails.ReasoningTokens)
	}
}

func TestProviderName(t *testing.T) {
	p := generativeai.New()
	if p.Name() != "google-generative-ai" {
		t.Errorf("name: got %q", p.Name())
	}
}

// ---------- integration tests (real API, skipped without env) ----------

func envOrSkip(t *testing.T, key string) string {
	t.Helper()
	v := os.Getenv(key)
	if v == "" {
		t.Skipf("skipping: %s not set", key)
	}
	return v
}

func newIntegrationProvider(t *testing.T) *generativeai.Provider {
	t.Helper()
	apiKey := envOrSkip(t, "GOOGLE_GENERATIVE_AI_API_KEY")
	opts := []generativeai.Option{generativeai.WithAPIKey(apiKey)}
	if base := os.Getenv("GOOGLE_GENERATIVE_AI_BASE_URL"); base != "" {
		opts = append(opts, generativeai.WithBaseURL(base))
	}
	return generativeai.New(opts...)
}

func integrationModel(t *testing.T) *sdk.Model {
	t.Helper()
	m := os.Getenv("GOOGLE_GENERATIVE_AI_MODEL")
	if m == "" {
		m = "gemini-2.0-flash"
	}
	return &sdk.Model{ID: m}
}

func TestIntegration_DoGenerate(t *testing.T) {
	p := newIntegrationProvider(t)
	model := integrationModel(t)
	model.Provider = p
	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model: model,
		Messages: []sdk.Message{{
			Role:    sdk.MessageRoleUser,
			Content: []sdk.MessagePart{sdk.TextPart{Text: "Say hello in one word."}},
		}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	t.Logf("text=%q finish=%s tokens=%d/%d", result.Text, result.FinishReason,
		result.Usage.InputTokens, result.Usage.OutputTokens)

	if result.Text == "" {
		t.Error("expected non-empty text")
	}
}

func TestIntegration_DoStream(t *testing.T) {
	p := newIntegrationProvider(t)
	model := integrationModel(t)
	model.Provider = p
	sr, err := p.DoStream(context.Background(), sdk.GenerateParams{
		Model: model,
		Messages: []sdk.Message{{
			Role:    sdk.MessageRoleUser,
			Content: []sdk.MessagePart{sdk.TextPart{Text: "Count from 1 to 5."}},
		}},
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}

	var text string
	for part := range sr.Stream {
		switch p := part.(type) {
		case *sdk.TextDeltaPart:
			text += p.Text
			t.Logf("text delta: %q", p.Text)
		case *sdk.ErrorPart:
			t.Fatalf("stream error: %v", p.Error)
		case *sdk.FinishPart:
			t.Logf("finish=%s", p.FinishReason)
		}
	}
	t.Logf("streamed text: %q", text)
	if text == "" {
		t.Error("expected non-empty streamed text")
	}
}

func TestIntegration_ToolCall(t *testing.T) {
	p := newIntegrationProvider(t)
	model := integrationModel(t)
	model.Provider = p
	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model: model,
		Messages: []sdk.Message{{
			Role:    sdk.MessageRoleUser,
			Content: []sdk.MessagePart{sdk.TextPart{Text: "What's the weather in San Francisco?"}},
		}},
		Tools: []sdk.Tool{{
			Name:        "get_weather",
			Description: "Get the weather for a location",
			Parameters: &jsonschema.Schema{
				Type: "object",
				Properties: map[string]*jsonschema.Schema{
					"location": {Type: "string", Description: "City name"},
				},
				Required: []string{"location"},
			},
		}},
		ToolChoice: "auto",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	t.Logf("finish=%s toolCalls=%d text=%q", result.FinishReason, len(result.ToolCalls), result.Text)

	if len(result.ToolCalls) == 0 {
		t.Error("expected at least one tool call")
	}
	for _, tc := range result.ToolCalls {
		t.Logf("  tool=%q id=%s input=%v", tc.ToolName, tc.ToolCallID, tc.Input)
	}
}

// ---------- ListModels / Test / TestModel unit tests ----------

func TestListModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodGet {
			t.Errorf("expected GET, got %s", r.Method)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"models": []map[string]any{
				{"name": "models/gemini-2.5-pro", "displayName": "Gemini 2.5 Pro"},
				{"name": "models/gemini-2.5-flash", "displayName": "Gemini 2.5 Flash"},
			},
		})
	}))
	defer srv.Close()

	p := generativeai.New(
		generativeai.WithAPIKey("test-key"),
		generativeai.WithBaseURL(srv.URL),
	)

	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels failed: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(models))
	}
	if models[0].ID != "gemini-2.5-pro" {
		t.Errorf("expected first model 'gemini-2.5-pro', got %q", models[0].ID)
	}
	if models[0].DisplayName != "Gemini 2.5 Pro" {
		t.Errorf("expected display name 'Gemini 2.5 Pro', got %q", models[0].DisplayName)
	}
}

func TestProviderTest_OK(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"models": []map[string]any{},
		})
	}))
	defer srv.Close()

	p := generativeai.New(
		generativeai.WithAPIKey("test-key"),
		generativeai.WithBaseURL(srv.URL),
	)

	result := p.Test(context.Background())
	if result.Status != sdk.ProviderStatusOK {
		t.Errorf("expected status OK, got %q", result.Status)
	}
}

func TestProviderTest_Unhealthy(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{"message": "API key not valid"},
		})
	}))
	defer srv.Close()

	p := generativeai.New(
		generativeai.WithAPIKey("bad-key"),
		generativeai.WithBaseURL(srv.URL),
	)

	result := p.Test(context.Background())
	if result.Status != sdk.ProviderStatusUnhealthy {
		t.Errorf("expected status Unhealthy, got %q", result.Status)
	}
}

func TestProviderTest_Unreachable(t *testing.T) {
	p := generativeai.New(
		generativeai.WithAPIKey("test-key"),
		generativeai.WithBaseURL("http://127.0.0.1:1"),
	)

	result := p.Test(context.Background())
	if result.Status != sdk.ProviderStatusUnreachable {
		t.Errorf("expected status Unreachable, got %q", result.Status)
	}
}

func TestTestModel_Supported(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models/gemini-2.5-pro" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"name": "models/gemini-2.5-pro", "displayName": "Gemini 2.5 Pro",
		})
	}))
	defer srv.Close()

	p := generativeai.New(
		generativeai.WithAPIKey("test-key"),
		generativeai.WithBaseURL(srv.URL),
	)

	result, err := p.TestModel(context.Background(), "gemini-2.5-pro")
	if err != nil {
		t.Fatalf("TestModel failed: %v", err)
	}
	if !result.Supported {
		t.Error("expected model to be supported")
	}
}

func TestTestModel_NotSupported(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{"message": "model not found"},
		})
	}))
	defer srv.Close()

	p := generativeai.New(
		generativeai.WithAPIKey("test-key"),
		generativeai.WithBaseURL(srv.URL),
	)

	result, err := p.TestModel(context.Background(), "nonexistent")
	if err != nil {
		t.Fatalf("TestModel failed: %v", err)
	}
	if result.Supported {
		t.Error("expected model to not be supported")
	}
}

func TestMain(m *testing.M) {
	testutil.LoadEnv()
	os.Exit(m.Run())
}
