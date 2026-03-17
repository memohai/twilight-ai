package openai_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/memohai/twilight-ai/internal/testutil"
	"github.com/memohai/twilight-ai/provider/openai"
	"github.com/memohai/twilight-ai/sdk"
)

// ---------- unit tests (mock server) ----------

func TestDoGenerate(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("unexpected auth header: %s", r.Header.Get("Authorization"))
		}

		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if body["model"] != "gpt-4o-mini" {
			t.Errorf("unexpected model: %v", body["model"])
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":      "chatcmpl-test",
			"object":  "chat.completion",
			"created": 1700000000,
			"model":   "gpt-4o-mini",
			"choices": []map[string]any{{
				"index":         0,
				"finish_reason": "stop",
				"message":       map[string]any{"role": "assistant", "content": "Hello!"},
			}},
			"usage": map[string]any{
				"prompt_tokens":     5,
				"completion_tokens": 2,
				"total_tokens":      7,
			},
		})
	}))
	defer srv.Close()

	p := openai.NewCompletions(
		openai.WithAPIKey("test-key"),
		openai.WithBaseURL(srv.URL),
	)

	model := &sdk.Model{ID: "gpt-4o-mini"}
	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model: model,
		Messages: []sdk.Message{{
			Role:  sdk.MessageRoleUser,
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

func TestDoStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("server does not support flushing")
		}

		chunks := []string{
			`{"id":"chunk-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}`,
			`{"id":"chunk-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}`,
			`{"id":"chunk-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`,
		}
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", c)
			flusher.Flush()
		}
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	p := openai.NewCompletions(
		openai.WithAPIKey("test-key"),
		openai.WithBaseURL(srv.URL),
	)

	model := &sdk.Model{ID: "gpt-4o-mini"}
	sr, err := p.DoStream(context.Background(), sdk.GenerateParams{
		Model: model,
		Messages: []sdk.Message{{
			Role:  sdk.MessageRoleUser,
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

func TestDoGenerate_WithImage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Role    string `json:"role"`
				Content any    `json:"content"`
			} `json:"messages"`
		}
		json.NewDecoder(r.Body).Decode(&body)

		if len(body.Messages) != 1 {
			t.Fatalf("expected 1 message, got %d", len(body.Messages))
		}

		parts, ok := body.Messages[0].Content.([]any)
		if !ok {
			t.Fatalf("expected array content, got %T", body.Messages[0].Content)
		}
		if len(parts) != 2 {
			t.Fatalf("expected 2 content parts, got %d", len(parts))
		}

		textPart := parts[0].(map[string]any)
		if textPart["type"] != "text" || textPart["text"] != "What is in this image?" {
			t.Errorf("unexpected text part: %v", textPart)
		}

		imgPart := parts[1].(map[string]any)
		if imgPart["type"] != "image_url" {
			t.Errorf("expected image_url type, got %v", imgPart["type"])
		}
		imgURL := imgPart["image_url"].(map[string]any)
		if imgURL["url"] != "https://example.com/cat.png" {
			t.Errorf("unexpected image url: %v", imgURL["url"])
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":      "chatcmpl-img",
			"object":  "chat.completion",
			"created": 1700000000,
			"model":   "gpt-4o-mini",
			"choices": []map[string]any{{
				"index":         0,
				"finish_reason": "stop",
				"message":       map[string]any{"role": "assistant", "content": "A cat."},
			}},
			"usage": map[string]any{
				"prompt_tokens":     20,
				"completion_tokens": 3,
				"total_tokens":      23,
			},
		})
	}))
	defer srv.Close()

	p := openai.NewCompletions(
		openai.WithAPIKey("test-key"),
		openai.WithBaseURL(srv.URL),
	)

	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model: &sdk.Model{ID: "gpt-4o-mini"},
		Messages: []sdk.Message{{
			Role: sdk.MessageRoleUser,
			Content: []sdk.MessagePart{
				sdk.TextPart{Text: "What is in this image?"},
				sdk.ImagePart{Image: "https://example.com/cat.png", MediaType: "image/png"},
			},
		}},
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}

	if result.Text != "A cat." {
		t.Errorf("expected 'A cat.', got %q", result.Text)
	}
	if result.Usage.InputTokens != 20 {
		t.Errorf("expected 20 input tokens, got %d", result.Usage.InputTokens)
	}
}

func TestDoGenerate_ToolCall(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Tools []struct {
				Type     string `json:"type"`
				Function struct {
					Name        string `json:"name"`
					Description string `json:"description"`
					Parameters  any    `json:"parameters"`
				} `json:"function"`
			} `json:"tools"`
			ToolChoice string `json:"tool_choice"`
		}
		json.NewDecoder(r.Body).Decode(&body)

		if len(body.Tools) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(body.Tools))
		}
		if body.Tools[0].Function.Name != "get_weather" {
			t.Errorf("tool name: got %q", body.Tools[0].Function.Name)
		}
		if body.ToolChoice != "auto" {
			t.Errorf("tool_choice: got %q", body.ToolChoice)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id": "chatcmpl-tool", "object": "chat.completion", "model": "gpt-4o-mini",
			"choices": []map[string]any{{
				"index":         0,
				"finish_reason": "tool_calls",
				"message": map[string]any{
					"role":    "assistant",
					"content": "",
					"tool_calls": []map[string]any{{
						"id":   "call_abc123",
						"type": "function",
						"function": map[string]any{
							"name":      "get_weather",
							"arguments": `{"location":"Beijing"}`,
						},
					}},
				},
			}},
			"usage": map[string]any{"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
		})
	}))
	defer srv.Close()

	p := openai.NewCompletions(openai.WithAPIKey("test-key"), openai.WithBaseURL(srv.URL))

	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model: &sdk.Model{ID: "gpt-4o-mini"},
		Messages: []sdk.Message{{
			Role:    sdk.MessageRoleUser,
			Content: []sdk.MessagePart{sdk.TextPart{Text: "What's the weather in Beijing?"}},
		}},
		Tools: []sdk.Tool{{
			Name:        "get_weather",
			Description: "Get the weather for a location",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{"type": "string"},
				},
				"required": []string{"location"},
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
	if tc.ToolCallID != "call_abc123" {
		t.Errorf("tool call id: got %q", tc.ToolCallID)
	}
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
			Messages []json.RawMessage `json:"messages"`
		}
		json.NewDecoder(r.Body).Decode(&body)

		if len(body.Messages) != 3 {
			t.Fatalf("expected 3 messages, got %d", len(body.Messages))
		}

		// verify assistant message has tool_calls
		var assistantMsg struct {
			Role      string `json:"role"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		}
		json.Unmarshal(body.Messages[1], &assistantMsg)
		if assistantMsg.Role != "assistant" {
			t.Errorf("msg[1] role: got %q", assistantMsg.Role)
		}
		if len(assistantMsg.ToolCalls) != 1 || assistantMsg.ToolCalls[0].ID != "call_abc" {
			t.Errorf("msg[1] tool_calls: %+v", assistantMsg.ToolCalls)
		}

		// verify tool result message
		var toolMsg struct {
			Role       string `json:"role"`
			ToolCallID string `json:"tool_call_id"`
			Content    string `json:"content"`
		}
		json.Unmarshal(body.Messages[2], &toolMsg)
		if toolMsg.Role != "tool" {
			t.Errorf("msg[2] role: got %q", toolMsg.Role)
		}
		if toolMsg.ToolCallID != "call_abc" {
			t.Errorf("msg[2] tool_call_id: got %q", toolMsg.ToolCallID)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id": "chatcmpl-2", "object": "chat.completion", "model": "gpt-4o-mini",
			"choices": []map[string]any{{
				"index": 0, "finish_reason": "stop",
				"message": map[string]any{"role": "assistant", "content": "It's sunny in Beijing."},
			}},
			"usage": map[string]any{"prompt_tokens": 30, "completion_tokens": 8, "total_tokens": 38},
		})
	}))
	defer srv.Close()

	p := openai.NewCompletions(openai.WithAPIKey("test-key"), openai.WithBaseURL(srv.URL))

	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model: &sdk.Model{ID: "gpt-4o-mini"},
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

func TestDoStream_ToolCall(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)

		chunks := []string{
			// first chunk: tool call start with id and name
			`{"id":"chunk-1","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_xyz","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}`,
			// second chunk: arguments delta
			`{"id":"chunk-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"location\""}}]},"finish_reason":null}]}`,
			// third chunk: arguments continued
			`{"id":"chunk-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\"Tokyo\"}"}}]},"finish_reason":null}]}`,
			// finish
			`{"id":"chunk-1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
		}
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", c)
			flusher.Flush()
		}
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	p := openai.NewCompletions(openai.WithAPIKey("test-key"), openai.WithBaseURL(srv.URL))

	sr, err := p.DoStream(context.Background(), sdk.GenerateParams{
		Model: &sdk.Model{ID: "gpt-4o-mini"},
		Messages: []sdk.Message{{
			Role:    sdk.MessageRoleUser,
			Content: []sdk.MessagePart{sdk.TextPart{Text: "Weather in Tokyo?"}},
		}},
		Tools: []sdk.Tool{{Name: "get_weather", Parameters: map[string]any{"type": "object"}}},
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}

	var (
		gotInputStart  bool
		gotInputEnd    bool
		argsDelta      string
		gotToolCall    *sdk.StreamToolCallPart
		gotFinishStep  bool
		gotFinish      bool
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
		case *sdk.FinishStepPart:
			gotFinishStep = true
			if p.FinishReason != sdk.FinishReasonToolCalls {
				t.Errorf("finish step reason: got %q", p.FinishReason)
			}
		case *sdk.FinishPart:
			gotFinish = true
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
	}
	if gotToolCall.ToolCallID != "call_xyz" || gotToolCall.ToolName != "get_weather" {
		t.Errorf("tool call: %+v", gotToolCall)
	}
	input, ok := gotToolCall.Input.(map[string]any)
	if !ok || input["location"] != "Tokyo" {
		t.Errorf("tool call input: %+v", gotToolCall.Input)
	}
	if !gotFinishStep {
		t.Error("missing FinishStepPart")
	}
	if !gotFinish {
		t.Error("missing FinishPart")
	}
}

func TestDoGenerate_NoModel(t *testing.T) {
	p := openai.NewCompletions(openai.WithAPIKey("k"))
	_, err := p.DoGenerate(context.Background(), sdk.GenerateParams{})
	if err == nil {
		t.Fatal("expected error for nil model")
	}
}

func TestDoStream_NoModel(t *testing.T) {
	p := openai.NewCompletions(openai.WithAPIKey("k"))
	_, err := p.DoStream(context.Background(), sdk.GenerateParams{})
	if err == nil {
		t.Fatal("expected error for nil model")
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

func newIntegrationProvider(t *testing.T) *openai.OpenAICompletionsProvider {
	t.Helper()
	apiKey := envOrSkip(t, "OPENAI_API_KEY")
	opts := []openai.OpenAICompletionsProviderOption{openai.WithAPIKey(apiKey)}
	if base := os.Getenv("OPENAI_BASE_URL"); base != "" {
		opts = append(opts, openai.WithBaseURL(base))
	}
	return openai.NewCompletions(opts...)
}

func integrationModel(t *testing.T) *sdk.Model {
	t.Helper()
	m := os.Getenv("OPENAI_MODEL")
	if m == "" {
		m = "gpt-4o-mini"
	}
	return &sdk.Model{ID: m}
}

func TestIntegration_DoGenerate(t *testing.T) {
	p := newIntegrationProvider(t)
	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model: integrationModel(t),
		Messages: []sdk.Message{{
			Role:  sdk.MessageRoleUser,
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
	sr, err := p.DoStream(context.Background(), sdk.GenerateParams{
		Model: integrationModel(t),
		Messages: []sdk.Message{{
			Role:  sdk.MessageRoleUser,
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

func TestMain(m *testing.M) {
	testutil.LoadEnv()
	os.Exit(m.Run())
}
