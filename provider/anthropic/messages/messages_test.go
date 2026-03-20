package messages_test

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
	"github.com/memohai/twilight-ai/provider/anthropic/messages"
	"github.com/memohai/twilight-ai/sdk"
)

// ---------- unit tests (mock server) ----------

func TestDoGenerate(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/messages" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get("x-api-key") != "test-key" {
			t.Errorf("unexpected x-api-key header: %s", r.Header.Get("x-api-key"))
		}
		if r.Header.Get("anthropic-version") != "2023-06-01" {
			t.Errorf("unexpected anthropic-version: %s", r.Header.Get("anthropic-version"))
		}

		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if body["model"] != "claude-sonnet-4-20250514" {
			t.Errorf("unexpected model: %v", body["model"])
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":    "msg_test123",
			"type":  "message",
			"model": "claude-sonnet-4-20250514",
			"role":  "assistant",
			"content": []map[string]any{
				{"type": "text", "text": "Hello!"},
			},
			"stop_reason": "end_turn",
			"usage": map[string]any{
				"input_tokens":  5,
				"output_tokens": 2,
			},
		})
	}))
	defer srv.Close()

	p := messages.New(
		messages.WithAPIKey("test-key"),
		messages.WithBaseURL(srv.URL),
	)

	model := &sdk.Model{ID: "claude-sonnet-4-20250514"}
	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model: model,
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
	if result.RawFinishReason != "end_turn" {
		t.Errorf("expected raw finish reason 'end_turn', got %q", result.RawFinishReason)
	}
	if result.Usage.InputTokens != 5 {
		t.Errorf("expected 5 input tokens, got %d", result.Usage.InputTokens)
	}
	if result.Usage.OutputTokens != 2 {
		t.Errorf("expected 2 output tokens, got %d", result.Usage.OutputTokens)
	}
	if result.Response.ID != "msg_test123" {
		t.Errorf("expected response id 'msg_test123', got %q", result.Response.ID)
	}
}

func TestDoGenerate_DefaultMaxTokens(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			MaxTokens int `json:"max_tokens"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if body.MaxTokens != 4096 {
			t.Fatalf("max_tokens: got %d, want 4096", body.MaxTokens)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id": "msg_default_max", "type": "message", "model": "claude-sonnet-4-20250514", "role": "assistant",
			"content":     []map[string]any{{"type": "text", "text": "OK"}},
			"stop_reason": "end_turn",
			"usage":       map[string]any{"input_tokens": 5, "output_tokens": 2},
		})
	}))
	defer srv.Close()

	p := messages.New(messages.WithAPIKey("test-key"), messages.WithBaseURL(srv.URL))

	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model:    &sdk.Model{ID: "claude-sonnet-4-20250514"},
		Messages: []sdk.Message{sdk.UserMessage("Hi")},
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}
	if result.Text != "OK" {
		t.Errorf("text: got %q", result.Text)
	}
}

func TestDoGenerate_DefaultMaxTokens_ThinkingBudgetReserveAnswerBudget(t *testing.T) {
	const budgetTokens = 50000
	const answerTokens = 4096

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			MaxTokens int `json:"max_tokens"`
			Thinking  struct {
				Type         string `json:"type"`
				BudgetTokens int    `json:"budget_tokens"`
			} `json:"thinking"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if body.MaxTokens != budgetTokens+answerTokens {
			t.Fatalf("max_tokens: got %d, want %d", body.MaxTokens, budgetTokens+answerTokens)
		}
		if body.Thinking.Type != "enabled" || body.Thinking.BudgetTokens != budgetTokens {
			t.Fatalf("thinking: got %+v", body.Thinking)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id": "msg_default_thinking_max", "type": "message", "model": "claude-sonnet-4-20250514", "role": "assistant",
			"content":     []map[string]any{{"type": "text", "text": "OK"}},
			"stop_reason": "end_turn",
			"usage":       map[string]any{"input_tokens": 5, "output_tokens": 2},
		})
	}))
	defer srv.Close()

	p := messages.New(
		messages.WithAPIKey("test-key"),
		messages.WithBaseURL(srv.URL),
		messages.WithThinking(messages.ThinkingConfig{
			Type:         "enabled",
			BudgetTokens: budgetTokens,
		}),
	)

	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model:    &sdk.Model{ID: "claude-sonnet-4-20250514"},
		Messages: []sdk.Message{sdk.UserMessage("Hi")},
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}
	if result.Text != "OK" {
		t.Errorf("text: got %q", result.Text)
	}
}

func TestDoGenerate_SystemMessage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			System   []map[string]any `json:"system"`
			Messages []map[string]any `json:"messages"`
		}
		json.NewDecoder(r.Body).Decode(&body)

		if len(body.System) != 1 {
			t.Fatalf("expected 1 system block, got %d", len(body.System))
		}
		if body.System[0]["text"] != "You are helpful." {
			t.Errorf("system text: got %q", body.System[0]["text"])
		}
		if len(body.Messages) != 1 || body.Messages[0]["role"] != "user" {
			t.Errorf("expected 1 user message, got %+v", body.Messages)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id": "msg_sys", "type": "message", "model": "claude-sonnet-4-20250514", "role": "assistant",
			"content":     []map[string]any{{"type": "text", "text": "OK"}},
			"stop_reason": "end_turn",
			"usage":       map[string]any{"input_tokens": 10, "output_tokens": 1},
		})
	}))
	defer srv.Close()

	p := messages.New(messages.WithAPIKey("test-key"), messages.WithBaseURL(srv.URL))
	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model:  &sdk.Model{ID: "claude-sonnet-4-20250514"},
		System: "You are helpful.",
		Messages: []sdk.Message{{
			Role:    sdk.MessageRoleUser,
			Content: []sdk.MessagePart{sdk.TextPart{Text: "Hi"}},
		}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if result.Text != "OK" {
		t.Errorf("text: got %q", result.Text)
	}
}

func TestDoGenerate_ToolCall(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Tools []struct {
				Name        string `json:"name"`
				Description string `json:"description"`
				InputSchema any    `json:"input_schema"`
			} `json:"tools"`
			ToolChoice map[string]any `json:"tool_choice"`
		}
		json.NewDecoder(r.Body).Decode(&body)

		if len(body.Tools) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(body.Tools))
		}
		if body.Tools[0].Name != "get_weather" {
			t.Errorf("tool name: got %q", body.Tools[0].Name)
		}
		if body.ToolChoice["type"] != "auto" {
			t.Errorf("tool_choice type: got %v", body.ToolChoice["type"])
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id": "msg_tool", "type": "message", "model": "claude-sonnet-4-20250514", "role": "assistant",
			"content": []map[string]any{
				{
					"type":  "tool_use",
					"id":    "toolu_abc123",
					"name":  "get_weather",
					"input": map[string]any{"location": "Beijing"},
				},
			},
			"stop_reason": "tool_use",
			"usage":       map[string]any{"input_tokens": 20, "output_tokens": 10},
		})
	}))
	defer srv.Close()

	p := messages.New(messages.WithAPIKey("test-key"), messages.WithBaseURL(srv.URL))

	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model: &sdk.Model{ID: "claude-sonnet-4-20250514"},
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
	if tc.ToolCallID != "toolu_abc123" {
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

		// Verify assistant message has tool_use block
		var assistantMsg struct {
			Role    string `json:"role"`
			Content []struct {
				Type  string `json:"type"`
				ID    string `json:"id,omitempty"`
				Name  string `json:"name,omitempty"`
				Input any    `json:"input,omitempty"`
			} `json:"content"`
		}
		json.Unmarshal(body.Messages[1], &assistantMsg)
		if assistantMsg.Role != "assistant" {
			t.Errorf("msg[1] role: got %q", assistantMsg.Role)
		}
		if len(assistantMsg.Content) != 1 || assistantMsg.Content[0].Type != "tool_use" {
			t.Errorf("msg[1] content: %+v", assistantMsg.Content)
		}
		if assistantMsg.Content[0].ID != "toolu_abc" {
			t.Errorf("msg[1] tool_use id: got %q", assistantMsg.Content[0].ID)
		}

		// Verify tool result in user message
		var userMsg struct {
			Role    string `json:"role"`
			Content []struct {
				Type      string `json:"type"`
				ToolUseID string `json:"tool_use_id,omitempty"`
				Content   any    `json:"content,omitempty"`
			} `json:"content"`
		}
		json.Unmarshal(body.Messages[2], &userMsg)
		if userMsg.Role != "user" {
			t.Errorf("msg[2] role: got %q, want user", userMsg.Role)
		}
		if len(userMsg.Content) != 1 || userMsg.Content[0].Type != "tool_result" {
			t.Errorf("msg[2] content: %+v", userMsg.Content)
		}
		if userMsg.Content[0].ToolUseID != "toolu_abc" {
			t.Errorf("msg[2] tool_use_id: got %q", userMsg.Content[0].ToolUseID)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id": "msg_2", "type": "message", "model": "claude-sonnet-4-20250514", "role": "assistant",
			"content":     []map[string]any{{"type": "text", "text": "It's sunny in Beijing."}},
			"stop_reason": "end_turn",
			"usage":       map[string]any{"input_tokens": 30, "output_tokens": 8},
		})
	}))
	defer srv.Close()

	p := messages.New(messages.WithAPIKey("test-key"), messages.WithBaseURL(srv.URL))

	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model: &sdk.Model{ID: "claude-sonnet-4-20250514"},
		Messages: []sdk.Message{
			{
				Role:    sdk.MessageRoleUser,
				Content: []sdk.MessagePart{sdk.TextPart{Text: "Weather?"}},
			},
			{
				Role: sdk.MessageRoleAssistant,
				Content: []sdk.MessagePart{sdk.ToolCallPart{
					ToolCallID: "toolu_abc",
					ToolName:   "get_weather",
					Input:      map[string]any{"location": "Beijing"},
				}},
			},
			{
				Role: sdk.MessageRoleTool,
				Content: []sdk.MessagePart{sdk.ToolResultPart{
					ToolCallID: "toolu_abc",
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

func TestDoGenerate_Thinking(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id": "msg_think", "type": "message", "model": "claude-sonnet-4-20250514", "role": "assistant",
			"content": []map[string]any{
				{"type": "thinking", "thinking": "Let me think about this..."},
				{"type": "text", "text": "The answer is 4."},
			},
			"stop_reason": "end_turn",
			"usage":       map[string]any{"input_tokens": 5, "output_tokens": 15},
		})
	}))
	defer srv.Close()

	p := messages.New(messages.WithAPIKey("k"), messages.WithBaseURL(srv.URL))
	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model:    &sdk.Model{ID: "claude-sonnet-4-20250514"},
		Messages: []sdk.Message{sdk.UserMessage("2+2?")},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if result.Text != "The answer is 4." {
		t.Errorf("text: got %q", result.Text)
	}
	if result.Reasoning != "Let me think about this..." {
		t.Errorf("reasoning: got %q", result.Reasoning)
	}
}

func TestDoGenerate_NoModel(t *testing.T) {
	p := messages.New(messages.WithAPIKey("k"))
	_, err := p.DoGenerate(context.Background(), sdk.GenerateParams{})
	if err == nil {
		t.Fatal("expected error for nil model")
	}
}

func TestDoStream_NoModel(t *testing.T) {
	p := messages.New(messages.WithAPIKey("k"))
	_, err := p.DoStream(context.Background(), sdk.GenerateParams{})
	if err == nil {
		t.Fatal("expected error for nil model")
	}
}

// ---------- streaming tests ----------

func TestDoStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		flusher := w.(http.Flusher)

		events := []struct{ event, data string }{
			{"message_start", `{"type":"message_start","message":{"id":"msg_s1","type":"message","model":"claude-sonnet-4-20250514","role":"assistant","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}`},
			{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}`},
			{"content_block_stop", `{"type":"content_block_stop","index":0}`},
			{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`},
			{"message_stop", `{"type":"message_stop"}`},
		}

		for _, e := range events {
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", e.event, e.data)
			flusher.Flush()
		}
	}))
	defer srv.Close()

	p := messages.New(
		messages.WithAPIKey("test-key"),
		messages.WithBaseURL(srv.URL),
	)

	sr, err := p.DoStream(context.Background(), sdk.GenerateParams{
		Model: &sdk.Model{ID: "claude-sonnet-4-20250514"},
		Messages: []sdk.Message{{
			Role:    sdk.MessageRoleUser,
			Content: []sdk.MessagePart{sdk.TextPart{Text: "Hi"}},
		}},
	})
	if err != nil {
		t.Fatalf("DoStream failed: %v", err)
	}

	var collected string
	var gotStart, gotFinish, gotTextStart, gotTextEnd bool
	for part := range sr.Stream {
		switch p := part.(type) {
		case *sdk.StartPart:
			gotStart = true
		case *sdk.TextStartPart:
			gotTextStart = true
		case *sdk.TextDeltaPart:
			collected += p.Text
		case *sdk.TextEndPart:
			gotTextEnd = true
		case *sdk.FinishPart:
			gotFinish = true
			if p.FinishReason != sdk.FinishReasonStop {
				t.Errorf("expected stop, got %q", p.FinishReason)
			}
		case *sdk.ErrorPart:
			t.Fatalf("error: %v", p.Error)
		}
	}

	if !gotStart {
		t.Error("missing StartPart")
	}
	if !gotTextStart {
		t.Error("missing TextStartPart")
	}
	if !gotTextEnd {
		t.Error("missing TextEndPart")
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

		events := []struct{ event, data string }{
			{"message_start", `{"type":"message_start","message":{"id":"msg_tc","type":"message","model":"claude-sonnet-4-20250514","role":"assistant","content":[],"usage":{"input_tokens":15,"output_tokens":0}}}`},
			{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_xyz","name":"get_weather"}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"location\""}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":":\"Tokyo\"}"}}`},
			{"content_block_stop", `{"type":"content_block_stop","index":0}`},
			{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":8}}`},
			{"message_stop", `{"type":"message_stop"}`},
		}

		for _, e := range events {
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", e.event, e.data)
			flusher.Flush()
		}
	}))
	defer srv.Close()

	p := messages.New(messages.WithAPIKey("test-key"), messages.WithBaseURL(srv.URL))

	sr, err := p.DoStream(context.Background(), sdk.GenerateParams{
		Model: &sdk.Model{ID: "claude-sonnet-4-20250514"},
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
		gotFinishStep bool
		gotFinish     bool
	)

	for part := range sr.Stream {
		switch p := part.(type) {
		case *sdk.ToolInputStartPart:
			gotInputStart = true
			if p.ToolName != "get_weather" {
				t.Errorf("input start tool name: got %q", p.ToolName)
			}
			if p.ID != "toolu_xyz" {
				t.Errorf("input start id: got %q", p.ID)
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
	if gotToolCall.ToolCallID != "toolu_xyz" || gotToolCall.ToolName != "get_weather" {
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

func TestDoStream_Thinking(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)

		events := []struct{ event, data string }{
			{"message_start", `{"type":"message_start","message":{"id":"msg_th","type":"message","model":"claude-sonnet-4-20250514","role":"assistant","content":[],"usage":{"input_tokens":8,"output_tokens":0}}}`},
			{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think"}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"..."}}`},
			{"content_block_stop", `{"type":"content_block_stop","index":0}`},
			{"content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"The answer is 4."}}`},
			{"content_block_stop", `{"type":"content_block_stop","index":1}`},
			{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20}}`},
			{"message_stop", `{"type":"message_stop"}`},
		}

		for _, e := range events {
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", e.event, e.data)
			flusher.Flush()
		}
	}))
	defer srv.Close()

	p := messages.New(messages.WithAPIKey("k"), messages.WithBaseURL(srv.URL))
	sr, err := p.DoStream(context.Background(), sdk.GenerateParams{
		Model:    &sdk.Model{ID: "claude-sonnet-4-20250514"},
		Messages: []sdk.Message{sdk.UserMessage("2+2?")},
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}

	var reasoning, text string
	var gotReasoningStart, gotReasoningEnd bool
	for part := range sr.Stream {
		switch p := part.(type) {
		case *sdk.ReasoningStartPart:
			gotReasoningStart = true
		case *sdk.ReasoningDeltaPart:
			reasoning += p.Text
		case *sdk.ReasoningEndPart:
			gotReasoningEnd = true
		case *sdk.TextDeltaPart:
			text += p.Text
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
	if reasoning != "Let me think..." {
		t.Errorf("reasoning: got %q", reasoning)
	}
	if text != "The answer is 4." {
		t.Errorf("text: got %q", text)
	}
}

func TestDoGenerate_ReasoningFromOtherProvider(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []json.RawMessage `json:"messages"`
		}
		json.NewDecoder(r.Body).Decode(&body)

		if len(body.Messages) != 2 {
			t.Fatalf("expected 2 messages, got %d", len(body.Messages))
		}

		var assistantMsg struct {
			Role    string           `json:"role"`
			Content []map[string]any `json:"content"`
		}
		json.Unmarshal(body.Messages[0], &assistantMsg)
		if assistantMsg.Role != "assistant" {
			t.Errorf("msg[0] role: got %q", assistantMsg.Role)
		}
		for _, block := range assistantMsg.Content {
			if block["type"] == "thinking" {
				t.Error("reasoning from other provider should not become a thinking block")
			}
		}
		if len(assistantMsg.Content) != 2 {
			t.Fatalf("expected 2 text blocks (reasoning fallback + text), got %+v", assistantMsg.Content)
		}
		if assistantMsg.Content[0]["type"] != "text" || assistantMsg.Content[0]["text"] != "thinking from gemini" {
			t.Errorf("expected reasoning fallback as text, got %+v", assistantMsg.Content[0])
		}
		if assistantMsg.Content[1]["type"] != "text" || assistantMsg.Content[1]["text"] != "The answer" {
			t.Errorf("expected original text block, got %+v", assistantMsg.Content[1])
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id": "msg_cross", "type": "message", "model": "claude-sonnet-4-20250514", "role": "assistant",
			"content":     []map[string]any{{"type": "text", "text": "OK"}},
			"stop_reason": "end_turn",
			"usage":       map[string]any{"input_tokens": 10, "output_tokens": 1},
		})
	}))
	defer srv.Close()

	p := messages.New(messages.WithAPIKey("test-key"), messages.WithBaseURL(srv.URL))
	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model: &sdk.Model{ID: "claude-sonnet-4-20250514"},
		Messages: []sdk.Message{
			{
				Role: sdk.MessageRoleAssistant,
				Content: []sdk.MessagePart{
					sdk.ReasoningPart{
						Text:             "thinking from gemini",
						ProviderMetadata: map[string]any{"google": map[string]any{"thoughtSignature": "abc123"}},
					},
					sdk.TextPart{Text: "The answer"},
				},
			},
			sdk.UserMessage("follow up"),
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if result.Text != "OK" {
		t.Errorf("text: got %q", result.Text)
	}
}

func TestDoGenerate_CacheUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id": "msg_cache", "type": "message", "model": "claude-sonnet-4-20250514", "role": "assistant",
			"content":     []map[string]any{{"type": "text", "text": "cached"}},
			"stop_reason": "end_turn",
			"usage": map[string]any{
				"input_tokens":                10,
				"output_tokens":               5,
				"cache_creation_input_tokens": 100,
				"cache_read_input_tokens":     50,
			},
		})
	}))
	defer srv.Close()

	p := messages.New(messages.WithAPIKey("k"), messages.WithBaseURL(srv.URL))
	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model:    &sdk.Model{ID: "claude-sonnet-4-20250514"},
		Messages: []sdk.Message{sdk.UserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}

	if result.Usage.CachedInputTokens != 50 {
		t.Errorf("CachedInputTokens: got %d, want 50", result.Usage.CachedInputTokens)
	}
	if result.Usage.InputTokenDetails.CacheReadTokens != 50 {
		t.Errorf("CacheReadTokens: got %d, want 50", result.Usage.InputTokenDetails.CacheReadTokens)
	}
	if result.Usage.InputTokenDetails.CacheWriteTokens != 100 {
		t.Errorf("CacheWriteTokens: got %d, want 100", result.Usage.InputTokenDetails.CacheWriteTokens)
	}
}

func TestDoGenerate_ErrorResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    "invalid_request_error",
				"message": "max_tokens is required",
			},
		})
	}))
	defer srv.Close()

	p := messages.New(messages.WithAPIKey("k"), messages.WithBaseURL(srv.URL))
	_, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model:    &sdk.Model{ID: "claude-sonnet-4-20250514"},
		Messages: []sdk.Message{sdk.UserMessage("hi")},
	})
	if err == nil {
		t.Fatal("expected error for bad request")
	}
}

// ---------- integration tests ----------

func envOrSkip(t *testing.T, key string) string {
	t.Helper()
	v := os.Getenv(key)
	if v == "" {
		t.Skipf("skipping: %s not set", key)
	}
	return v
}

func baseOpts(t *testing.T) []messages.Option {
	t.Helper()
	apiKey := envOrSkip(t, "ANTHROPIC_API_KEY")
	opts := []messages.Option{}
	if os.Getenv("ANTHROPIC_AUTH_MODE") == "bearer" {
		opts = append(opts, messages.WithAuthToken(apiKey))
	} else {
		opts = append(opts, messages.WithAPIKey(apiKey))
	}
	if base := os.Getenv("ANTHROPIC_BASE_URL"); base != "" {
		opts = append(opts, messages.WithBaseURL(base))
	}
	return opts
}

func newIntegrationProvider(t *testing.T) *messages.Provider {
	t.Helper()
	return messages.New(baseOpts(t)...)
}

func newReasoningProvider(t *testing.T) *messages.Provider {
	t.Helper()
	opts := baseOpts(t)
	opts = append(opts, messages.WithThinking(messages.ThinkingConfig{
		Type:         "enabled",
		BudgetTokens: 4000,
	}))
	return messages.New(opts...)
}

func integrationModel(t *testing.T) *sdk.Model {
	t.Helper()
	m := os.Getenv("ANTHROPIC_MODEL")
	if m == "" {
		m = "claude-sonnet-4-20250514"
	}
	return &sdk.Model{ID: m}
}

func reasoningModel(t *testing.T) *sdk.Model {
	t.Helper()
	m := os.Getenv("ANTHROPIC_REASONING_MODEL")
	if m == "" {
		t.Skip("skipping: ANTHROPIC_REASONING_MODEL not set")
	}
	return &sdk.Model{ID: m}
}

func TestIntegration_DoGenerate(t *testing.T) {
	p := newIntegrationProvider(t)
	maxTokens := 100
	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model:     integrationModel(t),
		MaxTokens: &maxTokens,
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
	maxTokens := 100
	sr, err := p.DoStream(context.Background(), sdk.GenerateParams{
		Model:     integrationModel(t),
		MaxTokens: &maxTokens,
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

func TestIntegration_DoGenerate_Reasoning(t *testing.T) {
	p := newReasoningProvider(t)
	model := reasoningModel(t)
	maxTokens := 8000
	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model:     model,
		MaxTokens: &maxTokens,
		Messages: []sdk.Message{{
			Role:    sdk.MessageRoleUser,
			Content: []sdk.MessagePart{sdk.TextPart{Text: "What is 15 * 37? Think step by step."}},
		}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	t.Logf("model=%s", model.ID)
	t.Logf("text=%q", result.Text)
	t.Logf("reasoning=%q", result.Reasoning)
	t.Logf("finish=%s tokens=%d/%d", result.FinishReason,
		result.Usage.InputTokens, result.Usage.OutputTokens)

	if result.Text == "" {
		t.Error("expected non-empty text")
	}
	if result.Reasoning == "" {
		t.Error("expected non-empty reasoning from thinking model")
	}
}

func TestIntegration_DoStream_Reasoning(t *testing.T) {
	p := newReasoningProvider(t)
	model := reasoningModel(t)
	maxTokens := 8000
	sr, err := p.DoStream(context.Background(), sdk.GenerateParams{
		Model:     model,
		MaxTokens: &maxTokens,
		Messages: []sdk.Message{{
			Role:    sdk.MessageRoleUser,
			Content: []sdk.MessagePart{sdk.TextPart{Text: "What is 15 * 37? Think step by step."}},
		}},
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}

	var text, reasoning string
	var gotReasoningStart, gotReasoningEnd bool
	for part := range sr.Stream {
		switch p := part.(type) {
		case *sdk.ReasoningStartPart:
			gotReasoningStart = true
			t.Log("--- reasoning start ---")
		case *sdk.ReasoningDeltaPart:
			reasoning += p.Text
		case *sdk.ReasoningEndPart:
			gotReasoningEnd = true
			t.Logf("--- reasoning end (len=%d) ---", len(reasoning))
		case *sdk.TextDeltaPart:
			text += p.Text
		case *sdk.ErrorPart:
			t.Fatalf("stream error: %v", p.Error)
		case *sdk.FinishPart:
			t.Logf("finish=%s total_usage=%+v", p.FinishReason, p.TotalUsage)
		}
	}

	t.Logf("model=%s", model.ID)
	t.Logf("streamed text: %q", text)
	t.Logf("reasoning length: %d chars", len(reasoning))

	if text == "" {
		t.Error("expected non-empty streamed text")
	}
	if reasoning == "" {
		t.Error("expected non-empty reasoning from thinking model")
	}
	if !gotReasoningStart {
		t.Error("missing ReasoningStartPart")
	}
	if !gotReasoningEnd {
		t.Error("missing ReasoningEndPart")
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
			"data": []map[string]any{
				{"id": "claude-sonnet-4-20250514", "type": "model", "display_name": "Claude Sonnet 4"},
				{"id": "claude-3-5-haiku-20241022", "type": "model", "display_name": "Claude 3.5 Haiku"},
			},
			"has_more": false,
		})
	}))
	defer srv.Close()

	p := messages.New(
		messages.WithAPIKey("test-key"),
		messages.WithBaseURL(srv.URL),
	)

	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels failed: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(models))
	}
	if models[0].ID != "claude-sonnet-4-20250514" {
		t.Errorf("expected first model id 'claude-sonnet-4-20250514', got %q", models[0].ID)
	}
	if models[0].DisplayName != "Claude Sonnet 4" {
		t.Errorf("expected display name 'Claude Sonnet 4', got %q", models[0].DisplayName)
	}
}

func TestProviderTest_OK(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{}, "has_more": false,
		})
	}))
	defer srv.Close()

	p := messages.New(
		messages.WithAPIKey("test-key"),
		messages.WithBaseURL(srv.URL),
	)

	result := p.Test(context.Background())
	if result.Status != sdk.ProviderStatusOK {
		t.Errorf("expected status OK, got %q", result.Status)
	}
}

func TestProviderTest_Unhealthy(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{"message": "invalid x-api-key"},
		})
	}))
	defer srv.Close()

	p := messages.New(
		messages.WithAPIKey("bad-key"),
		messages.WithBaseURL(srv.URL),
	)

	result := p.Test(context.Background())
	if result.Status != sdk.ProviderStatusUnhealthy {
		t.Errorf("expected status Unhealthy, got %q", result.Status)
	}
}

func TestProviderTest_Unreachable(t *testing.T) {
	p := messages.New(
		messages.WithAPIKey("test-key"),
		messages.WithBaseURL("http://127.0.0.1:1"),
	)

	result := p.Test(context.Background())
	if result.Status != sdk.ProviderStatusUnreachable {
		t.Errorf("expected status Unreachable, got %q", result.Status)
	}
}

func TestTestModel_Supported(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models/claude-sonnet-4-20250514" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id": "claude-sonnet-4-20250514", "type": "model", "display_name": "Claude Sonnet 4",
		})
	}))
	defer srv.Close()

	p := messages.New(
		messages.WithAPIKey("test-key"),
		messages.WithBaseURL(srv.URL),
	)

	result, err := p.TestModel(context.Background(), "claude-sonnet-4-20250514")
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

	p := messages.New(
		messages.WithAPIKey("test-key"),
		messages.WithBaseURL(srv.URL),
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
