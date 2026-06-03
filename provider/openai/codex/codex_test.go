package codex_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/memohai/twilight-ai/provider/openai/codex"
	"github.com/memohai/twilight-ai/sdk"
)

func TestCodexDoGenerate_RequestShapeAndStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/codex/responses" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer token-123" {
			t.Fatalf("unexpected auth header: %s", got)
		}
		if got := r.Header.Get("chatgpt-account-id"); got != "acct_123" {
			t.Fatalf("unexpected account header: %s", got)
		}
		if got := r.Header.Get("OpenAI-Beta"); got != "responses=experimental" {
			t.Fatalf("unexpected beta header: %s", got)
		}

		var body struct {
			Model        string            `json:"model"`
			Instructions string            `json:"instructions"`
			Input        []json.RawMessage `json:"input"`
			Store        bool              `json:"store"`
			Stream       bool              `json:"stream"`
			Include      []string          `json:"include"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode body: %v", err)
		}

		if body.Model != "gpt-5.2" {
			t.Fatalf("unexpected model: %s", body.Model)
		}
		if body.Instructions != "You are helpful." {
			t.Fatalf("unexpected instructions: %q", body.Instructions)
		}
		if body.Store {
			t.Fatalf("store should be false")
		}
		if !body.Stream {
			t.Fatalf("stream should be true")
		}
		if len(body.Include) != 1 || body.Include[0] != "reasoning.encrypted_content" {
			t.Fatalf("unexpected include: %#v", body.Include)
		}
		if len(body.Input) != 1 {
			t.Fatalf("expected 1 input item, got %d", len(body.Input))
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: response.created\n"))
		_, _ = w.Write([]byte("data: {\"response\":{\"id\":\"resp_123\",\"created_at\":1700000000,\"model\":\"gpt-5.2\"}}\n\n"))
		_, _ = w.Write([]byte("event: response.output_item.added\n"))
		_, _ = w.Write([]byte("data: {\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\"}}\n\n"))
		_, _ = w.Write([]byte("event: response.output_text.delta\n"))
		_, _ = w.Write([]byte("data: {\"item_id\":\"msg_1\",\"delta\":\"Hello\"}\n\n"))
		_, _ = w.Write([]byte("event: response.output_item.done\n"))
		_, _ = w.Write([]byte("data: {\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\"}}\n\n"))
		_, _ = w.Write([]byte("event: response.completed\n"))
		_, _ = w.Write([]byte("data: {\"response\":{\"usage\":{\"input_tokens\":5,\"output_tokens\":3}}}\n\n"))
	}))
	defer srv.Close()

	p := codex.New(
		codex.WithAccessToken("token-123"),
		codex.WithAccountID("acct_123"),
		codex.WithBaseURL(srv.URL),
	)

	result, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model:    p.ChatModel("gpt-5.2"),
		System:   "You are helpful.",
		Messages: []sdk.Message{sdk.UserMessage("Hi")},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if result.Text != "Hello" {
		t.Fatalf("unexpected text: %q", result.Text)
	}
	if result.FinishReason != sdk.FinishReasonStop {
		t.Fatalf("unexpected finish reason: %s", result.FinishReason)
	}
	if result.Usage.InputTokens != 5 || result.Usage.OutputTokens != 3 {
		t.Fatalf("unexpected usage: %+v", result.Usage)
	}
}

func TestCodexDoGenerate_MapsMaxReasoningEffortToXHigh(t *testing.T) {
	var body struct {
		Reasoning *struct {
			Effort string `json:"effort"`
		} `json:"reasoning"`
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: response.created\n"))
		_, _ = w.Write([]byte("data: {\"response\":{\"id\":\"resp_123\",\"created_at\":1700000000,\"model\":\"gpt-5.2\"}}\n\n"))
		_, _ = w.Write([]byte("event: response.output_item.added\n"))
		_, _ = w.Write([]byte("data: {\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\"}}\n\n"))
		_, _ = w.Write([]byte("event: response.output_text.delta\n"))
		_, _ = w.Write([]byte("data: {\"item_id\":\"msg_1\",\"delta\":\"ok\"}\n\n"))
		_, _ = w.Write([]byte("event: response.output_item.done\n"))
		_, _ = w.Write([]byte("data: {\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\"}}\n\n"))
		_, _ = w.Write([]byte("event: response.completed\n"))
		_, _ = w.Write([]byte("data: {\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n"))
	}))
	defer srv.Close()

	p := codex.New(codex.WithAccessToken("token-123"), codex.WithBaseURL(srv.URL))
	effort := "max"
	_, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model:           p.ChatModel("gpt-5.2"),
		Messages:        []sdk.Message{sdk.UserMessage("hi")},
		ReasoningEffort: &effort,
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if body.Reasoning == nil || body.Reasoning.Effort != "xhigh" {
		t.Fatalf("reasoning.effort: got %#v, want xhigh", body.Reasoning)
	}
}

func TestCodexListModels(t *testing.T) {
	p := codex.New(codex.WithAccessToken("token-123"), codex.WithAccountID("acct_123"))
	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}
	if len(models) != 6 {
		t.Fatalf("unexpected model count: %d", len(models))
	}
	if models[0].ID != "gpt-5.2" {
		t.Fatalf("unexpected first model: %s", models[0].ID)
	}
}
