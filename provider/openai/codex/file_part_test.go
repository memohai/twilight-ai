package codex_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/memohai/twilight-ai/provider/openai/codex"
	"github.com/memohai/twilight-ai/sdk"
)

// TestDoGenerate_FilePartOmittedNotice: Codex has no confirmed native file
// input, so an sdk.FilePart must surface as an explicit omission marker —
// never as raw base64 masquerading as input_text.
func TestDoGenerate_FilePartOmittedNotice(t *testing.T) {
	var rawBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rawBody, _ = io.ReadAll(r.Body)

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
		_, _ = w.Write([]byte("data: {\"response\":{\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n"))
	}))
	defer srv.Close()

	p := codex.New(
		codex.WithAccessToken("token-123"),
		codex.WithAccountID("acct_123"),
		codex.WithBaseURL(srv.URL),
	)

	_, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model: &sdk.Model{ID: "gpt-5.2"},
		Messages: []sdk.Message{{
			Role: sdk.MessageRoleUser,
			Content: []sdk.MessagePart{
				sdk.FilePart{Data: "JVBERi0xLjQ=", MediaType: "application/pdf", Filename: "report.pdf"},
				sdk.TextPart{Text: "Summarize this PDF"},
			},
		}},
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}

	var body struct {
		Input []json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(rawBody, &body); err != nil {
		t.Fatalf("decode request body: %v", err)
	}
	raw := string(rawBody)
	if strings.Contains(raw, "JVBERi0xLjQ=") {
		t.Errorf("raw base64 payload leaked into the request body")
	}
	wantNotice := "[file attachment omitted: this provider has no native file input; filename=report.pdf, mediaType=application/pdf]"
	if !strings.Contains(raw, wantNotice) {
		t.Errorf("omission notice missing from request body:\n%s", raw)
	}
}
