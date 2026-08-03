package copilot_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/memohai/twilight-ai/provider/github/copilot"
	"github.com/memohai/twilight-ai/sdk"
)

// TestDoGenerate_FilePartOmittedNotice: Copilot has no confirmed native file
// input, so an sdk.FilePart must surface as an explicit omission marker —
// never as raw base64 masquerading as text.
func TestDoGenerate_FilePartOmittedNotice(t *testing.T) {
	token := testToken()
	var rawBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rawBody, _ = io.ReadAll(r.Body)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":      "copilotcmpl-test",
			"object":  "chat.completion",
			"created": 1700000000,
			"model":   "github-managed-model",
			"choices": []map[string]any{{
				"index":         0,
				"finish_reason": "stop",
				"message":       map[string]any{"role": "assistant", "content": "ok"},
			}},
			"usage": map[string]any{"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
		})
	}))
	defer srv.Close()

	p := copilot.New(
		copilot.WithGitHubToken(token),
		copilot.WithBaseURL(srv.URL),
	)

	_, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model: &sdk.Model{ID: "gpt-4o"},
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

	body := string(rawBody)
	if strings.Contains(body, "JVBERi0xLjQ=") {
		t.Errorf("raw base64 payload leaked into the request body")
	}
	wantNotice := "[file attachment omitted: this provider has no native file input; filename=report.pdf, mediaType=application/pdf]"
	if !strings.Contains(body, wantNotice) {
		t.Errorf("omission notice missing from request body:\n%s", body)
	}
}
