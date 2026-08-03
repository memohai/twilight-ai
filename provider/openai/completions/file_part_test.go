package completions_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/memohai/twilight-ai/provider/openai/completions"
	"github.com/memohai/twilight-ai/sdk"
)

// TestDoGenerate_FilePartFileContent is the golden test for native PDF input
// on Chat Completions: an sdk.FilePart must reach the wire as a "file" content
// part with a data URL, never as a text part carrying raw base64.
func TestDoGenerate_FilePartFileContent(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&body)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":      "chatcmpl-test",
			"object":  "chat.completion",
			"created": 1700000000,
			"model":   "gpt-4o-mini",
			"choices": []map[string]any{{
				"index":         0,
				"finish_reason": "stop",
				"message":       map[string]any{"role": "assistant", "content": "A PDF."},
			}},
			"usage": map[string]any{"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
		})
	}))
	defer srv.Close()

	p := completions.New(
		completions.WithAPIKey("test-key"),
		completions.WithBaseURL(srv.URL),
	)

	_, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model: &sdk.Model{ID: "gpt-4o-mini"},
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

	var want any
	if err := json.Unmarshal([]byte(`[
		{
			"type": "file",
			"file": {"filename": "report.pdf", "file_data": "data:application/pdf;base64,JVBERi0xLjQ="}
		},
		{"type": "text", "text": "Summarize this PDF"}
	]`), &want); err != nil {
		t.Fatalf("bad golden JSON: %v", err)
	}

	msgs, ok := body["messages"].([]any)
	if !ok || len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %#v", body["messages"])
	}
	got := msgs[0].(map[string]any)["content"]
	if !reflect.DeepEqual(got, want) {
		t.Errorf("content mismatch\n got: %#v\nwant: %#v", got, want)
	}
}
