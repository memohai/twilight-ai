package generativeai_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/memohai/twilight-ai/provider/google/generativeai"
	"github.com/memohai/twilight-ai/sdk"
)

// TestDoGenerate_FilePartInlineData is the golden test for native PDF input
// on Gemini: an sdk.FilePart must reach the wire as inlineData with bare
// base64 — a data URL payload is stripped, and an unlabeled payload defaults
// to application/pdf rather than octet-stream.
func TestDoGenerate_FilePartInlineData(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&body)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{{
				"content": map[string]any{
					"role":  "model",
					"parts": []map[string]any{{"text": "A PDF."}},
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

	_, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model: p.ChatModel("gemini-2.0-flash"),
		Messages: []sdk.Message{{
			Role: sdk.MessageRoleUser,
			Content: []sdk.MessagePart{
				sdk.FilePart{Data: "data:application/pdf;base64,JVBERi0xLjQ="},
				sdk.TextPart{Text: "Summarize this PDF"},
			},
		}},
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}

	var want any
	if err := json.Unmarshal([]byte(`[
		{"inlineData": {"mimeType": "application/pdf", "data": "JVBERi0xLjQ="}},
		{"text": "Summarize this PDF"}
	]`), &want); err != nil {
		t.Fatalf("bad golden JSON: %v", err)
	}

	contents, ok := body["contents"].([]any)
	if !ok || len(contents) != 1 {
		t.Fatalf("expected 1 content, got %#v", body["contents"])
	}
	got := contents[0].(map[string]any)["parts"]
	if !reflect.DeepEqual(got, want) {
		t.Errorf("parts mismatch\n got: %#v\nwant: %#v", got, want)
	}
}
