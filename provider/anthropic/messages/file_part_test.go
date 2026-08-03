package messages_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/memohai/twilight-ai/provider/anthropic/messages"
	"github.com/memohai/twilight-ai/sdk"
)

// TestDoGenerate_FilePartDocumentBlock is the golden test for native PDF
// input: an sdk.FilePart must reach the wire as an Anthropic document block,
// never as a text block carrying raw base64.
func TestDoGenerate_FilePartDocumentBlock(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&body)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":          "msg_test123",
			"type":        "message",
			"model":       "claude-sonnet-4-20250514",
			"role":        "assistant",
			"content":     []map[string]any{{"type": "text", "text": "A PDF."}},
			"stop_reason": "end_turn",
			"usage":       map[string]any{"input_tokens": 5, "output_tokens": 2},
		})
	}))
	defer srv.Close()

	p := messages.New(
		messages.WithAPIKey("test-key"),
		messages.WithBaseURL(srv.URL),
	)

	_, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model: &sdk.Model{ID: "claude-sonnet-4-20250514"},
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

	want := mustJSONValue(t, `[
		{
			"type": "document",
			"source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0xLjQ="},
			"title": "report.pdf"
		},
		{"type": "text", "text": "Summarize this PDF"}
	]`)

	msgs, ok := body["messages"].([]any)
	if !ok || len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %#v", body["messages"])
	}
	got := msgs[0].(map[string]any)["content"]
	if !reflect.DeepEqual(got, want) {
		t.Errorf("content mismatch\n got: %#v\nwant: %#v", got, want)
	}
}

// TestDoGenerate_FilePartDataURLTolerated verifies a data URL payload is
// stripped to bare base64 before hitting the wire.
func TestDoGenerate_FilePartDataURLTolerated(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&body)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":          "msg_test123",
			"type":        "message",
			"model":       "claude-sonnet-4-20250514",
			"role":        "assistant",
			"content":     []map[string]any{{"type": "text", "text": "ok"}},
			"stop_reason": "end_turn",
			"usage":       map[string]any{"input_tokens": 1, "output_tokens": 1},
		})
	}))
	defer srv.Close()

	p := messages.New(
		messages.WithAPIKey("test-key"),
		messages.WithBaseURL(srv.URL),
	)

	_, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model: &sdk.Model{ID: "claude-sonnet-4-20250514"},
		Messages: []sdk.Message{{
			Role: sdk.MessageRoleUser,
			Content: []sdk.MessagePart{
				sdk.FilePart{Data: "data:application/pdf;base64,JVBERi0xLjQ="},
			},
		}},
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}

	want := mustJSONValue(t, `[
		{
			"type": "document",
			"source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0xLjQ="}
		}
	]`)

	msgs := body["messages"].([]any)
	got := msgs[0].(map[string]any)["content"]
	if !reflect.DeepEqual(got, want) {
		t.Errorf("content mismatch\n got: %#v\nwant: %#v", got, want)
	}
}

func mustJSONValue(t *testing.T, s string) any {
	t.Helper()
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		t.Fatalf("bad golden JSON: %v", err)
	}
	return v
}
