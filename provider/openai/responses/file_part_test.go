package responses_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/memohai/twilight-ai/provider/openai/responses"
	"github.com/memohai/twilight-ai/sdk"
)

// TestResponsesDoGenerate_FilePartInputFile is the golden test for native PDF
// input on the Responses API: an sdk.FilePart must reach the wire as an
// "input_file" part with a data URL, never as input_text carrying raw base64.
func TestResponsesDoGenerate_FilePartInputFile(t *testing.T) {
	var body struct {
		Input []json.RawMessage `json:"input"`
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&body)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":         "resp_test123",
			"created_at": 1700000000,
			"model":      "gpt-4o-mini",
			"output": []map[string]any{{
				"type": "message",
				"id":   "msg_001",
				"role": "assistant",
				"content": []map[string]any{{
					"type":        "output_text",
					"text":        "A PDF.",
					"annotations": []any{},
				}},
			}},
			"usage": map[string]any{"input_tokens": 5, "output_tokens": 2},
		})
	}))
	defer srv.Close()

	p := responses.New(
		responses.WithAPIKey("test-key"),
		responses.WithBaseURL(srv.URL),
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

	if len(body.Input) != 1 {
		t.Fatalf("expected 1 input item, got %d", len(body.Input))
	}

	var got any
	if err := json.Unmarshal(body.Input[0], &got); err != nil {
		t.Fatalf("decode input item: %v", err)
	}

	var want any
	if err := json.Unmarshal([]byte(`{
		"role": "user",
		"content": [
			{
				"type": "input_file",
				"filename": "report.pdf",
				"file_data": "data:application/pdf;base64,JVBERi0xLjQ="
			},
			{"type": "input_text", "text": "Summarize this PDF"}
		]
	}`), &want); err != nil {
		t.Fatalf("bad golden JSON: %v", err)
	}

	if !reflect.DeepEqual(got, want) {
		t.Errorf("input mismatch\n got: %#v\nwant: %#v", got, want)
	}
}
