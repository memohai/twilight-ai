package generativeai_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/memohai/twilight-ai/provider/google/generativeai"
	"github.com/memohai/twilight-ai/sdk"
)

// Gemini requires user/model turn alternation, and every functionResponse
// answering one functionCall turn must live in a single user turn.
// Consecutive tool messages — the shape a paused-then-resumed run produces
// (partial results committed at the pause, completion appended on resume) —
// must merge into one user content.
func TestConsecutiveToolMessagesMergeIntoOneUserTurn(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Contents []struct {
				Role  string `json:"role"`
				Parts []struct {
					FunctionResponse *struct {
						Name string `json:"name"`
					} `json:"functionResponse"`
					FunctionCall *struct {
						Name string `json:"name"`
					} `json:"functionCall"`
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"contents"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}

		wantRoles := []string{"user", "model", "user"}
		if len(body.Contents) != len(wantRoles) {
			t.Fatalf("content count: got %d, want %d (%+v)", len(body.Contents), len(wantRoles), body.Contents)
		}
		for i, want := range wantRoles {
			if body.Contents[i].Role != want {
				t.Fatalf("content %d role: got %q, want %q", i, body.Contents[i].Role, want)
			}
		}

		// The two tool messages must land in ONE trailing user turn with both
		// function responses.
		last := body.Contents[2]
		var fnNames []string
		for _, p := range last.Parts {
			if p.FunctionResponse != nil {
				fnNames = append(fnNames, p.FunctionResponse.Name)
			}
		}
		if len(fnNames) != 2 || fnNames[0] != "notify" || fnNames[1] != "deploy" {
			t.Fatalf("merged function responses: %v, want [notify deploy]", fnNames)
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{{
				"content":      map[string]any{"role": "model", "parts": []map[string]any{{"text": "ok"}}},
				"finishReason": "STOP",
			}},
			"usageMetadata": map[string]any{"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
		})
	}))
	defer srv.Close()

	p := generativeai.New(generativeai.WithAPIKey("k"), generativeai.WithBaseURL(srv.URL))
	_, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model: p.ChatModel("gemini-test"),
		Messages: []sdk.Message{
			sdk.UserMessage("deploy and notify"),
			{Role: sdk.MessageRoleAssistant, Content: []sdk.MessagePart{
				sdk.ToolCallPart{ToolCallID: "c1", ToolName: "notify", Input: map[string]any{}},
				sdk.ToolCallPart{ToolCallID: "c2", ToolName: "deploy", Input: map[string]any{}},
			}},
			sdk.ToolMessage(sdk.ToolResultPart{ToolCallID: "c1", ToolName: "notify", Result: "sent"}),
			sdk.ToolMessage(sdk.ToolResultPart{ToolCallID: "c2", ToolName: "deploy", Result: "deployed"}),
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
}

// A pause can resolve calls out of order (resolved sibling first, deferred
// call completed later). Gemini matches functionResponses to functionCalls
// by name and position, so the merged user turn must present responses in
// the model turn's call order even when the tool names repeat.
func TestMergedFunctionResponsesFollowCallOrder(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Contents []struct {
				Role  string `json:"role"`
				Parts []struct {
					FunctionResponse *struct {
						Name     string `json:"name"`
						Response struct {
							Content any `json:"content"`
						} `json:"response"`
					} `json:"functionResponse"`
				} `json:"parts"`
			} `json:"contents"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		last := body.Contents[len(body.Contents)-1]
		if last.Role != "user" {
			t.Fatalf("last turn role: %q", last.Role)
		}
		var outputs []any
		for _, p := range last.Parts {
			if p.FunctionResponse != nil {
				outputs = append(outputs, p.FunctionResponse.Response.Content)
			}
		}
		// Call order was c1 (deferred, resolved later) then c2 (resolved at
		// pause). Responses must arrive in call order: c1's output first.
		if len(outputs) != 2 || outputs[0] != "c1-late" || outputs[1] != "c2-early" {
			t.Fatalf("merged responses out of call order: %v, want [c1-late c2-early]", outputs)
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{{
				"content":      map[string]any{"role": "model", "parts": []map[string]any{{"text": "ok"}}},
				"finishReason": "STOP",
			}},
			"usageMetadata": map[string]any{"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
		})
	}))
	defer srv.Close()

	p := generativeai.New(generativeai.WithAPIKey("k"), generativeai.WithBaseURL(srv.URL))
	_, err := p.DoGenerate(context.Background(), sdk.GenerateParams{
		Model: p.ChatModel("gemini-test"),
		Messages: []sdk.Message{
			sdk.UserMessage("deploy twice"),
			{Role: sdk.MessageRoleAssistant, Content: []sdk.MessagePart{
				// Same tool name on both calls: positional association matters.
				sdk.ToolCallPart{ToolCallID: "c1", ToolName: "deploy", Input: map[string]any{}},
				sdk.ToolCallPart{ToolCallID: "c2", ToolName: "deploy", Input: map[string]any{}},
			}},
			// Pause resolved c2 first; resume completed c1 later.
			sdk.ToolMessage(sdk.ToolResultPart{ToolCallID: "c2", ToolName: "deploy", Result: "c2-early"}),
			sdk.ToolMessage(sdk.ToolResultPart{ToolCallID: "c1", ToolName: "deploy", Result: "c1-late"}),
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
}
