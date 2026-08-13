package sdk_test

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/memohai/twilight-ai/sdk"
)

func completionCall(id string) sdk.ToolCall {
	switch id {
	case "c1":
		return sdk.ToolCall{
			ToolCallID: "c1", ToolName: "lookup",
			Input:            map[string]any{"service": "api"},
			ProviderMetadata: map[string]any{"trace": map[string]any{"position": float64(1)}},
		}
	case "c2":
		return sdk.ToolCall{
			ToolCallID: "c2", ToolName: "deploy",
			Input:            map[string]any{"env": "prod", "regions": []any{"nrt", "iad"}},
			ProviderMetadata: map[string]any{"trace": map[string]any{"position": float64(2)}},
		}
	case "c3":
		return sdk.ToolCall{
			ToolCallID: "c3", ToolName: "ask_user",
			Input:            map[string]any{"question": "Proceed?"},
			ProviderMetadata: map[string]any{"trace": map[string]any{"position": float64(3)}},
		}
	default:
		panic("unknown completion call: " + id)
	}
}

func completionCallPart(id string) sdk.ToolCallPart {
	call := completionCall(id)
	return sdk.ToolCallPart{
		ToolCallID:       call.ToolCallID,
		ToolName:         call.ToolName,
		Input:            call.Input,
		ProviderMetadata: call.ProviderMetadata,
	}
}

func completionDeferred(id string) sdk.DeferredToolApproval {
	approval := sdk.ToolApprovalResult{
		Decision:   sdk.ToolApprovalDecisionDeferred,
		ApprovalID: "approval-" + id,
	}
	if id == "c2" {
		approval.Metadata = map[string]any{"workspace": "release"}
	}
	return sdk.DeferredToolApproval{ToolCall: completionCall(id), Approval: approval}
}

func completionPause() sdk.ToolApprovalPause {
	return sdk.ToolApprovalPause{
		BatchID: "apbatch-release",
		System:  "You coordinate releases.",
		Messages: []sdk.Message{
			sdk.UserMessage("deploy and ask for confirmation"),
			{
				Role: sdk.MessageRoleAssistant,
				Content: []sdk.MessagePart{
					sdk.TextPart{Text: "I will prepare the release."},
					completionCallPart("c1"),
					completionCallPart("c2"),
					completionCallPart("c3"),
				},
			},
			sdk.ToolMessage(sdk.ToolResultPart{
				ToolCallID: "c1",
				ToolName:   "lookup",
				Result:     map[string]any{"version": "v2"},
			}),
		},
		Pending: []sdk.DeferredToolApproval{completionDeferred("c2"), completionDeferred("c3")},
	}
}

func completionResults() []sdk.ToolResultPart {
	return []sdk.ToolResultPart{
		{
			ToolCallID:   "c3",
			Result:       map[string]any{"answer": "yes"},
			IsError:      true,
			CacheControl: &sdk.CacheControl{Type: "ephemeral", TTL: "1h"},
		},
		{
			ToolCallID: "c2",
			ToolName:   "deploy",
			Result:     map[string]any{"deployment": "dep-42"},
		},
	}
}

func toolResultParts(t *testing.T, message sdk.Message) []sdk.ToolResultPart {
	t.Helper()
	if message.Role != sdk.MessageRoleTool {
		t.Fatalf("message role = %q, want tool", message.Role)
	}
	results := make([]sdk.ToolResultPart, len(message.Content))
	for i, part := range message.Content {
		result, ok := part.(sdk.ToolResultPart)
		if !ok {
			t.Fatalf("message.Content[%d] = %T, want sdk.ToolResultPart", i, part)
		}
		results[i] = result
	}
	return results
}

func replaceAssistantCall(pause *sdk.ToolApprovalPause, id string, mutate func(*sdk.ToolCallPart)) {
	for messageIndex := range pause.Messages {
		for partIndex, part := range pause.Messages[messageIndex].Content {
			call, ok := part.(sdk.ToolCallPart)
			if ok && call.ToolCallID == id {
				mutate(&call)
				pause.Messages[messageIndex].Content[partIndex] = call
				return
			}
		}
	}
}

func TestCompleteToolApprovalPause_CompletesSupportedLayouts(t *testing.T) {
	tests := []struct {
		name    string
		prepare func(*sdk.ToolApprovalPause, *[]sdk.ToolResultPart)
		wantIDs []string
		verify  func(*testing.T, []sdk.Message)
	}{
		{
			name:    "resolved prefix and unordered host results",
			wantIDs: []string{"c2", "c3"},
			verify: func(t *testing.T, completed []sdk.Message) {
				existing := toolResultParts(t, completed[len(completed)-2])
				if len(existing) != 1 || existing[0].ToolCallID != "c1" || existing[0].Result.(map[string]any)["version"] != "v2" {
					t.Fatalf("existing sibling result changed: %#v", existing)
				}

				added := toolResultParts(t, completed[len(completed)-1])
				if added[0].ToolName != "deploy" || added[0].Result.(map[string]any)["deployment"] != "dep-42" {
					t.Fatalf("deploy result changed: %#v", added[0])
				}
				if added[1].ToolName != "ask_user" || !added[1].IsError || added[1].Result.(map[string]any)["answer"] != "yes" {
					t.Fatalf("ask_user result changed: %#v", added[1])
				}
				if added[1].CacheControl == nil || added[1].CacheControl.TTL != "1h" {
					t.Fatalf("cache control changed: %#v", added[1].CacheControl)
				}
			},
		},
		{
			name: "no resolved siblings",
			prepare: func(pause *sdk.ToolApprovalPause, results *[]sdk.ToolResultPart) {
				pause.Messages = pause.Messages[:len(pause.Messages)-1]
				pause.Pending = append([]sdk.DeferredToolApproval{completionDeferred("c1")}, pause.Pending...)
				*results = append(*results, sdk.ToolResultPart{ToolCallID: "c1"})
			},
			wantIDs: []string{"c1", "c2", "c3"},
			verify: func(t *testing.T, completed []sdk.Message) {
				result := toolResultParts(t, completed[len(completed)-1])[0]
				if result.ToolName != "lookup" || result.Result != nil {
					t.Fatalf("nil result was not preserved: %#v", result)
				}
			},
		},
		{
			name: "interleaved resolved sibling",
			prepare: func(pause *sdk.ToolApprovalPause, results *[]sdk.ToolResultPart) {
				pause.Messages[len(pause.Messages)-1] = sdk.ToolMessage(sdk.ToolResultPart{
					ToolCallID: "c2", ToolName: "deploy", Result: map[string]any{"deployment": "dep-42"},
				})
				pause.Pending = []sdk.DeferredToolApproval{completionDeferred("c1"), completionDeferred("c3")}
				*results = []sdk.ToolResultPart{
					{ToolCallID: "c3", Result: "yes"},
					{ToolCallID: "c1", Result: map[string]any{"version": "v2"}},
				}
			},
			wantIDs: []string{"c1", "c3"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pause := completionPause()
			results := completionResults()
			if test.prepare != nil {
				test.prepare(&pause, &results)
			}
			completed, err := sdk.CompleteToolApprovalPause(&pause, results)
			if err != nil {
				t.Fatalf("CompleteToolApprovalPause: %v", err)
			}
			if len(completed) != len(pause.Messages)+1 {
				t.Fatalf("completed messages = %d, want %d", len(completed), len(pause.Messages)+1)
			}
			if ids := resultIDs(toolResultParts(t, completed[len(completed)-1])); !reflect.DeepEqual(ids, test.wantIDs) {
				t.Fatalf("completing result IDs = %v, want %v", ids, test.wantIDs)
			}
			if test.verify != nil {
				test.verify(t, completed)
			}
		})
	}
}

func TestCompleteToolApprovalPause_RejectsInvalidState(t *testing.T) {
	t.Run("nil pause", func(t *testing.T) {
		completed, err := sdk.CompleteToolApprovalPause(nil, completionResults())
		if err == nil || !strings.Contains(err.Error(), "pause is nil") {
			t.Fatalf("error = %v, want nil pause error", err)
		}
		if completed != nil {
			t.Fatalf("completed = %#v on validation failure, want nil", completed)
		}
	})

	tests := []struct {
		name    string
		mutate  func(*sdk.ToolApprovalPause, *[]sdk.ToolResultPart)
		wantErr string
	}{
		// Host result validation.
		{
			name: "missing",
			mutate: func(_ *sdk.ToolApprovalPause, results *[]sdk.ToolResultPart) {
				*results = (*results)[:1]
			},
			wantErr: `missing result for pending tool call "c2"`,
		},
		{
			name: "unknown",
			mutate: func(_ *sdk.ToolApprovalPause, results *[]sdk.ToolResultPart) {
				*results = append(*results, sdk.ToolResultPart{ToolCallID: "ghost", ToolName: "deploy"})
			},
			wantErr: `unknown pending tool call "ghost"`,
		},
		{
			name: "duplicate",
			mutate: func(_ *sdk.ToolApprovalPause, results *[]sdk.ToolResultPart) {
				*results = append(*results, (*results)[0])
			},
			wantErr: `duplicate tool call ID "c3"`,
		},
		{
			name: "empty ID",
			mutate: func(_ *sdk.ToolApprovalPause, results *[]sdk.ToolResultPart) {
				(*results)[0].ToolCallID = ""
			},
			wantErr: "empty tool call ID",
		},
		{
			name: "already resolved sibling",
			mutate: func(_ *sdk.ToolApprovalPause, results *[]sdk.ToolResultPart) {
				*results = append(*results, sdk.ToolResultPart{ToolCallID: "c1", ToolName: "lookup"})
			},
			wantErr: `tool call "c1" already has a result`,
		},
		{
			name: "wrong tool name",
			mutate: func(_ *sdk.ToolApprovalPause, results *[]sdk.ToolResultPart) {
				(*results)[0].ToolName = "deploy"
			},
			wantErr: `names tool "deploy"; want "ask_user"`,
		},

		// Conversation shape validation.
		{
			name: "no pending calls",
			mutate: func(pause *sdk.ToolApprovalPause, _ *[]sdk.ToolResultPart) {
				pause.Pending = nil
			},
			wantErr: "pause has no pending tool calls",
		},
		{
			name: "no messages",
			mutate: func(pause *sdk.ToolApprovalPause, _ *[]sdk.ToolResultPart) {
				pause.Messages = nil
			},
			wantErr: "pause has no messages",
		},
		{
			name: "non-tool tail",
			mutate: func(pause *sdk.ToolApprovalPause, _ *[]sdk.ToolResultPart) {
				pause.Messages = append(pause.Messages, sdk.UserMessage("late input"))
			},
			wantErr: "must end with an assistant tool-call message followed only by tool messages",
		},
		{
			name: "assistant has no calls",
			mutate: func(pause *sdk.ToolApprovalPause, _ *[]sdk.ToolResultPart) {
				pause.Messages = []sdk.Message{sdk.UserMessage("hello"), sdk.AssistantMessage("done")}
			},
			wantErr: "final assistant message contains no tool calls",
		},
		{
			name: "assistant call empty ID",
			mutate: func(pause *sdk.ToolApprovalPause, _ *[]sdk.ToolResultPart) {
				replaceAssistantCall(pause, "c2", func(call *sdk.ToolCallPart) { call.ToolCallID = "" })
			},
			wantErr: "has an empty ID",
		},
		{
			name: "assistant call empty name",
			mutate: func(pause *sdk.ToolApprovalPause, _ *[]sdk.ToolResultPart) {
				replaceAssistantCall(pause, "c2", func(call *sdk.ToolCallPart) { call.ToolName = "" })
			},
			wantErr: `assistant tool call "c2" has an empty tool name`,
		},
		{
			name: "assistant duplicate call ID",
			mutate: func(pause *sdk.ToolApprovalPause, _ *[]sdk.ToolResultPart) {
				replaceAssistantCall(pause, "c3", func(call *sdk.ToolCallPart) { call.ToolCallID = "c2" })
			},
			wantErr: `duplicate tool call ID "c2"`,
		},
		{
			name: "empty tool message",
			mutate: func(pause *sdk.ToolApprovalPause, _ *[]sdk.ToolResultPart) {
				pause.Messages[len(pause.Messages)-1].Content = nil
			},
			wantErr: "tool message at index 2 is empty",
		},
		{
			name: "non-result in tool message",
			mutate: func(pause *sdk.ToolApprovalPause, _ *[]sdk.ToolResultPart) {
				pause.Messages[len(pause.Messages)-1].Content = append(pause.Messages[len(pause.Messages)-1].Content, sdk.TextPart{Text: "bad"})
			},
			wantErr: "contains a non-tool-result part",
		},
		{
			name: "existing result empty ID",
			mutate: func(pause *sdk.ToolApprovalPause, _ *[]sdk.ToolResultPart) {
				pause.Messages[len(pause.Messages)-1] = sdk.ToolMessage(sdk.ToolResultPart{ToolName: "lookup"})
			},
			wantErr: "has an empty tool call ID",
		},
		{
			name: "existing result unknown ID",
			mutate: func(pause *sdk.ToolApprovalPause, _ *[]sdk.ToolResultPart) {
				pause.Messages[len(pause.Messages)-1] = sdk.ToolMessage(sdk.ToolResultPart{ToolCallID: "ghost", ToolName: "lookup"})
			},
			wantErr: `unknown tool call "ghost"`,
		},
		{
			name: "existing result duplicate ID",
			mutate: func(pause *sdk.ToolApprovalPause, _ *[]sdk.ToolResultPart) {
				result := sdk.ToolResultPart{ToolCallID: "c1", ToolName: "lookup"}
				pause.Messages[len(pause.Messages)-1] = sdk.ToolMessage(result, result)
			},
			wantErr: `duplicate result for tool call "c1"`,
		},
		{
			name: "existing result wrong name",
			mutate: func(pause *sdk.ToolApprovalPause, _ *[]sdk.ToolResultPart) {
				pause.Messages[len(pause.Messages)-1] = sdk.ToolMessage(sdk.ToolResultPart{ToolCallID: "c1", ToolName: "deploy"})
			},
			wantErr: `names tool "deploy"; want "lookup"`,
		},
		{
			name: "existing results out of call order",
			mutate: func(pause *sdk.ToolApprovalPause, _ *[]sdk.ToolResultPart) {
				pause.Messages[len(pause.Messages)-1] = sdk.ToolMessage(
					sdk.ToolResultPart{ToolCallID: "c2", ToolName: "deploy"},
					sdk.ToolResultPart{ToolCallID: "c1", ToolName: "lookup"},
				)
				pause.Pending = pause.Pending[1:]
			},
			wantErr: `do not follow the assistant tool-call order at "c1"`,
		},

		// Pending snapshot validation.
		{
			name: "missing pending entry",
			mutate: func(pause *sdk.ToolApprovalPause, _ *[]sdk.ToolResultPart) {
				pause.Pending = pause.Pending[:1]
			},
			wantErr: "lists 1 calls but pause.Messages leaves 2 unresolved",
		},
		{
			name: "pending order",
			mutate: func(pause *sdk.ToolApprovalPause, _ *[]sdk.ToolResultPart) {
				pause.Pending[0], pause.Pending[1] = pause.Pending[1], pause.Pending[0]
			},
			wantErr: `pause.Pending[0] identifies tool call "c3"; want "c2"`,
		},
		{
			name: "pending tool name",
			mutate: func(pause *sdk.ToolApprovalPause, _ *[]sdk.ToolResultPart) {
				pause.Pending[0].ToolCall.ToolName = "ask_user"
			},
			wantErr: `names tool "ask_user"; want "deploy"`,
		},
		{
			name: "pending input",
			mutate: func(pause *sdk.ToolApprovalPause, _ *[]sdk.ToolResultPart) {
				pause.Pending[0].ToolCall.Input.(map[string]any)["env"] = "staging"
			},
			wantErr: "input for tool call \"c2\" does not match",
		},
		{
			name: "pending provider metadata",
			mutate: func(pause *sdk.ToolApprovalPause, _ *[]sdk.ToolResultPart) {
				pause.Pending[0].ToolCall.ProviderMetadata["trace"].(map[string]any)["position"] = float64(99)
			},
			wantErr: "provider metadata for tool call \"c2\" does not match",
		},
		{
			name: "zero decision is not pending",
			mutate: func(pause *sdk.ToolApprovalPause, _ *[]sdk.ToolResultPart) {
				pause.Pending[0].Approval.Decision = ""
			},
			wantErr: `has decision ""; want "deferred"`,
		},
		{
			name: "non JSON-compatible pending input",
			mutate: func(pause *sdk.ToolApprovalPause, _ *[]sdk.ToolResultPart) {
				ch := make(chan int)
				pause.Pending[0].ToolCall.Input = ch
				replaceAssistantCall(pause, "c2", func(call *sdk.ToolCallPart) { call.Input = ch })
			},
			wantErr: "recorded value is not JSON-compatible",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pause := completionPause()
			results := completionResults()
			test.mutate(&pause, &results)
			completed, err := sdk.CompleteToolApprovalPause(&pause, results)
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("error = %v, want substring %q", err, test.wantErr)
			}
			if completed != nil {
				t.Fatalf("completed = %#v on validation failure, want nil", completed)
			}
		})
	}
}

func TestCompleteToolApprovalPause_AcceptsJSONEquivalentSnapshot(t *testing.T) {
	type deploymentInput struct {
		Env     string   `json:"env"`
		Regions []string `json:"regions"`
	}
	type trace struct {
		Position int `json:"position"`
	}

	pause := completionPause()
	pause.Pending[0].ToolCall.Input = deploymentInput{Env: "prod", Regions: []string{"nrt", "iad"}}
	pause.Pending[0].ToolCall.ProviderMetadata = nil
	replaceAssistantCall(&pause, "c2", func(call *sdk.ToolCallPart) {
		call.ProviderMetadata = nil
	})
	pause.Pending[1].ToolCall.ProviderMetadata = map[string]any{"trace": trace{Position: 3}}

	completed, err := sdk.CompleteToolApprovalPause(&pause, completionResults())
	if err != nil {
		t.Fatalf("JSON-equivalent struct/map snapshot was rejected: %v", err)
	}
	if len(completed) != len(pause.Messages)+1 {
		t.Fatalf("completed messages = %d, want %d", len(completed), len(pause.Messages)+1)
	}
}

func TestCompleteToolApprovalPause_JSONRoundTrip(t *testing.T) {
	pause := roundTripJSON(t, completionPause())
	results := roundTripJSON(t, completionResults())

	completed, err := sdk.CompleteToolApprovalPause(&pause, results)
	if err != nil {
		t.Fatalf("complete JSON round-trip pause: %v", err)
	}
	if ids := resultIDs(toolResultParts(t, completed[len(completed)-1])); !reflect.DeepEqual(ids, []string{"c2", "c3"}) {
		t.Fatalf("completed result IDs = %v, want [c2 c3]", ids)
	}
}

func TestCompleteToolApprovalPause_ReturnsDeepSnapshot(t *testing.T) {
	pause := completionPause()
	results := completionResults()
	completed, err := sdk.CompleteToolApprovalPause(&pause, results)
	if err != nil {
		t.Fatalf("CompleteToolApprovalPause: %v", err)
	}
	completedSnapshot := snapshotJSON(t, completed)

	// Mutating either input after completion must not rewrite the returned
	// conversation, including values nested behind interfaces and pointers.
	replaceAssistantCall(&pause, "c2", func(call *sdk.ToolCallPart) {
		call.Input.(map[string]any)["regions"].([]any)[0] = "mutated"
		call.ProviderMetadata["trace"].(map[string]any)["position"] = float64(99)
	})
	existing := pause.Messages[len(pause.Messages)-1].Content[0].(sdk.ToolResultPart)
	existing.Result.(map[string]any)["version"] = "mutated"
	results[0].Result.(map[string]any)["answer"] = "mutated"
	results[0].CacheControl.TTL = "mutated"
	if got := snapshotJSON(t, completed); got != completedSnapshot {
		t.Fatalf("completed conversation changed through its inputs\ngot:  %s\nwant: %s", got, completedSnapshot)
	}

	// Mutating the returned snapshot must likewise leave both inputs intact.
	pauseSnapshot := snapshotJSON(t, pause)
	resultsSnapshot := snapshotJSON(t, results)
	added := toolResultParts(t, completed[len(completed)-1])
	added[0].Result.(map[string]any)["deployment"] = "changed-in-output"
	if got := snapshotJSON(t, pause); got != pauseSnapshot {
		t.Fatalf("pause changed through completed conversation\ngot:  %s\nwant: %s", got, pauseSnapshot)
	}
	if got := snapshotJSON(t, results); got != resultsSnapshot {
		t.Fatalf("host results changed through completed conversation\ngot:  %s\nwant: %s", got, resultsSnapshot)
	}
}

func resultIDs(results []sdk.ToolResultPart) []string {
	ids := make([]string, len(results))
	for i, result := range results {
		ids[i] = result.ToolCallID
	}
	return ids
}

func roundTripJSON[T any](t *testing.T, value T) T {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal JSON: %v", err)
	}
	var result T
	if err := json.Unmarshal(data, &result); err != nil {
		t.Fatalf("unmarshal JSON: %v", err)
	}
	return result
}

func snapshotJSON(t *testing.T, value any) string {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal JSON snapshot: %v", err)
	}
	return string(data)
}
