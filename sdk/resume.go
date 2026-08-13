package sdk

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
)

// CompleteToolApprovalPause validates final results supplied by the host for
// every pending call and returns a protocol-complete conversation.
//
// This function is deliberately limited to conversation assembly. It does not
// decide approvals, execute tools, call a model, or persist anything. The host
// remains responsible for those operations and may therefore apply its own
// transactional, fencing, and retry guarantees before completing the pause.
// Results may be provided in any order; the completing tool message follows
// the original assistant tool-call order.
func CompleteToolApprovalPause(pause *ToolApprovalPause, results []ToolResultPart) ([]Message, error) {
	if pause == nil {
		return nil, completePauseErrorf("pause is nil")
	}
	parsed, err := parseToolApprovalPause(pause)
	if err != nil {
		return nil, err
	}

	resultsByID := make(map[string]ToolResultPart, len(results))
	for i, result := range results {
		if result.ToolCallID == "" {
			return nil, completePauseErrorf("results[%d] has an empty tool call ID", i)
		}
		if _, duplicate := resultsByID[result.ToolCallID]; duplicate {
			return nil, completePauseErrorf("results contain duplicate tool call ID %q", result.ToolCallID)
		}
		call, pending := parsed.pendingByID[result.ToolCallID]
		if !pending {
			if _, resolved := parsed.existingResultIDs[result.ToolCallID]; resolved {
				return nil, completePauseErrorf("tool call %q already has a result in pause.Messages", result.ToolCallID)
			}
			return nil, completePauseErrorf("result refers to unknown pending tool call %q", result.ToolCallID)
		}
		if result.ToolName != "" && result.ToolName != call.ToolName {
			return nil, completePauseErrorf("result for tool call %q names tool %q; want %q", result.ToolCallID, result.ToolName, call.ToolName)
		}
		resultsByID[result.ToolCallID] = result
	}

	ordered := make([]ToolResultPart, len(parsed.pending))
	for i, call := range parsed.pending {
		result, ok := resultsByID[call.ToolCallID]
		if !ok {
			return nil, completePauseErrorf("missing result for pending tool call %q", call.ToolCallID)
		}
		result = cloneSDKValue(result)
		result.ToolCallID = call.ToolCallID
		result.ToolName = call.ToolName
		ordered[i] = result
	}

	completed := make([]Message, 0, len(pause.Messages)+1)
	completed = append(completed, clonePauseMessages(pause.Messages)...)
	completed = append(completed, ToolMessage(ordered...))
	return completed, nil
}

type parsedToolApprovalPause struct {
	pending           []ToolCall
	pendingByID       map[string]ToolCall
	existingResultIDs map[string]struct{}
}

func parseToolApprovalPause(pause *ToolApprovalPause) (*parsedToolApprovalPause, error) {
	if len(pause.Pending) == 0 {
		return nil, completePauseErrorf("pause has no pending tool calls")
	}
	if len(pause.Messages) == 0 {
		return nil, completePauseErrorf("pause has no messages")
	}

	assistantIndex := len(pause.Messages) - 1
	for assistantIndex >= 0 && pause.Messages[assistantIndex].Role == MessageRoleTool {
		assistantIndex--
	}
	if assistantIndex < 0 || pause.Messages[assistantIndex].Role != MessageRoleAssistant {
		return nil, completePauseErrorf("pause.Messages must end with an assistant tool-call message followed only by tool messages")
	}

	calls := make([]ToolCall, 0)
	callIndexByID := make(map[string]int)
	for partIndex, part := range pause.Messages[assistantIndex].Content {
		callPart, ok := part.(ToolCallPart)
		if !ok {
			if _, isResult := part.(ToolResultPart); isResult {
				return nil, completePauseErrorf("assistant message contains a tool result at content index %d", partIndex)
			}
			continue
		}
		if callPart.ToolCallID == "" {
			return nil, completePauseErrorf("assistant tool call at content index %d has an empty ID", partIndex)
		}
		if callPart.ToolName == "" {
			return nil, completePauseErrorf("assistant tool call %q has an empty tool name", callPart.ToolCallID)
		}
		if _, duplicate := callIndexByID[callPart.ToolCallID]; duplicate {
			return nil, completePauseErrorf("assistant message contains duplicate tool call ID %q", callPart.ToolCallID)
		}
		callIndexByID[callPart.ToolCallID] = len(calls)
		calls = append(calls, ToolCall{
			ToolCallID:       callPart.ToolCallID,
			ToolName:         callPart.ToolName,
			Input:            callPart.Input,
			ProviderMetadata: callPart.ProviderMetadata,
		})
	}
	if len(calls) == 0 {
		return nil, completePauseErrorf("final assistant message contains no tool calls")
	}

	existingResultIDs := make(map[string]struct{})
	lastCallIndex := -1
	for messageIndex := assistantIndex + 1; messageIndex < len(pause.Messages); messageIndex++ {
		message := pause.Messages[messageIndex]
		if len(message.Content) == 0 {
			return nil, completePauseErrorf("tool message at index %d is empty", messageIndex)
		}
		for partIndex, part := range message.Content {
			result, ok := part.(ToolResultPart)
			if !ok {
				return nil, completePauseErrorf("tool message at index %d contains a non-tool-result part at content index %d", messageIndex, partIndex)
			}
			if result.ToolCallID == "" {
				return nil, completePauseErrorf("existing tool result at message %d content %d has an empty tool call ID", messageIndex, partIndex)
			}
			if _, duplicate := existingResultIDs[result.ToolCallID]; duplicate {
				return nil, completePauseErrorf("pause.Messages contains duplicate result for tool call %q", result.ToolCallID)
			}
			callIndex, known := callIndexByID[result.ToolCallID]
			if !known {
				return nil, completePauseErrorf("existing result refers to unknown tool call %q", result.ToolCallID)
			}
			call := calls[callIndex]
			if result.ToolName == "" || result.ToolName != call.ToolName {
				return nil, completePauseErrorf("existing result for tool call %q names tool %q; want %q", result.ToolCallID, result.ToolName, call.ToolName)
			}
			if callIndex <= lastCallIndex {
				return nil, completePauseErrorf("existing tool results do not follow the assistant tool-call order at %q", result.ToolCallID)
			}
			lastCallIndex = callIndex
			existingResultIDs[result.ToolCallID] = struct{}{}
		}
	}

	pending := make([]ToolCall, 0, len(calls)-len(existingResultIDs))
	for _, call := range calls {
		if _, resolved := existingResultIDs[call.ToolCallID]; !resolved {
			pending = append(pending, call)
		}
	}
	if len(pending) != len(pause.Pending) {
		return nil, completePauseErrorf("pause.Pending lists %d calls but pause.Messages leaves %d unresolved", len(pause.Pending), len(pending))
	}

	pendingByID := make(map[string]ToolCall, len(pending))
	for i, call := range pending {
		recorded := pause.Pending[i]
		if recorded.Approval.Decision != ToolApprovalDecisionDeferred {
			return nil, completePauseErrorf("pause.Pending[%d] for tool call %q has decision %q; want %q", i, recorded.ToolCall.ToolCallID, recorded.Approval.Decision, ToolApprovalDecisionDeferred)
		}
		if recorded.ToolCall.ToolCallID != call.ToolCallID {
			return nil, completePauseErrorf("pause.Pending[%d] identifies tool call %q; want %q", i, recorded.ToolCall.ToolCallID, call.ToolCallID)
		}
		if recorded.ToolCall.ToolName != call.ToolName {
			return nil, completePauseErrorf("pause.Pending[%d] for tool call %q names tool %q; want %q", i, call.ToolCallID, recorded.ToolCall.ToolName, call.ToolName)
		}
		if err := requireEquivalentJSON(recorded.ToolCall.Input, call.Input); err != nil {
			return nil, completePauseErrorf("pause.Pending[%d] input for tool call %q does not match pause.Messages: %v", i, call.ToolCallID, err)
		}
		if err := requireEquivalentJSON(recorded.ToolCall.ProviderMetadata, call.ProviderMetadata); err != nil {
			return nil, completePauseErrorf("pause.Pending[%d] provider metadata for tool call %q does not match pause.Messages: %v", i, call.ToolCallID, err)
		}
		pendingByID[call.ToolCallID] = call
	}

	return &parsedToolApprovalPause{
		pending:           pending,
		pendingByID:       pendingByID,
		existingResultIDs: existingResultIDs,
	}, nil
}

func requireEquivalentJSON(got, want any) error {
	gotJSON, err := normalizeJSONValue(got)
	if err != nil {
		return fmt.Errorf("recorded value is not JSON-compatible: %w", err)
	}
	wantJSON, err := normalizeJSONValue(want)
	if err != nil {
		return fmt.Errorf("conversation value is not JSON-compatible: %w", err)
	}
	if !reflect.DeepEqual(gotJSON, wantJSON) {
		return fmt.Errorf("values differ")
	}
	return nil
}

func normalizeJSONValue(value any) (any, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var normalized any
	if err := decoder.Decode(&normalized); err != nil {
		return nil, err
	}
	return normalized, nil
}

func completePauseErrorf(format string, args ...any) error {
	return fmt.Errorf("twilightai: complete tool approval pause: "+format, args...)
}
