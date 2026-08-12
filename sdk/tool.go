package sdk

import (
	"context"
)

// ToolExecuteFunc is the signature for a tool's execution handler.
// input is the parsed arguments from the LLM. The return value becomes the
// tool result output sent back to the model.
type ToolExecuteFunc func(ctx *ToolExecContext, input any) (any, error)

// ToolExecContext is passed to ToolExecuteFunc and carries the parent context,
// call metadata, and a mechanism for streaming progress updates.
type ToolExecContext struct {
	context.Context
	ToolCallID   string
	ToolName     string
	SendProgress func(content any) // nil when not in streaming mode
}

type ToolApprovalDecision string

const (
	ToolApprovalDecisionApproved ToolApprovalDecision = "approved"
	ToolApprovalDecisionRejected ToolApprovalDecision = "rejected"
	ToolApprovalDecisionDeferred ToolApprovalDecision = "deferred"
)

type ToolApprovalResult struct {
	Decision   ToolApprovalDecision `json:"decision"`
	ApprovalID string               `json:"approvalId,omitempty"`
	Reason     string               `json:"reason,omitempty"`
	Metadata   map[string]any       `json:"metadata,omitempty"`
}

// DeferredToolApproval pairs a tool call awaiting a user decision with the
// approval state returned by the ApprovalHandler for that call.
type DeferredToolApproval struct {
	ToolCall ToolCall           `json:"toolCall"`
	Approval ToolApprovalResult `json:"approval"`
}

// ToolApprovalBatchResult is one call's answer in a batch approval response,
// keyed by ToolCallID so the SDK can verify the batch is complete and
// correctly addressed regardless of the order the host assembled it in.
type ToolApprovalBatchResult struct {
	ToolCallID string             `json:"toolCallId"`
	Result     ToolApprovalResult `json:"result"`
}

// ToolApprovalPause is the portable state of a run that stopped on deferred
// tool approvals (FinishReason == FinishReasonPaused). It contains the exact
// system prompt and conversation used for the paused model step, followed by
// that step's assistant message and any already-resolved tool results, plus
// the calls awaiting a decision. Hosts may persist or transfer this plain data
// while collecting decisions. JSON persistence supports the SDK's built-in
// MessagePart types and requires tool inputs, tool results, and metadata to be
// JSON-compatible.
type ToolApprovalPause struct {
	// BatchID identifies the approval batch that produced this pause. It is
	// set only when an ApprovalBatchHandler was consulted — it is the same ID
	// that invocation received, giving hosts a stable correlation key. It is a
	// correlation ID, not an idempotency key — retrying a step mints a new one;
	// hosts needing idempotency must key on their own run identity plus the
	// tool-call IDs. The absence of a pause alone does not establish that host
	// records are orphaned because a fully approved or rejected batch also
	// completes without one.
	// Empty when the per-call ApprovalHandler was used (that handler never
	// sees a batch ID, so records cannot be correlated by it).
	BatchID string `json:"batchId,omitempty"`
	// System is the root instruction used for the paused model step, including
	// any override applied by OnStep or PrepareStep.
	System string `json:"system,omitempty"`
	// Messages is the effective input of the paused model step followed by that
	// step's assistant message and any already-resolved tool results.
	Messages []Message `json:"messages"`
	// Pending lists the tool calls awaiting a user decision, in the paused
	// assistant message's order.
	Pending []DeferredToolApproval `json:"pending"`
}

type Tool struct {
	Name            string          `json:"name"`
	Description     string          `json:"description,omitempty"`
	Parameters      any             `json:"parameters"` // *jsonschema.Schema, or a Go struct for automatic inference
	Execute         ToolExecuteFunc `json:"-"`
	RequireApproval bool            `json:"-"`
	// CacheControl enables prompt caching for this tool's definition.
	// Only supported by Anthropic; other providers ignore this field.
	CacheControl *CacheControl `json:"-"`
}
