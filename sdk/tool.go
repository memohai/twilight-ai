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

// ToolApprovalPause is the portable state of a run that stopped on deferred
// tool approvals (FinishReason == FinishReasonPaused). It carries everything
// needed to resume: the run's system prompt, the full conversation (input
// plus every message the run produced, ending in the paused step's assistant
// message and the tool message covering the already-resolved calls) and the
// calls awaiting a decision. The value is plain data — persist it, ship it
// across processes, and hand it back to ResumeText / ApplyToolDecisions once
// decisions arrive.
type ToolApprovalPause struct {
	// System is the root instruction the paused run was started with. Resume
	// re-applies it so the model keeps its original instructions; a resume
	// that also passes WithSystem is rejected as conflicting.
	System string `json:"system,omitempty"`
	// Messages is the full conversation at the pause point.
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
