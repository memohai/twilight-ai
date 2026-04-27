package sdk

import (
	"context"
	"errors"
	"fmt"
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

var ErrToolApprovalDeferred = errors.New("tool approval deferred")

type ToolApprovalDeferredError struct {
	Approval ToolApprovalResult
}

func (e *ToolApprovalDeferredError) Error() string {
	if e == nil {
		return ErrToolApprovalDeferred.Error()
	}
	if e.Approval.ApprovalID == "" {
		return ErrToolApprovalDeferred.Error()
	}
	return fmt.Sprintf("%s: %s", ErrToolApprovalDeferred, e.Approval.ApprovalID)
}

func (e *ToolApprovalDeferredError) Is(target error) bool {
	return target == ErrToolApprovalDeferred
}

type Tool struct {
	Name            string          `json:"name"`
	Description     string          `json:"description,omitempty"`
	Parameters      any             `json:"parameters"` // *jsonschema.Schema, or a Go struct for automatic inference
	Execute         ToolExecuteFunc `json:"-"`
	RequireApproval bool            `json:"-"`
}
