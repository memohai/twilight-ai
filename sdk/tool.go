package sdk

import "context"

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

type Tool struct {
	Name            string          `json:"name"`
	Description     string          `json:"description,omitempty"`
	Parameters      any             `json:"parameters"` // JSON Schema object
	Execute         ToolExecuteFunc `json:"-"`
	RequireApproval bool            `json:"-"`
}
