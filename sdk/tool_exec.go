package sdk

import (
	"context"
	"fmt"
	"sync"
)

func buildToolMap(tools []Tool) map[string]*Tool {
	m := make(map[string]*Tool, len(tools))
	for i := range tools {
		m[tools[i].Name] = &tools[i]
	}
	return m
}

func hasExecutableTools(toolCalls []ToolCall, toolMap map[string]*Tool) bool {
	for _, tc := range toolCalls {
		if t, ok := toolMap[tc.ToolName]; ok && t.Execute != nil {
			return true
		}
	}
	return false
}

type pendingToolExec struct {
	idx  int
	tc   ToolCall
	tool *Tool
}

func executeTools(
	ctx context.Context,
	toolCalls []ToolCall,
	toolMap map[string]*Tool,
	approvalHandler func(context.Context, ToolCall) (bool, error),
	sendProgress func(StreamPart),
) ([]ToolResultPart, error) {
	results := make([]ToolResultPart, len(toolCalls))
	pending := make([]pendingToolExec, 0, len(toolCalls))

	// Phase 1: resolve tools and check approvals (sequential, user-facing).
	for i, tc := range toolCalls {
		tool, ok := toolMap[tc.ToolName]
		if !ok || tool.Execute == nil {
			results[i] = ToolResultPart{
				ToolCallID: tc.ToolCallID,
				ToolName:   tc.ToolName,
				Result:     fmt.Sprintf("tool %q not found or has no execute handler", tc.ToolName),
				IsError:    true,
			}
			continue
		}

		if tool.RequireApproval {
			if sendProgress != nil {
				sendProgress(&ToolApprovalRequestPart{
					ToolCallID: tc.ToolCallID,
					ToolName:   tc.ToolName,
					Input:      tc.Input,
				})
			}

			if approvalHandler == nil {
				if sendProgress != nil {
					sendProgress(&ToolOutputDeniedPart{
						ToolCallID: tc.ToolCallID,
						ToolName:   tc.ToolName,
					})
				}
				results[i] = ToolResultPart{
					ToolCallID: tc.ToolCallID,
					ToolName:   tc.ToolName,
					Result:     "tool execution denied: no approval handler",
					IsError:    true,
				}
				continue
			}

			approved, err := approvalHandler(ctx, tc)
			if err != nil {
				return nil, fmt.Errorf("twilightai: approval handler for %q: %w", tc.ToolName, err)
			}
			if !approved {
				if sendProgress != nil {
					sendProgress(&ToolOutputDeniedPart{
						ToolCallID: tc.ToolCallID,
						ToolName:   tc.ToolName,
					})
				}
				results[i] = ToolResultPart{
					ToolCallID: tc.ToolCallID,
					ToolName:   tc.ToolName,
					Result:     "tool execution denied by user",
					IsError:    true,
				}
				continue
			}
		}

		pending = append(pending, pendingToolExec{idx: i, tc: tc, tool: tool})
	}

	// Phase 2: execute approved tools in parallel.
	if len(pending) == 1 {
		results[pending[0].idx] = runTool(ctx, pending[0].tc, pending[0].tool, sendProgress)
	} else if len(pending) > 1 {
		var wg sync.WaitGroup
		wg.Add(len(pending))
		for _, p := range pending {
			go func(p pendingToolExec) {
				defer wg.Done()
				results[p.idx] = runTool(ctx, p.tc, p.tool, sendProgress)
			}(p)
		}
		wg.Wait()
	}

	return results, nil
}

func runTool(ctx context.Context, tc ToolCall, tool *Tool, sendProgress func(StreamPart)) ToolResultPart {
	var progressFn func(content any)
	if sendProgress != nil {
		progressFn = func(content any) {
			sendProgress(&ToolProgressPart{
				ToolCallID: tc.ToolCallID,
				ToolName:   tc.ToolName,
				Content:    content,
			})
		}
	}

	execCtx := &ToolExecContext{
		Context:      ctx,
		ToolCallID:   tc.ToolCallID,
		ToolName:     tc.ToolName,
		SendProgress: progressFn,
	}

	output, err := tool.Execute(execCtx, tc.Input)
	if err != nil {
		if sendProgress != nil {
			sendProgress(&StreamToolErrorPart{
				ToolCallID: tc.ToolCallID,
				ToolName:   tc.ToolName,
				Error:      err,
			})
		}
		return ToolResultPart{
			ToolCallID: tc.ToolCallID,
			ToolName:   tc.ToolName,
			Result:     err.Error(),
			IsError:    true,
		}
	}

	if sendProgress != nil {
		sendProgress(&StreamToolResultPart{
			ToolCallID: tc.ToolCallID,
			ToolName:   tc.ToolName,
			Input:      tc.Input,
			Output:     output,
		})
	}
	return ToolResultPart{
		ToolCallID: tc.ToolCallID,
		ToolName:   tc.ToolName,
		Result:     output,
	}
}

func toolCallResultsFromParts(parts []ToolResultPart) []ToolResult {
	out := make([]ToolResult, len(parts))
	for i, p := range parts {
		out[i] = ToolResult{
			ToolCallID: p.ToolCallID,
			ToolName:   p.ToolName,
			Output:     p.Result,
		}
	}
	return out
}
