package sdk

import (
	"context"
	"fmt"
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

func executeTools(
	ctx context.Context,
	toolCalls []ToolCall,
	toolMap map[string]*Tool,
	approvalHandler func(context.Context, ToolCall) (bool, error),
	sendProgress func(StreamPart),
) ([]ToolResultPart, error) {
	results := make([]ToolResultPart, 0, len(toolCalls))

	for _, tc := range toolCalls {
		tool, ok := toolMap[tc.ToolName]
		if !ok || tool.Execute == nil {
			results = append(results, ToolResultPart{
				ToolCallID: tc.ToolCallID,
				ToolName:   tc.ToolName,
				Result:     fmt.Sprintf("tool %q not found or has no execute handler", tc.ToolName),
				IsError:    true,
			})
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
				results = append(results, ToolResultPart{
					ToolCallID: tc.ToolCallID,
					ToolName:   tc.ToolName,
					Result:     "tool execution denied: no approval handler",
					IsError:    true,
				})
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
				results = append(results, ToolResultPart{
					ToolCallID: tc.ToolCallID,
					ToolName:   tc.ToolName,
					Result:     "tool execution denied by user",
					IsError:    true,
				})
				continue
			}
		}

		var progressFn func(content any)
		if sendProgress != nil {
			callID, toolName := tc.ToolCallID, tc.ToolName
			progressFn = func(content any) {
				sendProgress(&ToolProgressPart{
					ToolCallID: callID,
					ToolName:   toolName,
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
			results = append(results, ToolResultPart{
				ToolCallID: tc.ToolCallID,
				ToolName:   tc.ToolName,
				Result:     err.Error(),
				IsError:    true,
			})
			continue
		}

		if sendProgress != nil {
			sendProgress(&StreamToolResultPart{
				ToolCallID: tc.ToolCallID,
				ToolName:   tc.ToolName,
				Input:      tc.Input,
				Output:     output,
			})
		}
		results = append(results, ToolResultPart{
			ToolCallID: tc.ToolCallID,
			ToolName:   tc.ToolName,
			Result:     output,
		})
	}

	return results, nil
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
