package sdk

import (
	"context"
	"crypto/rand"
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

type toolDecisionKind int

const (
	toolDecisionNotFound toolDecisionKind = iota
	toolDecisionNoHandler
	toolDecisionApproved
	toolDecisionRejected
	toolDecisionDeferred
)

// toolDecision is the resolved outcome of the approval phase for one call.
type toolDecision struct {
	tc       ToolCall
	tool     *Tool
	kind     toolDecisionKind
	approval ToolApprovalResult
}

// approvalConfig carries the host's approval callbacks into executeTools.
// At most one of the two fields is set (enforced by buildConfig).
type approvalConfig struct {
	// handler answers one call at a time.
	handler func(context.Context, ToolCall) (ToolApprovalResult, error)
	// batchHandler answers all approval-gated calls of a step in one
	// invocation, receiving the SDK-generated batch ID that will reappear as
	// ToolApprovalPause.BatchID if any call defers.
	batchHandler func(ctx context.Context, batchID string, calls []ToolCall) ([]ToolApprovalBatchResult, error)
}

func (a approvalConfig) configured() bool {
	return a.handler != nil || a.batchHandler != nil
}

// newApprovalBatchID mints the identifier that links a batch-handler
// invocation to the pause it may produce.
func newApprovalBatchID() string {
	b := make([]byte, 12)
	if _, err := rand.Read(b); err != nil {
		panic("twilightai: approval batch ID entropy failure: " + err.Error())
	}
	return fmt.Sprintf("apbatch_%x", b)
}

// executeTools resolves each tool call, runs approval checks, and executes the
// approved calls in parallel.
//
// Results are returned in tool-call order. When one or more calls are deferred
// (awaiting a user decision), deferred lists them in tool-call order,
// batchID identifies the approval batch (it reappears as
// ToolApprovalPause.BatchID), and results holds only the calls already
// resolved in the same batch (executed, rejected, or failed); callers
// resuming the run must combine the two to form a protocol-complete tool
// message. A non-nil error means the batch failed outright (approval handler
// error, unknown decision) — never a deferral.
//
// The approval phase is free of side effects: every handler is consulted
// before anything is emitted or executed, so a handler error cannot orphan
// approval requests that were already announced to the stream, and no tool
// runs in a batch that is about to fail. With a batch handler this extends
// to the host: a transactional handler that rolled back leaves no trace on
// either side.
func executeTools(
	ctx context.Context,
	toolCalls []ToolCall,
	toolMap map[string]*Tool,
	approvals approvalConfig,
	sendProgress func(StreamPart),
) (results []ToolResultPart, deferred []DeferredToolApproval, batchID string, err error) {
	// Phase 1: resolve tools and collect every approval decision. No
	// emissions, no executions.
	decisions := make([]toolDecision, len(toolCalls))
	seenIDs := make(map[string]bool, len(toolCalls))
	duplicateIDs := false
	var gated []int // indices of approval-gated calls, in tool-call order
	for i, tc := range toolCalls {
		if tc.ToolCallID != "" {
			if seenIDs[tc.ToolCallID] {
				duplicateIDs = true
			}
			seenIDs[tc.ToolCallID] = true
		}
		d := toolDecision{tc: tc}
		tool, ok := toolMap[tc.ToolName]
		switch {
		case !ok || tool.Execute == nil:
			d.kind = toolDecisionNotFound
		case !tool.RequireApproval:
			d.tool = tool
			d.kind = toolDecisionApproved
		case !approvals.configured():
			d.tool = tool
			d.kind = toolDecisionNoHandler
		default:
			d.tool = tool
			gated = append(gated, i)
		}
		decisions[i] = d
	}

	// Consult the host: one batch invocation covering every gated call, or
	// the per-call handler in tool-call order. Both paths finish before any
	// emission or execution.
	if len(gated) > 0 {
		batchID, err = consultApprovals(ctx, approvals, decisions, gated)
		if err != nil {
			return nil, nil, "", err
		}
	}

	// A deferral in a batch with duplicate or missing tool-call IDs would
	// produce a pause whose pending calls cannot be addressed by decisions —
	// permanently unresumable. Fail here, before any emission or execution,
	// rather than at resume time.
	for i := range decisions {
		if decisions[i].kind != toolDecisionDeferred {
			continue
		}
		if id := decisions[i].tc.ToolCallID; id == "" {
			return nil, nil, "", fmt.Errorf("twilightai: deferred tool call %q has no tool call ID; the pause would be unresumable", decisions[i].tc.ToolName)
		}
		if duplicateIDs {
			return nil, nil, "", fmt.Errorf("twilightai: tool call IDs are not unique in a batch with a deferred approval; the pause would be unresumable")
		}
	}

	// Phase 2: announce decisions and build the result skeleton.
	results = make([]ToolResultPart, len(toolCalls))
	pending := make([]pendingToolExec, 0, len(toolCalls))
	for i := range decisions {
		d := &decisions[i]
		switch d.kind {
		case toolDecisionNotFound:
			results[i] = toolNotFoundResult(d.tc)
		case toolDecisionNoHandler:
			if sendProgress != nil {
				sendProgress(&ToolOutputDeniedPart{
					ToolCallID: d.tc.ToolCallID,
					ToolName:   d.tc.ToolName,
				})
			}
			results[i] = ToolResultPart{
				ToolCallID: d.tc.ToolCallID,
				ToolName:   d.tc.ToolName,
				Result:     "tool execution denied: no approval handler",
				IsError:    true,
			}
		case toolDecisionRejected:
			if sendProgress != nil {
				sendProgress(&ToolApprovalRequestPart{
					ApprovalID: d.approval.ApprovalID,
					ToolCallID: d.tc.ToolCallID,
					ToolName:   d.tc.ToolName,
					Input:      d.tc.Input,
					Metadata:   d.approval.Metadata,
				})
				sendProgress(&ToolOutputDeniedPart{
					ToolCallID: d.tc.ToolCallID,
					ToolName:   d.tc.ToolName,
				})
			}
			results[i] = rejectedToolResult(d.tc, d.approval)
		case toolDecisionDeferred:
			if sendProgress != nil {
				sendProgress(&ToolApprovalRequestPart{
					ApprovalID: d.approval.ApprovalID,
					ToolCallID: d.tc.ToolCallID,
					ToolName:   d.tc.ToolName,
					Input:      d.tc.Input,
					Metadata:   d.approval.Metadata,
				})
			}
			deferred = append(deferred, DeferredToolApproval{ToolCall: d.tc, Approval: d.approval})
		case toolDecisionApproved:
			pending = append(pending, pendingToolExec{idx: i, tc: d.tc, tool: d.tool})
		}
	}

	// Phase 3: execute approved tools in parallel. This runs even when some
	// calls were deferred, so granted approvals are not wasted and their
	// results are recorded before the run pauses.
	runToolsParallel(ctx, pending, results, sendProgress)

	// Phase 4: on a pause, compact the results to the resolved calls only —
	// the deferred slots hold zero values that must not reach a tool message.
	if len(deferred) > 0 {
		resolved := make([]ToolResultPart, 0, len(toolCalls)-len(deferred))
		for i := range decisions {
			if decisions[i].kind != toolDecisionDeferred {
				resolved = append(resolved, results[i])
			}
		}
		return resolved, deferred, batchID, nil
	}

	return results, nil, "", nil
}

// consultApprovals collects the host's decision for every gated call and
// resolves each decision kind in place. It is the only side-effect-free spot
// where the host is consulted; any error leaves decisions unusable and the
// whole batch fails before emission or execution.
func consultApprovals(ctx context.Context, approvals approvalConfig, decisions []toolDecision, gated []int) (string, error) {
	if approvals.batchHandler != nil {
		batchID := newApprovalBatchID()
		if err := consultBatchHandler(ctx, approvals.batchHandler, batchID, decisions, gated); err != nil {
			return "", err
		}
		return batchID, nil
	}
	for _, i := range gated {
		approval, err := approvals.handler(ctx, decisions[i].tc)
		if err != nil {
			return "", fmt.Errorf("twilightai: approval handler for %q: %w", decisions[i].tc.ToolName, err)
		}
		decisions[i].approval = approval
		// The per-call handler keeps its historical convention: an empty
		// Decision from this deliberately-written callback means approve.
		switch approval.Decision {
		case "", ToolApprovalDecisionApproved:
			decisions[i].kind = toolDecisionApproved
		case ToolApprovalDecisionRejected:
			decisions[i].kind = toolDecisionRejected
		case ToolApprovalDecisionDeferred:
			decisions[i].kind = toolDecisionDeferred
		default:
			return "", fmt.Errorf("twilightai: unknown approval decision %q for %q", approval.Decision, decisions[i].tc.ToolName)
		}
	}
	return "", nil
}

// consultBatchHandler invokes the batch handler once for all gated calls and
// verifies the keyed response is complete: exactly one result per call, no
// unknown or duplicate IDs, every Decision explicit. Host assembly order
// does not matter — results are re-associated by ToolCallID, never by
// position. Batch responses come from data assembly (rows, slices), where a
// missed assignment yields the zero value: an empty Decision fails closed,
// never as an implicit approval.
func consultBatchHandler(ctx context.Context, handler func(context.Context, string, []ToolCall) ([]ToolApprovalBatchResult, error), batchID string, decisions []toolDecision, gated []int) error {
	calls := make([]ToolCall, len(gated))
	for j, i := range gated {
		calls[j] = decisions[i].tc
	}
	answers, err := handler(ctx, batchID, calls)
	if err != nil {
		return fmt.Errorf("twilightai: approval batch handler: %w", err)
	}
	byID := make(map[string]ToolApprovalResult, len(answers))
	for _, a := range answers {
		if _, dup := byID[a.ToolCallID]; dup {
			return fmt.Errorf("twilightai: approval batch handler returned duplicate results for tool call %q", a.ToolCallID)
		}
		byID[a.ToolCallID] = a.Result
	}
	for _, i := range gated {
		id := decisions[i].tc.ToolCallID
		approval, ok := byID[id]
		if !ok {
			return fmt.Errorf("twilightai: approval batch handler returned no result for tool call %q", id)
		}
		delete(byID, id)
		decisions[i].approval = approval
		switch approval.Decision {
		case ToolApprovalDecisionApproved:
			decisions[i].kind = toolDecisionApproved
		case ToolApprovalDecisionRejected:
			decisions[i].kind = toolDecisionRejected
		case ToolApprovalDecisionDeferred:
			decisions[i].kind = toolDecisionDeferred
		default:
			return fmt.Errorf("twilightai: approval batch handler decision for %q must be explicitly approved, rejected, or deferred (got %q)", id, approval.Decision)
		}
	}
	for id := range byID {
		return fmt.Errorf("twilightai: approval batch handler returned a result for unknown tool call %q", id)
	}
	return nil
}

// runToolsParallel executes the pending tools, writing each result into
// results at its recorded index. A single tool runs inline; several run
// concurrently.
func runToolsParallel(ctx context.Context, pending []pendingToolExec, results []ToolResultPart, sendProgress func(StreamPart)) {
	if len(pending) == 1 {
		results[pending[0].idx] = runTool(ctx, pending[0].tc, pending[0].tool, sendProgress)
		return
	}
	if len(pending) > 1 {
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
}

func toolNotFoundResult(tc ToolCall) ToolResultPart {
	return ToolResultPart{
		ToolCallID: tc.ToolCallID,
		ToolName:   tc.ToolName,
		Result:     fmt.Sprintf("tool %q not found or has no execute handler", tc.ToolName),
		IsError:    true,
	}
}

func rejectedToolResultText(approval ToolApprovalResult) string {
	if approval.Reason != "" {
		return "tool execution denied by user: " + approval.Reason
	}
	return "tool execution denied by user"
}

// rejectedToolResult is the tool result recorded for a rejected call — the
// single definition of the denial envelope, shared by the in-run rejection
// path and resume-time rejections.
func rejectedToolResult(tc ToolCall, approval ToolApprovalResult) ToolResultPart {
	return ToolResultPart{
		ToolCallID: tc.ToolCallID,
		ToolName:   tc.ToolName,
		Result:     rejectedToolResultText(approval),
		IsError:    true,
	}
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
