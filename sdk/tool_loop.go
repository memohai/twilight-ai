package sdk

import (
	"context"
)

// stepOutcome is the transport-agnostic product of one provider call.
// GenerateTextResult maps a DoGenerate response onto it directly; StreamText
// folds the part stream into it while forwarding parts to the consumer.
type stepOutcome struct {
	text            string
	reasoning       string
	reasoningMeta   map[string]any
	toolCalls       []ToolCall
	usage           Usage
	response        ResponseMetadata
	finishReason    FinishReason
	rawFinishReason string
}

// pauseSnapshot captures a complete pause before the commit barrier runs.
// Approval requests are announced and sibling side effects may have happened
// by then, so the snapshot must survive even a barrier failure.
type pauseSnapshot struct {
	deferred []DeferredToolApproval
	batchID  string
	// messages is the effective conversation sent to the provider for the
	// paused step, followed by that step's assistant/tool messages. Keeping the
	// complete snapshot matters when PrepareStep replaced or compacted history.
	messages []Message
	system   string
}

// toolLoopState accumulates what a multi-step run produces. When runToolLoop
// returns an error the state still holds everything committed before the
// failing step; the streaming entry point exposes it, the blocking entry
// point discards it (matching their historical behavior).
type toolLoopState struct {
	steps           []StepResult
	messages        []Message
	totalUsage      Usage
	finishReason    FinishReason
	rawFinishReason string
	// pendingPause is set the moment a step defers, before the commit
	// barrier — the pause's source of truth whether or not the step commits.
	pendingPause *pauseSnapshot
}

// pause returns the portable paused state when the run stopped on deferred
// approvals, or nil if it finished normally. It is available even when the
// paused step's commit barrier failed: by that point the approval requests
// were announced and sibling side effects may have happened, so the pause is the
// host's only handle for reconciliation.
func (st *toolLoopState) pause() *ToolApprovalPause {
	if st.pendingPause == nil {
		return nil
	}
	msgs := clonePauseMessages(st.pendingPause.messages)
	// Copy Pending so mutations by the host cannot rewrite the step record
	// (the StepResult holds the same backing array otherwise).
	pending := cloneDeferredToolApprovals(st.pendingPause.deferred)
	return &ToolApprovalPause{
		BatchID:  st.pendingPause.batchID,
		System:   st.pendingPause.system,
		Messages: msgs,
		Pending:  pending,
	}
}

// recordPause snapshots the exact context used for the paused provider call
// and appends that call's output. It runs before the commit barrier so the
// host can still reconcile a pause when committing the step fails.
func (st *toolLoopState) recordPause(system string, inputMessages, stepMessages []Message, deferred []DeferredToolApproval, batchID string) {
	messages := make([]Message, 0, len(inputMessages)+len(stepMessages))
	messages = append(messages, inputMessages...)
	messages = append(messages, stepMessages...)
	st.pendingPause = &pauseSnapshot{
		deferred: cloneDeferredToolApprovals(deferred),
		batchID:  batchID,
		messages: clonePauseMessages(messages),
		system:   system,
	}
}

// runToolLoop drives the multi-step tool-execution state machine shared by
// GenerateTextResult and StreamText: call the model, execute requested tools,
// feed results back, repeat until a non-tool finish, the step budget, a
// deferred approval, or an error.
//
// The loop derives its behavior from cfg so the two transports cannot
// diverge: cfg.MaxSteps == 0 means one provider call with no tool
// auto-execution (the documented single-call mode; it reaches the loop when
// a commit barrier or other loop-only feature is requested).
//
// doStep performs one provider call and folds it into a stepOutcome; the
// streaming transport forwards parts to its consumer as a side effect and
// returns an error wrapping context cancellation when the consumer is gone.
// sendProgress forwards tool-execution parts (nil in blocking mode).
//
// Step indices are derived from committed state (len of steps), never from a
// parallel counter: the index doStep and OnStepCommitted observe is always
// the position the step will occupy in the result.
func runToolLoop(
	ctx context.Context,
	cfg *generateConfig,
	doStep func(stepIndex int, params GenerateParams) (stepOutcome, error),
	sendProgress func(StreamPart),
) (toolLoopState, error) {
	var st toolLoopState
	autoExecuteTools := cfg.MaxSteps != 0
	maxSteps := cfg.MaxSteps
	if maxSteps == 0 {
		maxSteps = 1
	}

	toolMap := buildToolMap(cfg.Params.Tools)
	// Keep the loop's working slice independent from the caller's slice. A
	// pause takes a fresh snapshot again at the exact step where it occurs.
	messages := append([]Message(nil), cfg.Params.Messages...)

	for iter := 0; shouldContinueLoop(maxSteps, iter); iter++ {
		// Prepare before every model call that follows a committed step.
		if len(st.steps) > 0 {
			messages = applyPrepareStep(cfg, messages)
		}

		params := cfg.Params
		params.Messages = messages

		// When this run can pause, freeze the exact conversation at the provider
		// boundary. The provider may retain or transform its request while
		// producing a result; a later pause must still reflect what was submitted.
		var pauseInput []Message
		canPause := (cfg.ApprovalHandler != nil || cfg.ApprovalBatchHandler != nil) && hasConfiguredApprovalGatedTool(toolMap)
		if canPause {
			pauseInput = clonePauseMessages(params.Messages)
		}

		out, err := doStep(len(st.steps), params)
		if err != nil {
			return st, err
		}
		st.finishReason = out.finishReason
		st.rawFinishReason = out.rawFinishReason
		st.totalUsage = addUsage(&st.totalUsage, &out.usage)

		// No executable tool calls (or execution disabled) → final step.
		if !autoExecuteTools || out.finishReason != FinishReasonToolCalls || len(out.toolCalls) == 0 || !hasExecutableTools(out.toolCalls, toolMap) {
			stepMsgs := buildStepMessages(out.text, out.reasoning, out.reasoningMeta, out.toolCalls, nil, &out.usage)
			if err := st.commitStep(ctx, cfg, &out, stepMsgs, nil, nil); err != nil {
				return st, err
			}
			break
		}

		// Preserve the effective provider request and output only when this step
		// can actually defer. These snapshots are not passed to approval callbacks
		// or tools, so their established input ownership semantics remain unchanged.
		var pauseOut *stepOutcome
		if canPause && hasApprovalGatedTools(out.toolCalls, toolMap) {
			cloned := cloneStepOutcome(out)
			pauseOut = &cloned
		}

		toolResults, deferred, batchID, err := executeTools(ctx, out.toolCalls, toolMap,
			approvalConfig{handler: cfg.ApprovalHandler, batchHandler: cfg.ApprovalBatchHandler}, sendProgress)
		if err != nil {
			return st, err
		}

		// When calls were deferred, the step's tool message carries only the
		// results already resolved in this batch; the run pauses and reports
		// FinishReasonPaused so callers can tell a pause from a normal
		// tool-calls finish. The step itself keeps the provider's finish
		// reason — it describes the model call, not the run.
		//
		stepMsgs := buildStepMessages(out.text, out.reasoning, out.reasoningMeta, out.toolCalls, toolResults, &out.usage)
		// Record the complete pause BEFORE the commit barrier: by this point
		// approval requests are announced and sibling side effects may have
		// happened, so even a barrier failure must not lose the pause — it is
		// the only handle the host has to reconcile what already occurred.
		if len(deferred) > 0 {
			// A deferred decision can only come from an approval-gated tool with a
			// configured handler, so pauseOut must have been captured above.
			pauseStepMsgs := buildStepMessages(pauseOut.text, pauseOut.reasoning, pauseOut.reasoningMeta, pauseOut.toolCalls, toolResults, &pauseOut.usage)
			pausePending := deferredWithOriginalCalls(deferred, pauseOut.toolCalls)
			st.recordPause(params.System, pauseInput, pauseStepMsgs, pausePending, batchID)
		}
		if err := st.commitStep(ctx, cfg, &out, stepMsgs, toolResults, deferred); err != nil {
			return st, err
		}
		if len(deferred) > 0 {
			st.finishReason = FinishReasonPaused
			break
		}
		messages = append(messages, stepMsgs...)
	}

	return st, nil
}

// commitStep assembles the StepResult for one step, passes it through the
// commit barrier, and records it. The step's index is its position in the
// committed sequence. toolResults holds the resolved results (nil on final
// steps); deferred lists the calls still awaiting a user decision.
func (st *toolLoopState) commitStep(
	ctx context.Context,
	cfg *generateConfig,
	out *stepOutcome,
	stepMsgs []Message,
	toolResults []ToolResultPart,
	deferred []DeferredToolApproval,
) error {
	sr := StepResult{
		Text:                  out.text,
		Reasoning:             out.reasoning,
		FinishReason:          out.finishReason,
		RawFinishReason:       out.rawFinishReason,
		Usage:                 out.usage,
		ToolCalls:             out.toolCalls,
		Response:              out.response,
		DeferredToolApprovals: deferred,
		Messages:              stepMsgs,
	}
	if len(toolResults) > 0 {
		sr.ToolResults = toolCallResultsFromParts(toolResults)
	}
	// Preserve the callback's established mutation semantics: annotations made
	// by the commit barrier are recorded in Steps and observed by OnStep. A
	// pending pause was snapshotted before this point, so those mutations cannot
	// corrupt its stored conversation.
	if err := applyOnStepCommitted(ctx, cfg, len(st.steps), &sr); err != nil {
		return err
	}
	st.steps = append(st.steps, sr)
	st.messages = append(st.messages, stepMsgs...)
	applyOnStep(cfg, &sr)
	return nil
}
