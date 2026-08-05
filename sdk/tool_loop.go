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
	// input is the conversation the run started from, snapshotted before the
	// first step. PrepareStep and OnStep overrides mutate cfg.Params during
	// the loop, so the pause must be built from this snapshot — never from
	// cfg.Params.Messages after the loop has run.
	input []Message
	// system is the run's root instruction, snapshotted alongside input so a
	// pause carries everything needed to resume in another process.
	system string
}

// pause returns the portable resume state when the run stopped on deferred
// approvals, or nil if it finished normally. Only the last committed step can
// carry deferrals: the loop breaks immediately after committing one.
func (st *toolLoopState) pause() *ToolApprovalPause {
	n := len(st.steps)
	if n == 0 || len(st.steps[n-1].DeferredToolApprovals) == 0 {
		return nil
	}
	msgs := make([]Message, 0, len(st.input)+len(st.messages))
	msgs = append(msgs, st.input...)
	msgs = append(msgs, st.messages...)
	// Copy Pending so mutations by the host cannot rewrite the step record
	// (the StepResult holds the same backing array otherwise).
	pending := make([]DeferredToolApproval, len(st.steps[n-1].DeferredToolApprovals))
	copy(pending, st.steps[n-1].DeferredToolApprovals)
	return &ToolApprovalPause{
		System:   st.system,
		Messages: msgs,
		Pending:  pending,
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
	// Snapshot the pristine input and system prompt before the first step:
	// PrepareStep/OnStep overrides mutate cfg.Params mid-loop, and the pause
	// must be reconstructable from what the run actually started with.
	st.input = make([]Message, len(cfg.Params.Messages))
	copy(st.input, cfg.Params.Messages)
	st.system = cfg.Params.System
	// The working conversation starts as an alias of the snapshot; the alias
	// is safe because len==cap, so the first append reallocates, and
	// PrepareStep (the only in-place mutator) runs no earlier than the
	// iteration after that append.
	messages := st.input

	for iter := 0; shouldContinueLoop(maxSteps, iter); iter++ {
		// Prepare before every model call that follows a committed step.
		if len(st.steps) > 0 {
			messages = applyPrepareStep(cfg, messages)
		}

		params := cfg.Params
		params.Messages = messages

		out, err := doStep(len(st.steps), params)
		if err != nil {
			return st, err
		}
		st.finishReason = out.finishReason
		st.rawFinishReason = out.rawFinishReason
		st.totalUsage = addUsage(&st.totalUsage, &out.usage)

		// No executable tool calls (or execution disabled) → final step.
		if !autoExecuteTools || out.finishReason != FinishReasonToolCalls || len(out.toolCalls) == 0 || !hasExecutableTools(out.toolCalls, toolMap) {
			if _, err := st.commitStep(ctx, cfg, &out, nil, nil); err != nil {
				return st, err
			}
			break
		}

		toolResults, deferred, err := executeTools(ctx, out.toolCalls, toolMap, cfg.ApprovalHandler, sendProgress)
		if err != nil {
			return st, err
		}

		// When calls were deferred, the step's tool message carries only the
		// results already resolved in this batch; the run pauses and reports
		// FinishReasonPaused so callers can tell a pause from a normal
		// tool-calls finish. The step itself keeps the provider's finish
		// reason — it describes the model call, not the run.
		stepMsgs, err := st.commitStep(ctx, cfg, &out, toolResults, deferred)
		if err != nil {
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
	toolResults []ToolResultPart,
	deferred []DeferredToolApproval,
) ([]Message, error) {
	stepMsgs := buildStepMessages(out.text, out.reasoning, out.reasoningMeta, out.toolCalls, toolResults, &out.usage)
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
	if err := applyOnStepCommitted(ctx, cfg, len(st.steps), &sr); err != nil {
		return nil, err
	}
	st.steps = append(st.steps, sr)
	st.messages = append(st.messages, stepMsgs...)
	applyOnStep(cfg, &sr)
	return stepMsgs, nil
}
