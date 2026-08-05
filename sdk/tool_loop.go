package sdk

import (
	"context"
	"errors"
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
}

// deferredToolApproval returns the approval of the step that paused the run,
// or nil if the run finished normally.
func (st *toolLoopState) deferredToolApproval() *ToolApprovalResult {
	for i := range st.steps {
		if st.steps[i].DeferredToolApproval != nil {
			return st.steps[i].DeferredToolApproval
		}
	}
	return nil
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
	messages := make([]Message, len(cfg.Params.Messages))
	copy(messages, cfg.Params.Messages)

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

		toolResults, err := executeTools(ctx, out.toolCalls, toolMap, cfg.ApprovalHandler, sendProgress)
		if err != nil {
			var deferred *ToolApprovalDeferredError
			if errors.As(err, &deferred) {
				if _, err := st.commitStep(ctx, cfg, &out, nil, &deferred.Approval); err != nil {
					return st, err
				}
				break
			}
			return st, err
		}

		stepMsgs, err := st.commitStep(ctx, cfg, &out, toolResults, nil)
		if err != nil {
			return st, err
		}
		messages = append(messages, stepMsgs...)
	}

	return st, nil
}

// commitStep assembles the StepResult for one step, passes it through the
// commit barrier, and records it. The step's index is its position in the
// committed sequence. toolResults is nil for final and deferred steps;
// deferredApproval is set only when the step paused on a tool approval.
func (st *toolLoopState) commitStep(
	ctx context.Context,
	cfg *generateConfig,
	out *stepOutcome,
	toolResults []ToolResultPart,
	deferredApproval *ToolApprovalResult,
) ([]Message, error) {
	stepMsgs := buildStepMessages(out.text, out.reasoning, out.reasoningMeta, out.toolCalls, toolResults, &out.usage)
	sr := StepResult{
		Text:                 out.text,
		Reasoning:            out.reasoning,
		FinishReason:         out.finishReason,
		RawFinishReason:      out.rawFinishReason,
		Usage:                out.usage,
		ToolCalls:            out.toolCalls,
		Response:             out.response,
		DeferredToolApproval: deferredApproval,
		Messages:             stepMsgs,
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
