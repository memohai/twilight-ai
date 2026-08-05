package sdk

import (
	"context"
	"errors"
	"fmt"
)

// errLoopAborted signals that the stream consumer went away and the loop
// should stop without reporting an error. Produced and consumed only by this
// transport.
var errLoopAborted = errors.New("twilightai: stream consumer gone")

// StreamText returns a streaming result. When MaxSteps != 0 and tools have
// Execute handlers, the client orchestrates a multi-step loop, forwarding all
// stream parts (including ToolProgressPart) through a single channel.
//
// StreamResult.Steps and StreamResult.Messages are populated during stream
// consumption and safe to read after Stream is fully consumed.
func (c *Client) StreamText(ctx context.Context, options ...GenerateOption) (*StreamResult, error) {
	cfg, prov, err := buildConfig(options)
	if err != nil {
		return nil, err
	}

	// Preserve the direct provider fast path unless the caller requested a
	// commit barrier, which requires the SDK to assemble and validate the step.
	if cfg.MaxSteps == 0 && cfg.OnStepCommitted == nil {
		return prov.DoStream(ctx, cfg.Params)
	}

	// Snapshot the conversation before returning: the caller may reuse or
	// mutate its slice after StreamText returns, and the loop goroutine must
	// not race with that.
	cfg.Params.Messages = append([]Message(nil), cfg.Params.Messages...)

	ch := make(chan StreamPart, 64)
	sr := &StreamResult{Stream: ch}

	go func() {
		send := func(part StreamPart) bool {
			select {
			case ch <- part:
				return true
			case <-ctx.Done():
				return false
			}
		}

		// The shared loop owns the state machine; this transport performs one
		// DoStream call per step, forwarding every part to the consumer while
		// folding them into the step outcome.
		var st toolLoopState
		defer func() {
			sr.Steps = st.steps
			sr.Messages = st.messages
			sr.DeferredToolApproval = st.deferredToolApproval()
			close(ch)
		}()

		doStep := func(stepIndex int, params GenerateParams) (stepOutcome, error) {
			provSR, err := prov.DoStream(ctx, params)
			if err != nil {
				return stepOutcome{}, fmt.Errorf("twilightai: stream step %d: %w", stepIndex, err)
			}

			var out stepOutcome
			sawFinishStep := false
			for part := range provSR.Stream {
				switch p := part.(type) {
				case *TextDeltaPart:
					out.text += p.Text
				case *ReasoningDeltaPart:
					out.reasoning += p.Text
				case *ReasoningEndPart:
					if p.ProviderMetadata != nil {
						out.reasoningMeta = p.ProviderMetadata
					}
				case *StreamToolCallPart:
					out.toolCalls = append(out.toolCalls, ToolCall{
						ToolCallID:       p.ToolCallID,
						ToolName:         p.ToolName,
						Input:            p.Input,
						ProviderMetadata: p.ProviderMetadata,
					})
				case *FinishStepPart:
					sawFinishStep = true
					out.usage = p.Usage
					out.response = p.Response
					out.finishReason = p.FinishReason
					out.rawFinishReason = p.RawFinishReason
				case *FinishPart:
					out.finishReason = p.FinishReason
					out.rawFinishReason = p.RawFinishReason
					continue
				}

				if !send(part) {
					return stepOutcome{}, errLoopAborted
				}
			}
			if !sawFinishStep {
				if ctx.Err() != nil {
					return stepOutcome{}, errLoopAborted
				}
				return stepOutcome{}, fmt.Errorf("twilightai: stream step %d ended before finish-step", stepIndex)
			}
			return out, nil
		}

		var err error
		st, err = runToolLoop(ctx, cfg, doStep, func(part StreamPart) { send(part) })
		if err != nil {
			if !errors.Is(err, errLoopAborted) {
				send(&ErrorPart{Error: err})
			}
			return
		}

		if !send(&FinishPart{
			FinishReason:    st.finishReason,
			RawFinishReason: st.rawFinishReason,
			TotalUsage:      st.totalUsage,
		}) {
			return
		}

		if cfg.OnFinish != nil {
			cfg.OnFinish(&GenerateResult{
				FinishReason:         st.finishReason,
				RawFinishReason:      st.rawFinishReason,
				Usage:                st.totalUsage,
				Steps:                st.steps,
				Messages:             st.messages,
				DeferredToolApproval: st.deferredToolApproval(),
			})
		}
	}()

	return sr, nil
}
