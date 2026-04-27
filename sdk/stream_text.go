package sdk

import (
	"context"
	"errors"
	"fmt"
)

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

	// Single-step fast path: delegate directly to provider.
	if cfg.MaxSteps == 0 {
		return prov.DoStream(ctx, cfg.Params)
	}

	toolMap := buildToolMap(cfg.Params.Tools)
	messages := make([]Message, len(cfg.Params.Messages))
	copy(messages, cfg.Params.Messages)

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

		var totalUsage Usage
		var lastFinishReason FinishReason
		var lastRawFinishReason string
		var allSteps []StepResult
		var allMessages []Message

		for step := 0; shouldContinueLoop(cfg.MaxSteps, step); step++ {
			if step > 0 {
				messages = applyPrepareStep(cfg, messages)
			}

			params := cfg.Params
			params.Messages = messages

			provSR, err := prov.DoStream(ctx, params)
			if err != nil {
				send(&ErrorPart{Error: fmt.Errorf("twilightai: stream step %d: %w", step, err)})
				break
			}

			var (
				stepText          string
				stepReasoning     string
				stepReasoningMeta map[string]any
				stepToolCalls     []ToolCall
				stepUsage         Usage
				stepResponse      ResponseMetadata
			)

			for part := range provSR.Stream {
				switch p := part.(type) {
				case *TextDeltaPart:
					stepText += p.Text
				case *ReasoningDeltaPart:
					stepReasoning += p.Text
				case *ReasoningEndPart:
					if p.ProviderMetadata != nil {
						stepReasoningMeta = p.ProviderMetadata
					}
				case *StreamToolCallPart:
					stepToolCalls = append(stepToolCalls, ToolCall{
						ToolCallID: p.ToolCallID,
						ToolName:   p.ToolName,
						Input:      p.Input,
					})
				case *FinishStepPart:
					stepUsage = p.Usage
					stepResponse = p.Response
					lastFinishReason = p.FinishReason
					lastRawFinishReason = p.RawFinishReason
				case *FinishPart:
					lastFinishReason = p.FinishReason
					lastRawFinishReason = p.RawFinishReason
					continue
				}

				if !send(part) {
					break
				}
			}

			totalUsage = addUsage(&totalUsage, &stepUsage)

			// No tool calls or not a tool-calls finish → done
			if lastFinishReason != FinishReasonToolCalls || len(stepToolCalls) == 0 || !hasExecutableTools(stepToolCalls, toolMap) {
				stepMsgs := buildStepMessages(stepText, stepReasoning, stepReasoningMeta, stepToolCalls, nil, &stepUsage)
				stepR := StepResult{
					Text:            stepText,
					Reasoning:       stepReasoning,
					FinishReason:    lastFinishReason,
					RawFinishReason: lastRawFinishReason,
					Usage:           stepUsage,
					ToolCalls:       stepToolCalls,
					Response:        stepResponse,
					Messages:        stepMsgs,
				}
				allSteps = append(allSteps, stepR)
				allMessages = append(allMessages, stepMsgs...)
				applyOnStep(cfg, &stepR)
				break
			}

			// Execute tools
			sendProgress := func(part StreamPart) { send(part) }
			toolResults, err := executeTools(ctx, stepToolCalls, toolMap, cfg.ApprovalHandler, sendProgress)
			if err != nil {
				var deferred *ToolApprovalDeferredError
				if errors.As(err, &deferred) {
					stepMsgs := buildStepMessages(stepText, stepReasoning, stepReasoningMeta, stepToolCalls, nil, &stepUsage)
					stepR := StepResult{
						Text:                 stepText,
						Reasoning:            stepReasoning,
						FinishReason:         lastFinishReason,
						RawFinishReason:      lastRawFinishReason,
						Usage:                stepUsage,
						ToolCalls:            stepToolCalls,
						Response:             stepResponse,
						DeferredToolApproval: &deferred.Approval,
						Messages:             stepMsgs,
					}
					allSteps = append(allSteps, stepR)
					allMessages = append(allMessages, stepMsgs...)
					applyOnStep(cfg, &stepR)
					break
				}
				send(&ErrorPart{Error: err})
				break
			}

			stepMsgs := buildStepMessages(stepText, stepReasoning, stepReasoningMeta, stepToolCalls, toolResults, &stepUsage)
			stepR := StepResult{
				Text:            stepText,
				Reasoning:       stepReasoning,
				FinishReason:    lastFinishReason,
				RawFinishReason: lastRawFinishReason,
				Usage:           stepUsage,
				ToolCalls:       stepToolCalls,
				ToolResults:     toolCallResultsFromParts(toolResults),
				Response:        stepResponse,
				Messages:        stepMsgs,
			}
			allSteps = append(allSteps, stepR)
			allMessages = append(allMessages, stepMsgs...)
			applyOnStep(cfg, &stepR)

			messages = append(messages, stepMsgs...)
		}

		// Populate StreamResult fields before closing the channel.
		// Channel close happens-before the consumer's range-loop exit,
		// so these are safe to read after consumption.
		sr.Steps = allSteps
		sr.Messages = allMessages
		for i := range allSteps {
			if allSteps[i].DeferredToolApproval != nil {
				sr.DeferredToolApproval = allSteps[i].DeferredToolApproval
				break
			}
		}

		send(&FinishPart{
			FinishReason:    lastFinishReason,
			RawFinishReason: lastRawFinishReason,
			TotalUsage:      totalUsage,
		})

		if cfg.OnFinish != nil {
			cfg.OnFinish(&GenerateResult{
				FinishReason:         lastFinishReason,
				RawFinishReason:      lastRawFinishReason,
				Usage:                totalUsage,
				Steps:                allSteps,
				Messages:             allMessages,
				DeferredToolApproval: sr.DeferredToolApproval,
			})
		}

		close(ch)
	}()

	return sr, nil
}
