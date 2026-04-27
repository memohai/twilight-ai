package sdk

import (
	"context"
	"errors"
)

func (c *Client) GenerateText(ctx context.Context, options ...GenerateOption) (string, error) {
	result, err := c.GenerateTextResult(ctx, options...)
	if err != nil {
		return "", err
	}
	return result.Text, nil
}

// GenerateTextResult returns the full generation result, supporting multi-step
// tool execution when MaxSteps != 0.
func (c *Client) GenerateTextResult(ctx context.Context, options ...GenerateOption) (*GenerateResult, error) {
	cfg, prov, err := buildConfig(options)
	if err != nil {
		return nil, err
	}

	// MaxSteps == 0: single call, no tool auto-execution.
	if cfg.MaxSteps == 0 {
		result, err := prov.DoGenerate(ctx, cfg.Params)
		if err != nil {
			return nil, err
		}
		stepMsgs := buildStepMessages(result.Text, result.Reasoning, result.ReasoningProviderMetadata, result.ToolCalls, nil, &result.Usage)
		step := StepResult{
			Text:            result.Text,
			Reasoning:       result.Reasoning,
			FinishReason:    result.FinishReason,
			RawFinishReason: result.RawFinishReason,
			Usage:           result.Usage,
			ToolCalls:       result.ToolCalls,
			Response:        result.Response,
			Messages:        stepMsgs,
		}
		result.Steps = []StepResult{step}
		result.Messages = stepMsgs
		applyOnStep(cfg, &step)
		if cfg.OnFinish != nil {
			cfg.OnFinish(result)
		}
		return result, nil
	}

	toolMap := buildToolMap(cfg.Params.Tools)
	messages := make([]Message, len(cfg.Params.Messages))
	copy(messages, cfg.Params.Messages)

	var (
		totalUsage  Usage
		lastResult  *GenerateResult
		allSteps    []StepResult
		allMessages []Message
	)

	for step := 0; shouldContinueLoop(cfg.MaxSteps, step); step++ {
		if step > 0 {
			messages = applyPrepareStep(cfg, messages)
		}

		params := cfg.Params
		params.Messages = messages

		result, err := prov.DoGenerate(ctx, params)
		if err != nil {
			return nil, err
		}
		lastResult = result
		totalUsage = addUsage(&totalUsage, &result.Usage)

		// No tool calls or not a tool-calls finish → final step
		if result.FinishReason != FinishReasonToolCalls || len(result.ToolCalls) == 0 || !hasExecutableTools(result.ToolCalls, toolMap) {
			stepMsgs := buildStepMessages(result.Text, result.Reasoning, result.ReasoningProviderMetadata, result.ToolCalls, nil, &result.Usage)
			sr := StepResult{
				Text:            result.Text,
				Reasoning:       result.Reasoning,
				FinishReason:    result.FinishReason,
				RawFinishReason: result.RawFinishReason,
				Usage:           result.Usage,
				ToolCalls:       result.ToolCalls,
				Response:        result.Response,
				Messages:        stepMsgs,
			}
			allSteps = append(allSteps, sr)
			allMessages = append(allMessages, stepMsgs...)
			applyOnStep(cfg, &sr)
			break
		}

		// Execute tools
		toolResults, err := executeTools(ctx, result.ToolCalls, toolMap, cfg.ApprovalHandler, nil)
		if err != nil {
			var deferred *ToolApprovalDeferredError
			if errors.As(err, &deferred) {
				stepMsgs := buildStepMessages(result.Text, result.Reasoning, result.ReasoningProviderMetadata, result.ToolCalls, nil, &result.Usage)
				sr := StepResult{
					Text:                 result.Text,
					Reasoning:            result.Reasoning,
					FinishReason:         result.FinishReason,
					RawFinishReason:      result.RawFinishReason,
					Usage:                result.Usage,
					ToolCalls:            result.ToolCalls,
					Response:             result.Response,
					DeferredToolApproval: &deferred.Approval,
					Messages:             stepMsgs,
				}
				allSteps = append(allSteps, sr)
				allMessages = append(allMessages, stepMsgs...)
				applyOnStep(cfg, &sr)
				result.DeferredToolApproval = &deferred.Approval
				break
			}
			return nil, err
		}

		stepMsgs := buildStepMessages(result.Text, result.Reasoning, result.ReasoningProviderMetadata, result.ToolCalls, toolResults, &result.Usage)
		sr := StepResult{
			Text:            result.Text,
			Reasoning:       result.Reasoning,
			FinishReason:    result.FinishReason,
			RawFinishReason: result.RawFinishReason,
			Usage:           result.Usage,
			ToolCalls:       result.ToolCalls,
			ToolResults:     toolCallResultsFromParts(toolResults),
			Response:        result.Response,
			Messages:        stepMsgs,
		}
		allSteps = append(allSteps, sr)
		allMessages = append(allMessages, stepMsgs...)
		applyOnStep(cfg, &sr)

		messages = append(messages, stepMsgs...)
	}

	if lastResult != nil {
		lastResult.Usage = totalUsage
		lastResult.Steps = allSteps
		lastResult.Messages = allMessages
		if lastResult.DeferredToolApproval == nil {
			for i := range allSteps {
				if allSteps[i].DeferredToolApproval != nil {
					lastResult.DeferredToolApproval = allSteps[i].DeferredToolApproval
					break
				}
			}
		}
	}

	if cfg.OnFinish != nil && lastResult != nil {
		cfg.OnFinish(lastResult)
	}

	return lastResult, nil
}
