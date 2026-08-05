package sdk

import (
	"context"
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

	// The shared loop owns the state machine for every configuration —
	// MaxSteps == 0 runs it for exactly one call with no tool execution.
	// This transport only performs blocking provider calls and keeps the raw
	// provider result so the final return value preserves provider-specific
	// fields (sources, files, response metadata).
	var lastResult *GenerateResult
	doStep := func(_ int, params GenerateParams) (stepOutcome, error) {
		result, err := prov.DoGenerate(ctx, params)
		if err != nil {
			return stepOutcome{}, err
		}
		lastResult = result
		return stepOutcome{
			text:            result.Text,
			reasoning:       result.Reasoning,
			reasoningMeta:   result.ReasoningProviderMetadata,
			toolCalls:       result.ToolCalls,
			usage:           result.Usage,
			response:        result.Response,
			finishReason:    result.FinishReason,
			rawFinishReason: result.RawFinishReason,
		}, nil
	}

	st, err := runToolLoop(ctx, cfg, doStep, nil)
	if err != nil {
		return nil, err
	}

	if lastResult != nil {
		lastResult.Usage = st.totalUsage
		lastResult.Steps = st.steps
		lastResult.Messages = st.messages
		if lastResult.DeferredToolApproval == nil {
			lastResult.DeferredToolApproval = st.deferredToolApproval()
		}
	}

	if cfg.OnFinish != nil && lastResult != nil {
		cfg.OnFinish(lastResult)
	}

	return lastResult, nil
}
