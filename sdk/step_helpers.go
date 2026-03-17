package sdk

import "fmt"

func buildConfig(options []GenerateOption) (*generateConfig, Provider, error) {
	cfg := &generateConfig{}
	for _, opt := range options {
		opt(cfg)
	}
	if cfg.Params.Model == nil {
		return nil, nil, fmt.Errorf("twilightai: model is required (use WithModel)")
	}
	if cfg.Params.Model.Provider == nil {
		return nil, nil, fmt.Errorf("twilightai: model %q has no provider", cfg.Params.Model.ID)
	}
	return cfg, cfg.Params.Model.Provider, nil
}

func shouldContinueLoop(maxSteps, step int) bool {
	if maxSteps < 0 {
		return true
	}
	return step < maxSteps
}

func addUsage(total, step Usage) Usage {
	total.InputTokens += step.InputTokens
	total.OutputTokens += step.OutputTokens
	total.TotalTokens += step.TotalTokens
	total.ReasoningTokens += step.ReasoningTokens
	total.CachedInputTokens += step.CachedInputTokens
	total.InputTokenDetails.NoCacheTokens += step.InputTokenDetails.NoCacheTokens
	total.InputTokenDetails.CacheReadTokens += step.InputTokenDetails.CacheReadTokens
	total.InputTokenDetails.CacheWriteTokens += step.InputTokenDetails.CacheWriteTokens
	total.OutputTokenDetails.TextTokens += step.OutputTokenDetails.TextTokens
	total.OutputTokenDetails.ReasoningTokens += step.OutputTokenDetails.ReasoningTokens
	return total
}

// buildStepMessages creates the messages produced by a step: an assistant
// message (text/reasoning/tool-calls) and optionally a tool message.
func buildStepMessages(text, reasoning string, toolCalls []ToolCall, toolResults []ToolResultPart) []Message {
	var assistantParts []MessagePart
	if text != "" {
		assistantParts = append(assistantParts, TextPart{Text: text})
	}
	if reasoning != "" {
		assistantParts = append(assistantParts, ReasoningPart{Text: reasoning})
	}
	for _, tc := range toolCalls {
		assistantParts = append(assistantParts, ToolCallPart{
			ToolCallID: tc.ToolCallID,
			ToolName:   tc.ToolName,
			Input:      tc.Input,
		})
	}

	msgs := []Message{{Role: MessageRoleAssistant, Content: assistantParts}}
	if len(toolResults) > 0 {
		msgs = append(msgs, ToolMessage(toolResults...))
	}
	return msgs
}

// applyOnStep calls the OnStep callback and applies the returned override if non-nil.
func applyOnStep(cfg *generateConfig, stepResult *StepResult) {
	if cfg.OnStep == nil {
		return
	}
	if override := cfg.OnStep(stepResult); override != nil {
		cfg.Params = *override
	}
}

// applyPrepareStep calls the PrepareStep callback and applies the returned override if non-nil.
func applyPrepareStep(cfg *generateConfig, messages []Message) []Message {
	if cfg.PrepareStep == nil {
		return messages
	}
	cfg.Params.Messages = messages
	if override := cfg.PrepareStep(&cfg.Params); override != nil {
		cfg.Params = *override
	}
	return cfg.Params.Messages
}
