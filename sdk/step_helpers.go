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
	for i := range cfg.Params.Tools {
		schema, err := resolveSchema(cfg.Params.Tools[i].Parameters)
		if err != nil {
			return nil, nil, fmt.Errorf("twilightai: tool %q: %w", cfg.Params.Tools[i].Name, err)
		}
		cfg.Params.Tools[i].Parameters = schema
	}
	return cfg, cfg.Params.Model.Provider, nil
}

func shouldContinueLoop(maxSteps, step int) bool {
	if maxSteps < 0 {
		return true
	}
	return step < maxSteps
}

func addUsage(total, step *Usage) Usage {
	result := *total
	result.InputTokens += step.InputTokens
	result.OutputTokens += step.OutputTokens
	result.TotalTokens += step.TotalTokens
	result.ReasoningTokens += step.ReasoningTokens
	result.CachedInputTokens += step.CachedInputTokens
	result.InputTokenDetails.NoCacheTokens += step.InputTokenDetails.NoCacheTokens
	result.InputTokenDetails.CacheReadTokens += step.InputTokenDetails.CacheReadTokens
	result.InputTokenDetails.CacheWriteTokens += step.InputTokenDetails.CacheWriteTokens
	result.OutputTokenDetails.TextTokens += step.OutputTokenDetails.TextTokens
	result.OutputTokenDetails.ReasoningTokens += step.OutputTokenDetails.ReasoningTokens
	return result
}

// buildStepMessages creates the messages produced by a step: an assistant
// message (text/reasoning/tool-calls) and optionally a tool message.
// The usage is attached to the assistant message for output tracking.
func buildStepMessages(text, reasoning string, reasoningMeta map[string]any, toolCalls []ToolCall, toolResults []ToolResultPart, usage *Usage) []Message {
	var assistantParts []MessagePart
	if reasoning != "" {
		assistantParts = append(assistantParts, ReasoningPart{Text: reasoning, ProviderMetadata: reasoningMeta})
	}
	if text != "" {
		assistantParts = append(assistantParts, TextPart{Text: text})
	}
	for _, tc := range toolCalls {
		assistantParts = append(assistantParts, ToolCallPart{
			ToolCallID: tc.ToolCallID,
			ToolName:   tc.ToolName,
			Input:      tc.Input,
		})
	}

	msgs := []Message{{Role: MessageRoleAssistant, Content: assistantParts, Usage: usage}}
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
