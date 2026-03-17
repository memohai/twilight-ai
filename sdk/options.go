package sdk

import "context"

// generateConfig holds both the provider-level params and client-level
// orchestration settings (callbacks, max steps, approval handler).
type generateConfig struct {
	Params GenerateParams

	// MaxSteps controls the tool auto-execution loop.
	//   0  = single LLM call, no auto-execution (default, backward-compatible)
	//  >0  = at most N LLM calls
	//  -1  = unlimited loop until LLM stops producing tool calls
	MaxSteps        int
	OnFinish        func(*GenerateResult)
	OnStep          func(*StepResult) *GenerateParams
	PrepareStep     func(*GenerateParams) *GenerateParams
	ApprovalHandler func(ctx context.Context, call ToolCall) (bool, error)
}

// GenerateOption configures a text generation request.
type GenerateOption func(*generateConfig)

// --- Provider-level options ---

func WithModel(model *Model) GenerateOption {
	return func(c *generateConfig) { c.Params.Model = model }
}

func WithMessages(messages []Message) GenerateOption {
	return func(c *generateConfig) { c.Params.Messages = messages }
}

func WithSystem(text string) GenerateOption {
	return func(c *generateConfig) { c.Params.System = text }
}

func WithTools(tools []Tool) GenerateOption {
	return func(c *generateConfig) { c.Params.Tools = tools }
}

func WithToolChoice(choice any) GenerateOption {
	return func(c *generateConfig) { c.Params.ToolChoice = choice }
}

func WithResponseFormat(rf ResponseFormat) GenerateOption {
	return func(c *generateConfig) { c.Params.ResponseFormat = &rf }
}

func WithTemperature(t float64) GenerateOption {
	return func(c *generateConfig) { c.Params.Temperature = &t }
}

func WithTopP(topP float64) GenerateOption {
	return func(c *generateConfig) { c.Params.TopP = &topP }
}

func WithMaxTokens(n int) GenerateOption {
	return func(c *generateConfig) { c.Params.MaxTokens = &n }
}

func WithStopSequences(s []string) GenerateOption {
	return func(c *generateConfig) { c.Params.StopSequences = s }
}

func WithFrequencyPenalty(penalty float64) GenerateOption {
	return func(c *generateConfig) { c.Params.FrequencyPenalty = &penalty }
}

func WithPresencePenalty(penalty float64) GenerateOption {
	return func(c *generateConfig) { c.Params.PresencePenalty = &penalty }
}

func WithSeed(s int) GenerateOption {
	return func(c *generateConfig) { c.Params.Seed = &s }
}

func WithReasoningEffort(effort string) GenerateOption {
	return func(c *generateConfig) { c.Params.ReasoningEffort = &effort }
}

// --- Client-level orchestration options ---

// WithMaxSteps sets the maximum number of LLM calls in the tool-execution loop.
//
//	0  (default) = single call, no auto tool execution
//	N  (N > 0)   = at most N calls
//	-1           = unlimited, loops until LLM stops requesting tools
func WithMaxSteps(n int) GenerateOption {
	return func(c *generateConfig) { c.MaxSteps = n }
}

// WithOnFinish registers a callback invoked once when all steps complete.
func WithOnFinish(fn func(*GenerateResult)) GenerateOption {
	return func(c *generateConfig) { c.OnFinish = fn }
}

// WithOnStep registers a callback invoked after each step (LLM call + tool round).
// If the callback returns a non-nil *GenerateParams, it overrides the params
// for the next step.
func WithOnStep(fn func(*StepResult) *GenerateParams) GenerateOption {
	return func(c *generateConfig) { c.OnStep = fn }
}

// WithPrepareStep registers a callback invoked before each step (starting from
// the second step). It receives the current params and may return new params to
// override them. Returning nil keeps the (possibly mutated) original params.
func WithPrepareStep(fn func(*GenerateParams) *GenerateParams) GenerateOption {
	return func(c *generateConfig) { c.PrepareStep = fn }
}

// WithApprovalHandler registers a function that decides whether a tool call
// marked with RequireApproval should proceed. Return (true, nil) to approve.
func WithApprovalHandler(fn func(ctx context.Context, call ToolCall) (bool, error)) GenerateOption {
	return func(c *generateConfig) { c.ApprovalHandler = fn }
}
