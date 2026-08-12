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
	OnStepCommitted func(ctx context.Context, stepIndex int, step *StepResult) error
	PrepareStep     func(*GenerateParams) *GenerateParams
	ApprovalHandler func(ctx context.Context, call ToolCall) (ToolApprovalResult, error)
	// ApprovalBatchHandler answers all approval-gated calls of one model step
	// in a single invocation. Mutually exclusive with ApprovalHandler.
	ApprovalBatchHandler func(ctx context.Context, batchID string, calls []ToolCall) ([]ToolApprovalBatchResult, error)
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

// WithSystem sets the stable root instruction placed before the conversation.
// Use SystemMessage inside WithMessages for an instruction at a specific point
// in the message timeline.
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

func WithPromptCacheKey(key string) GenerateOption {
	return func(c *generateConfig) { c.Params.PromptCacheKey = &key }
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

// WithOnStepCommitted registers a synchronous commit barrier invoked after a
// complete step has been assembled, including tool results or a deferred
// approval marker, and before the SDK accepts the step or starts the next model
// call. stepIndex is zero-based. Returning an error stops generation and leaves
// the step uncommitted. A deferred-approval Pause is frozen before this callback
// and is not rewritten by callback mutations.
func WithOnStepCommitted(fn func(ctx context.Context, stepIndex int, step *StepResult) error) GenerateOption {
	return func(c *generateConfig) { c.OnStepCommitted = fn }
}

// WithPrepareStep registers a callback invoked before each step (starting from
// the second step). It receives the current params and may return new params to
// override them. Returning nil keeps the (possibly mutated) original params.
func WithPrepareStep(fn func(*GenerateParams) *GenerateParams) GenerateOption {
	return func(c *generateConfig) { c.PrepareStep = fn }
}

// WithApprovalHandler registers a function that decides how to handle a tool
// call marked with RequireApproval.
func WithApprovalHandler(fn func(ctx context.Context, call ToolCall) (ToolApprovalResult, error)) GenerateOption {
	return func(c *generateConfig) { c.ApprovalHandler = fn }
}

// WithApprovalBatchHandler registers a handler that answers all
// approval-gated tool calls of one model step in a single invocation —
// enabling hosts to create every pending approval inside one transaction
// (all-or-nothing) instead of row by row.
//
// batchID is generated by the SDK, is unique per invocation, and reappears as
// ToolApprovalPause.BatchID when any call defers. It gives the host a stable
// correlation key between that handler invocation and a resulting pause. It
// is not an idempotency key — retrying a step mints a new one — and the absence
// of a pause alone does not establish that host records are orphaned because a
// fully approved or rejected batch also completes without one.
//
// calls lists the step's approval-gated, executable calls in tool-call order.
// Before invoking the handler, the SDK verifies those calls have non-empty
// IDs and that every non-empty tool-call ID in the step is unique. The
// response is keyed by ToolCallID, and the SDK verifies it is complete:
// exactly one result per call, no unknown or duplicate IDs, and every Decision
// explicitly approved, rejected, or deferred — the zero value is an error
// here, never an implicit approval (a missed assignment in host code must not
// approve a gated tool).
// A handler error fails the whole batch before SDK approval/tool-execution
// events or tool execution; provider stream events may already have been
// forwarded. A transactional handler can additionally roll back its own
// writes.
//
// Mutually exclusive with WithApprovalHandler — configuring both is an
// error. Hosts without transactional persistence requirements can keep the
// simpler per-call handler.
func WithApprovalBatchHandler(fn func(ctx context.Context, batchID string, calls []ToolCall) ([]ToolApprovalBatchResult, error)) GenerateOption {
	return func(c *generateConfig) { c.ApprovalBatchHandler = fn }
}

// WithApprovalHandlerBool adapts the original bool-based approval callback.
func WithApprovalHandlerBool(fn func(ctx context.Context, call ToolCall) (bool, error)) GenerateOption {
	return func(c *generateConfig) {
		c.ApprovalHandler = func(ctx context.Context, call ToolCall) (ToolApprovalResult, error) {
			approved, err := fn(ctx, call)
			if err != nil {
				return ToolApprovalResult{}, err
			}
			if approved {
				return ToolApprovalResult{Decision: ToolApprovalDecisionApproved}, nil
			}
			return ToolApprovalResult{Decision: ToolApprovalDecisionRejected}, nil
		}
	}
}
