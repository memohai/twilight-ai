package sdk

import "github.com/google/jsonschema-go/jsonschema"

type FinishReason string

const (
	FinishReasonStop          FinishReason = "stop"
	FinishReasonLength        FinishReason = "length"
	FinishReasonContentFilter FinishReason = "content-filter"
	FinishReasonToolCalls     FinishReason = "tool-calls"
	FinishReasonError         FinishReason = "error"
	FinishReasonOther         FinishReason = "other"
	FinishReasonUnknown       FinishReason = "unknown"
	// FinishReasonPaused is reported on the overall result when the run
	// stopped because one or more tool calls are awaiting a user decision
	// (see DeferredToolApprovals). Individual steps keep the provider's
	// finish reason; only the run-level result carries this value.
	FinishReasonPaused FinishReason = "paused"
)

type ResponseFormatType string

const (
	ResponseFormatText       ResponseFormatType = "text"
	ResponseFormatJSONObject ResponseFormatType = "json_object"
	ResponseFormatJSONSchema ResponseFormatType = "json_schema"
)

type ResponseFormat struct {
	Type       ResponseFormatType `json:"type"`
	JSONSchema *jsonschema.Schema `json:"jsonSchema,omitempty"`
}

type GenerateParams struct {
	Model *Model `json:"model,omitempty"`
	// System is the stable root instruction placed before the conversation.
	// Use SystemMessage when an instruction belongs at a specific point in the
	// message timeline.
	System   string    `json:"system,omitempty"`
	Messages []Message `json:"messages,omitempty"`

	Tools      []Tool `json:"tools,omitempty"`
	ToolChoice any    `json:"toolChoice,omitempty"` // "auto", "none", "required", or {"type":"function","function":{"name":"..."}}

	ResponseFormat *ResponseFormat `json:"responseFormat,omitempty"`

	Temperature      *float64 `json:"temperature,omitempty"`
	TopP             *float64 `json:"topP,omitempty"`
	MaxTokens        *int     `json:"maxTokens,omitempty"`
	StopSequences    []string `json:"stopSequences,omitempty"`
	FrequencyPenalty *float64 `json:"frequencyPenalty,omitempty"`
	PresencePenalty  *float64 `json:"presencePenalty,omitempty"`
	Seed             *int     `json:"seed,omitempty"`
	ReasoningEffort  *string  `json:"reasoningEffort,omitempty"`
	PromptCacheKey   *string  `json:"promptCacheKey,omitempty"`
}

// StepResult represents the outcome of a single step (one LLM call + tool execution round).
type StepResult struct {
	Text            string           `json:"text"`
	Reasoning       string           `json:"reasoning,omitempty"`
	FinishReason    FinishReason     `json:"finishReason"`
	RawFinishReason string           `json:"rawFinishReason,omitempty"`
	Usage           Usage            `json:"usage"`
	ToolCalls       []ToolCall       `json:"toolCalls,omitempty"`
	ToolResults     []ToolResult     `json:"toolResults,omitempty"`
	Response        ResponseMetadata `json:"response,omitempty"`
	// DeferredToolApprovals lists every tool call in this step awaiting a user
	// decision, in tool-call order. When non-empty, ToolResults and the step's
	// tool message cover only the calls that were already resolved; the caller
	// must supply results for the deferred calls before resuming the run.
	DeferredToolApprovals []DeferredToolApproval `json:"deferredToolApprovals,omitempty"`
	// Messages holds the messages produced by this step (assistant + tool),
	// excluding any prior context from earlier steps.
	Messages []Message `json:"messages,omitempty"`
}

type GenerateResult struct {
	Text                      string           `json:"text"`
	Reasoning                 string           `json:"reasoning,omitempty"`
	ReasoningProviderMetadata map[string]any   `json:"-"`
	FinishReason              FinishReason     `json:"finishReason"`
	RawFinishReason           string           `json:"rawFinishReason,omitempty"`
	Usage                     Usage            `json:"usage"`
	Sources                   []Source         `json:"sources,omitempty"`
	Files                     []GeneratedFile  `json:"files,omitempty"`
	ToolCalls                 []ToolCall       `json:"toolCalls,omitempty"`
	ToolResults               []ToolResult     `json:"toolResults,omitempty"`
	Response                  ResponseMetadata `json:"response,omitempty"`
	// Pause is set when the run stopped on deferred tool approvals
	// (FinishReason == FinishReasonPaused). It is the portable resume state:
	// hand it to ResumeText / ApplyToolDecisions with the user's decisions.
	Pause *ToolApprovalPause `json:"pause,omitempty"`
	// Steps holds the result of each step in a multi-step execution.
	Steps []StepResult `json:"steps,omitempty"`
	// Messages holds all output messages across all steps (assistant + tool),
	// excluding the original input messages.
	Messages []Message `json:"messages,omitempty"`
}
