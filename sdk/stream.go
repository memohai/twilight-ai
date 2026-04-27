package sdk

type StreamPartType string

const (
	StreamPartTypeTextStart           StreamPartType = "text-start"
	StreamPartTypeTextDelta           StreamPartType = "text-delta"
	StreamPartTypeTextEnd             StreamPartType = "text-end"
	StreamPartTypeReasoningStart      StreamPartType = "reasoning-start"
	StreamPartTypeReasoningDelta      StreamPartType = "reasoning-delta"
	StreamPartTypeReasoningEnd        StreamPartType = "reasoning-end"
	StreamPartTypeToolInputStart      StreamPartType = "tool-input-start"
	StreamPartTypeToolInputDelta      StreamPartType = "tool-input-delta"
	StreamPartTypeToolInputEnd        StreamPartType = "tool-input-end"
	StreamPartTypeToolCall            StreamPartType = "tool-call"
	StreamPartTypeToolResult          StreamPartType = "tool-result"
	StreamPartTypeToolError           StreamPartType = "tool-error"
	StreamPartTypeToolOutputDenied    StreamPartType = "tool-output-denied"
	StreamPartTypeToolApprovalRequest StreamPartType = "tool-approval-request"
	StreamPartTypeToolProgress        StreamPartType = "tool-progress"
	StreamPartTypeSource              StreamPartType = "source"
	StreamPartTypeFile                StreamPartType = "file"
	StreamPartTypeStart               StreamPartType = "start"
	StreamPartTypeFinish              StreamPartType = "finish"
	StreamPartTypeStartStep           StreamPartType = "start-step"
	StreamPartTypeFinishStep          StreamPartType = "finish-step"
	StreamPartTypeError               StreamPartType = "error"
	StreamPartTypeAbort               StreamPartType = "abort"
	StreamPartTypeRaw                 StreamPartType = "raw"
)

// StreamPart is the interface implemented by all stream chunk types.
// Consumers should use a type switch to handle specific part types.
type StreamPart interface {
	Type() StreamPartType
}

// --- Text ---

type TextStartPart struct {
	ID               string
	ProviderMetadata map[string]any
}

func (p *TextStartPart) Type() StreamPartType { return StreamPartTypeTextStart }

type TextDeltaPart struct {
	ID               string
	Text             string
	ProviderMetadata map[string]any
}

func (p *TextDeltaPart) Type() StreamPartType { return StreamPartTypeTextDelta }

type TextEndPart struct {
	ID               string
	ProviderMetadata map[string]any
}

func (p *TextEndPart) Type() StreamPartType { return StreamPartTypeTextEnd }

// --- Reasoning ---

type ReasoningStartPart struct {
	ID               string
	ProviderMetadata map[string]any
}

func (p *ReasoningStartPart) Type() StreamPartType { return StreamPartTypeReasoningStart }

type ReasoningDeltaPart struct {
	ID               string
	Text             string
	ProviderMetadata map[string]any
}

func (p *ReasoningDeltaPart) Type() StreamPartType { return StreamPartTypeReasoningDelta }

type ReasoningEndPart struct {
	ID               string
	ProviderMetadata map[string]any
}

func (p *ReasoningEndPart) Type() StreamPartType { return StreamPartTypeReasoningEnd }

// --- Tool Input ---

type ToolInputStartPart struct {
	ID               string
	ToolName         string
	ProviderMetadata map[string]any
}

func (p *ToolInputStartPart) Type() StreamPartType { return StreamPartTypeToolInputStart }

type ToolInputDeltaPart struct {
	ID               string
	Delta            string
	ProviderMetadata map[string]any
}

func (p *ToolInputDeltaPart) Type() StreamPartType { return StreamPartTypeToolInputDelta }

type ToolInputEndPart struct {
	ID               string
	ProviderMetadata map[string]any
}

func (p *ToolInputEndPart) Type() StreamPartType { return StreamPartTypeToolInputEnd }

// --- Tool Execution ---

type StreamToolCallPart struct {
	ToolCallID string
	ToolName   string
	Input      any
}

func (p *StreamToolCallPart) Type() StreamPartType { return StreamPartTypeToolCall }

type StreamToolResultPart struct {
	ToolCallID string
	ToolName   string
	Input      any
	Output     any
}

func (p *StreamToolResultPart) Type() StreamPartType { return StreamPartTypeToolResult }

type StreamToolErrorPart struct {
	ToolCallID string
	ToolName   string
	Error      error
}

func (p *StreamToolErrorPart) Type() StreamPartType { return StreamPartTypeToolError }

type ToolOutputDeniedPart struct {
	ToolCallID string
	ToolName   string
}

func (p *ToolOutputDeniedPart) Type() StreamPartType { return StreamPartTypeToolOutputDenied }

type ToolApprovalRequestPart struct {
	ApprovalID string
	ToolCallID string
	ToolName   string
	Input      any
	Metadata   map[string]any
}

func (p *ToolApprovalRequestPart) Type() StreamPartType { return StreamPartTypeToolApprovalRequest }

// --- Tool Progress (UX streaming during execution) ---

type ToolProgressPart struct {
	ToolCallID string
	ToolName   string
	Content    any
}

func (p *ToolProgressPart) Type() StreamPartType { return StreamPartTypeToolProgress }

// --- Source & File ---

type StreamSourcePart struct {
	Source Source
}

func (p *StreamSourcePart) Type() StreamPartType { return StreamPartTypeSource }

type StreamFilePart struct {
	File GeneratedFile
}

func (p *StreamFilePart) Type() StreamPartType { return StreamPartTypeFile }

// --- Lifecycle ---

type StartPart struct{}

func (p *StartPart) Type() StreamPartType { return StreamPartTypeStart }

type FinishPart struct {
	FinishReason    FinishReason
	RawFinishReason string
	TotalUsage      Usage
}

func (p *FinishPart) Type() StreamPartType { return StreamPartTypeFinish }

type StartStepPart struct{}

func (p *StartStepPart) Type() StreamPartType { return StreamPartTypeStartStep }

type FinishStepPart struct {
	FinishReason     FinishReason
	RawFinishReason  string
	Usage            Usage
	Response         ResponseMetadata
	ProviderMetadata map[string]any
}

func (p *FinishStepPart) Type() StreamPartType { return StreamPartTypeFinishStep }

type ErrorPart struct {
	Error error
}

func (p *ErrorPart) Type() StreamPartType { return StreamPartTypeError }

type AbortPart struct {
	Reason string
}

func (p *AbortPart) Type() StreamPartType { return StreamPartTypeAbort }

type RawPart struct {
	RawValue any
}

func (p *RawPart) Type() StreamPartType { return StreamPartTypeRaw }

// StreamResult holds a channel that yields StreamPart chunks.
// The channel is closed when the stream ends.
//
// Steps and Messages are populated during stream consumption and are safe to
// read after Stream is fully consumed (i.e., after a for-range loop exits).
type StreamResult struct {
	Stream <-chan StreamPart
	// Steps holds the result of each step. Populated as the stream is consumed.
	Steps []StepResult
	// Messages holds all output messages across all steps (assistant + tool),
	// excluding the original input messages. Populated as the stream is consumed.
	Messages             []Message
	DeferredToolApproval *ToolApprovalResult
}

// Text consumes the entire stream and returns the concatenated text content.
func (sr *StreamResult) Text() (string, error) {
	var text string
	for part := range sr.Stream {
		switch p := part.(type) {
		case *TextDeltaPart:
			text += p.Text
		case *ErrorPart:
			return text, p.Error
		}
	}
	return text, nil
}

// ToResult consumes the entire stream and assembles a GenerateResult.
func (sr *StreamResult) ToResult() (*GenerateResult, error) {
	result := &GenerateResult{}
	var reasoning string

	for part := range sr.Stream {
		switch p := part.(type) {
		case *TextDeltaPart:
			result.Text += p.Text
		case *ReasoningDeltaPart:
			reasoning += p.Text
		case *StreamToolCallPart:
			result.ToolCalls = append(result.ToolCalls, ToolCall{
				ToolCallID: p.ToolCallID,
				ToolName:   p.ToolName,
				Input:      p.Input,
			})
		case *StreamToolResultPart:
			result.ToolResults = append(result.ToolResults, ToolResult{
				ToolCallID: p.ToolCallID,
				ToolName:   p.ToolName,
				Input:      p.Input,
				Output:     p.Output,
			})
		case *StreamSourcePart:
			result.Sources = append(result.Sources, p.Source)
		case *StreamFilePart:
			result.Files = append(result.Files, p.File)
		case *FinishStepPart:
			result.Response = p.Response
		case *FinishPart:
			result.FinishReason = p.FinishReason
			result.RawFinishReason = p.RawFinishReason
			result.Usage = p.TotalUsage
		case *ErrorPart:
			return result, p.Error
		}
	}

	result.Reasoning = reasoning
	result.Steps = sr.Steps
	result.Messages = sr.Messages
	result.DeferredToolApproval = sr.DeferredToolApproval
	return result, nil
}
