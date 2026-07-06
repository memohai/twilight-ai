package images

const (
	taskStatusSucceeded = "SUCCEEDED"
	taskStatusFailed    = "FAILED"
	taskStatusUnknown   = "UNKNOWN"
	taskStatusCanceled  = "CANCELED"
)

type dashScopeImageRequest struct {
	Model      string                   `json:"model"`
	Input      dashScopeImageInput      `json:"input"`
	Parameters dashScopeImageParameters `json:"parameters,omitempty"`
}

type dashScopePromptInputRequest struct {
	Model      string                   `json:"model"`
	Input      dashScopePromptInput     `json:"input"`
	Parameters dashScopeImageParameters `json:"parameters,omitempty"`
}

type dashScopeImageInput struct {
	Messages []dashScopeImageMessage `json:"messages"`
}

type dashScopePromptInput struct {
	Prompt string `json:"prompt"`
}

type dashScopeImageMessage struct {
	Role    string                  `json:"role"`
	Content []dashScopeImageContent `json:"content"`
}

type dashScopeImageContent struct {
	Text  string `json:"text,omitempty"`
	Image string `json:"image,omitempty"`
	Type  string `json:"type,omitempty"`
}

type dashScopeImageParameters struct {
	N    *int   `json:"n,omitempty"`
	Size string `json:"size,omitempty"`
}

type dashScopeResponse struct {
	RequestID string          `json:"request_id,omitempty"`
	Output    dashScopeOutput `json:"output"`
	Usage     *dashScopeUsage `json:"usage,omitempty"`
	Code      string          `json:"code,omitempty"`
	Message   string          `json:"message,omitempty"`
}

type dashScopeOutput struct {
	TaskID     string                 `json:"task_id,omitempty"`
	TaskStatus string                 `json:"task_status,omitempty"`
	Choices    []dashScopeImageChoice `json:"choices,omitempty"`
	Results    []dashScopeImageResult `json:"results,omitempty"`
	Code       string                 `json:"code,omitempty"`
	Message    string                 `json:"message,omitempty"`
}

type dashScopeImageChoice struct {
	FinishReason string                `json:"finish_reason,omitempty"`
	Message      dashScopeImageMessage `json:"message"`
}

type dashScopeImageResult struct {
	URL          string `json:"url,omitempty"`
	Code         string `json:"code,omitempty"`
	Message      string `json:"message,omitempty"`
	OrigPrompt   string `json:"orig_prompt,omitempty"`
	ActualPrompt string `json:"actual_prompt,omitempty"`
}

type dashScopeUsage struct {
	TotalTokens  int `json:"total_tokens,omitempty"`
	InputTokens  int `json:"input_tokens,omitempty"`
	OutputTokens int `json:"output_tokens,omitempty"`
	ImageCount   int `json:"image_count,omitempty"`
}
