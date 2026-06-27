# Twilight AI API Reference

This file is the detailed API companion for `skill/SKILL.md`.

Use it when the task needs exact package names, exported types, function signatures, provider options, or stream/event shapes.

## Package Map

- `github.com/memohai/twilight-ai/sdk`
- `github.com/memohai/twilight-ai/provider/openai/completions`
- `github.com/memohai/twilight-ai/provider/openai/responses`
- `github.com/memohai/twilight-ai/provider/anthropic/messages`
- `github.com/memohai/twilight-ai/provider/google/generativeai`
- `github.com/memohai/twilight-ai/provider/openai/codex`
- `github.com/memohai/twilight-ai/provider/openai/images`
- `github.com/memohai/twilight-ai/provider/openai/embedding`
- `github.com/memohai/twilight-ai/provider/google/embedding`

## Package `sdk`

### Client And Top-Level Helpers

```go
type Client struct{}

func NewClient() *Client

func (c *Client) GenerateText(ctx context.Context, options ...GenerateOption) (string, error)
func (c *Client) GenerateTextResult(ctx context.Context, options ...GenerateOption) (*GenerateResult, error)
func (c *Client) StreamText(ctx context.Context, options ...GenerateOption) (*StreamResult, error)
func (c *Client) Embed(ctx context.Context, value string, options ...EmbedOption) ([]float64, error)
func (c *Client) EmbedMany(ctx context.Context, values []string, options ...EmbedOption) (*EmbedResult, error)
func (c *Client) GenerateImage(ctx context.Context, options ...ImageGenerateOption) (*ImageResult, error)
func (c *Client) EditImage(ctx context.Context, options ...ImageEditOption) (*ImageResult, error)

func GenerateText(ctx context.Context, options ...GenerateOption) (string, error)
func GenerateTextResult(ctx context.Context, options ...GenerateOption) (*GenerateResult, error)
func StreamText(ctx context.Context, options ...GenerateOption) (*StreamResult, error)
func Embed(ctx context.Context, value string, options ...EmbedOption) ([]float64, error)
func EmbedMany(ctx context.Context, values []string, options ...EmbedOption) (*EmbedResult, error)
func GenerateImage(ctx context.Context, options ...ImageGenerateOption) (*ImageResult, error)
func EditImage(ctx context.Context, options ...ImageEditOption) (*ImageResult, error)
```

### Provider Contracts

```go
type Provider interface {
    Name() string
    ListModels(ctx context.Context) ([]Model, error)
    Test(ctx context.Context) *ProviderTestResult
    TestModel(ctx context.Context, modelID string) (*ModelTestResult, error)
    DoGenerate(ctx context.Context, params GenerateParams) (*GenerateResult, error)
    DoStream(ctx context.Context, params GenerateParams) (*StreamResult, error)
}

type ProviderStatus string

const (
    ProviderStatusOK          ProviderStatus = "ok"
    ProviderStatusUnhealthy   ProviderStatus = "unhealthy"
    ProviderStatusUnreachable ProviderStatus = "unreachable"
)

type ProviderTestResult struct {
    Status  ProviderStatus
    Message string
    Error   error
}

type ModelTestResult struct {
    Supported bool
    Message   string
}
```

### Models

```go
type ModelType string

const ModelTypeChat      ModelType = "chat"
const ModelTypeEmbedding ModelType = "embedding"

type Model struct {
    ID          string
    DisplayName string
    Provider    Provider
    Type        ModelType
    MaxTokens   int
}

func (m *Model) Test(ctx context.Context) (*ModelTestResult, error)
```

### Messages

```go
type MessageRole string

const (
    MessageRoleUser      MessageRole = "user"
    MessageRoleAssistant MessageRole = "assistant"
    MessageRoleSystem    MessageRole = "system"
    MessageRoleTool      MessageRole = "tool"
)

type MessagePartType string

const (
    MessagePartTypeText       MessagePartType = "text"
    MessagePartTypeReasoning  MessagePartType = "reasoning"
    MessagePartTypeImage      MessagePartType = "image"
    MessagePartTypeFile       MessagePartType = "file"
    MessagePartTypeToolCall   MessagePartType = "tool-call"
    MessagePartTypeToolResult MessagePartType = "tool-result"
)

type MessagePart interface {
    PartType() MessagePartType
}

type TextPart struct {
    Text         string
    CacheControl *CacheControl  // optional, Anthropic only
}

type ReasoningPart struct {
    Text      string
    Signature string
}

type ImagePart struct {
    Image        string
    MediaType    string
    CacheControl *CacheControl  // optional, Anthropic only
}

type FilePart struct {
    Data         string
    MediaType    string
    Filename     string
    CacheControl *CacheControl  // optional, Anthropic only
}

type ToolCallPart struct {
    ToolCallID   string
    ToolName     string
    Input        any
    CacheControl *CacheControl  // optional, Anthropic only
}

type ToolResultPart struct {
    ToolCallID   string
    ToolName     string
    Result       any
    IsError      bool
    CacheControl *CacheControl  // optional, Anthropic only
}

// CacheControl marks a content block as an Anthropic prompt-caching breakpoint.
type CacheControl struct {
    Type string  // "ephemeral"
    TTL  string  // "" (5-minute default) | "1h"
}

type Message struct {
    Role    MessageRole
    Content []MessagePart
}

func UserMessage(text string, extra ...MessagePart) Message
func SystemMessage(text string) Message
func AssistantMessage(text string) Message
func ToolMessage(results ...ToolResultPart) Message
```

Notes:

- `UserMessage` accepts a text string plus optional extra parts such as `ImagePart`.
- `Message` supports JSON marshal and unmarshal with type discrimination.

### Generation

```go
type FinishReason string

const (
    FinishReasonStop          FinishReason = "stop"
    FinishReasonLength        FinishReason = "length"
    FinishReasonContentFilter FinishReason = "content-filter"
    FinishReasonToolCalls     FinishReason = "tool-calls"
    FinishReasonError         FinishReason = "error"
    FinishReasonOther         FinishReason = "other"
    FinishReasonUnknown       FinishReason = "unknown"
)

type ResponseFormatType string

const (
    ResponseFormatText       ResponseFormatType = "text"
    ResponseFormatJSONObject ResponseFormatType = "json_object"
    ResponseFormatJSONSchema ResponseFormatType = "json_schema"
)

type ResponseFormat struct {
    Type       ResponseFormatType
    JSONSchema any
}

type GenerateParams struct {
    Model            *Model
    System           string
    Messages         []Message
    Tools            []Tool
    ToolChoice       any
    ResponseFormat   *ResponseFormat
    Temperature      *float64
    TopP             *float64
    MaxTokens        *int
    StopSequences    []string
    FrequencyPenalty *float64
    PresencePenalty  *float64
    Seed             *int
    ReasoningEffort  *string
}

type StepResult struct {
    Text            string
    Reasoning       string
    FinishReason    FinishReason
    RawFinishReason string
    Usage           Usage
    ToolCalls       []ToolCall
    ToolResults     []ToolResult
    Response        ResponseMetadata
    Messages        []Message
}

type GenerateResult struct {
    Text            string
    Reasoning       string
    FinishReason    FinishReason
    RawFinishReason string
    Usage           Usage
    Sources         []Source
    Files           []GeneratedFile
    ToolCalls       []ToolCall
    ToolResults     []ToolResult
    Response        ResponseMetadata
    Steps           []StepResult
    Messages        []Message
}
```

### Generate Options

```go
type GenerateOption func(*generateConfig)

func WithModel(model *Model) GenerateOption
func WithMessages(messages []Message) GenerateOption
func WithSystem(text string) GenerateOption
func WithTools(tools []Tool) GenerateOption
func WithToolChoice(choice any) GenerateOption
func WithResponseFormat(rf ResponseFormat) GenerateOption
func WithTemperature(t float64) GenerateOption
func WithTopP(topP float64) GenerateOption
func WithMaxTokens(n int) GenerateOption
func WithStopSequences(s []string) GenerateOption
func WithFrequencyPenalty(penalty float64) GenerateOption
func WithPresencePenalty(penalty float64) GenerateOption
func WithSeed(s int) GenerateOption
func WithReasoningEffort(effort string) GenerateOption

func WithMaxSteps(n int) GenerateOption
func WithOnFinish(fn func(*GenerateResult)) GenerateOption
func WithOnStep(fn func(*StepResult) *GenerateParams) GenerateOption
func WithPrepareStep(fn func(*GenerateParams) *GenerateParams) GenerateOption
func WithApprovalHandler(fn func(ctx context.Context, call ToolCall) (bool, error)) GenerateOption
```

Behavior notes:

- `WithMaxSteps(0)` is the default single-call mode.
- `WithMaxSteps(N)` enables automatic tool execution for up to `N` LLM calls.
- `WithMaxSteps(-1)` means unlimited loop until the model stops requesting tools.
- `WithToolChoice` accepts `"auto"`, `"none"`, or `"required"`.

### Tools

```go
type ToolExecuteFunc func(ctx *ToolExecContext, input any) (any, error)

type ToolExecContext struct {
    context.Context
    ToolCallID   string
    ToolName     string
    SendProgress func(content any)
}

type Tool struct {
    Name            string
    Description     string
    Parameters      any
    Execute         ToolExecuteFunc
    RequireApproval bool
    CacheControl    *CacheControl  // optional, Anthropic only
}

func NewTool[T any](
    name, description string,
    execute func(ctx *ToolExecContext, input T) (any, error),
) Tool

type ToolCall struct {
    ToolCallID string
    ToolName   string
    Input      any
}

type ToolResult struct {
    ToolCallID string
    ToolName   string
    Input      any
    Output     any
    IsError    bool
}
```

### MCP

```go
type MCPTransportType string

const (
    MCPTransportHTTP MCPTransportType = "http"
    MCPTransportSSE  MCPTransportType = "sse"
)

type MCPClientConfig struct {
    Type       MCPTransportType
    URL        string
    Headers    map[string]string
    Transport  mcp.Transport
    HTTPClient *http.Client
    Name       string
    Version    string
}

type MCPClient struct { /* unexported fields */ }

func CreateMCPClient(ctx context.Context, config *MCPClientConfig) (*MCPClient, error)
func (c *MCPClient) Tools(ctx context.Context) ([]Tool, error)
func (c *MCPClient) Close() error
```

Usage notes:

- `MCPTransportHTTP` is the default built-in transport and uses the official MCP Go SDK's streamable HTTP client transport.
- `MCPTransportSSE` uses the official MCP Go SDK's SSE client transport.
- For stdio or other custom transports, create the transport with `github.com/modelcontextprotocol/go-sdk/mcp` and pass it through `Transport`.
- `Tools(ctx)` converts remote MCP tools into ordinary `sdk.Tool` values suitable for `WithTools(...)`.
- MCP tool schemas are converted from MCP `InputSchema` into `*jsonschema.Schema`.
- MCP execution wrappers call `tools/call` and return concatenated text content to the model.

### Streaming

```go
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

type StreamPart interface {
    Type() StreamPartType
}

type TextStartPart struct {
    ID               string
    ProviderMetadata map[string]any
}

type TextDeltaPart struct {
    ID               string
    Text             string
    ProviderMetadata map[string]any
}

type TextEndPart struct {
    ID               string
    ProviderMetadata map[string]any
}

type ReasoningStartPart struct {
    ID               string
    ProviderMetadata map[string]any
}

type ReasoningDeltaPart struct {
    ID               string
    Text             string
    ProviderMetadata map[string]any
}

type ReasoningEndPart struct {
    ID               string
    ProviderMetadata map[string]any
}

type ToolInputStartPart struct {
    ID               string
    ToolName         string
    ProviderMetadata map[string]any
}

type ToolInputDeltaPart struct {
    ID               string
    Delta            string
    ProviderMetadata map[string]any
}

type ToolInputEndPart struct {
    ID               string
    ProviderMetadata map[string]any
}

type StreamToolCallPart struct {
    ToolCallID string
    ToolName   string
    Input      any
}

type StreamToolResultPart struct {
    ToolCallID string
    ToolName   string
    Input      any
    Output     any
}

type StreamToolErrorPart struct {
    ToolCallID string
    ToolName   string
    Error      error
}

type ToolOutputDeniedPart struct {
    ToolCallID string
    ToolName   string
}

type ToolApprovalRequestPart struct {
    ApprovalID string
    ToolCallID string
    ToolName   string
    Input      any
}

type ToolProgressPart struct {
    ToolCallID string
    ToolName   string
    Content    any
}

type StreamSourcePart struct {
    Source Source
}

type StreamFilePart struct {
    File GeneratedFile
}

type StartPart struct{}

type FinishPart struct {
    FinishReason    FinishReason
    RawFinishReason string
    TotalUsage      Usage
}

type StartStepPart struct{}

type FinishStepPart struct {
    FinishReason     FinishReason
    RawFinishReason  string
    Usage            Usage
    Response         ResponseMetadata
    ProviderMetadata map[string]any
}

type ErrorPart struct {
    Error error
}

type AbortPart struct {
    Reason string
}

type RawPart struct {
    RawValue any
}

type StreamResult struct {
    Stream   <-chan StreamPart
    Steps    []StepResult
    Messages []Message
}

func (sr *StreamResult) Text() (string, error)
func (sr *StreamResult) ToResult() (*GenerateResult, error)
```

### Usage, Sources, Files, Response Metadata

```go
type Usage struct {
    InputTokens        int
    OutputTokens       int
    TotalTokens        int
    ReasoningTokens    int
    CachedInputTokens  int
    InputTokenDetails  InputTokenDetail
    OutputTokenDetails OutputTokenDetail
}

type InputTokenDetail struct {
    NoCacheTokens      int
    CacheReadTokens    int
    CacheWriteTokens   int
    CacheWrite5mTokens int  // Anthropic: 5-minute cache writes
    CacheWrite1hTokens int  // Anthropic: 1-hour cache writes (ttl="1h")
}

type OutputTokenDetail struct {
    TextTokens      int
    ReasoningTokens int
    AudioTokens     int
}

type Source struct {
    SourceType       string
    ID               string
    URL              string
    Title            string
    ProviderMetadata map[string]any
}

type GeneratedFile struct {
    Data      string
    MediaType string
}

type ResponseMetadata struct {
    ID        string
    ModelID   string
    Timestamp time.Time
    Headers   map[string]string
}
```

### Embeddings

```go
type EmbeddingProvider interface {
    DoEmbed(ctx context.Context, params EmbedParams) (*EmbedResult, error)
}

type EmbeddingModel struct {
    ID                   string
    Provider             EmbeddingProvider
    MaxEmbeddingsPerCall int
}

type EmbedParams struct {
    Model      *EmbeddingModel
    Values     []string
    Dimensions *int
}

type EmbedResult struct {
    Embeddings [][]float64
    Usage      EmbeddingUsage
}

type EmbeddingUsage struct {
    Tokens int
}

type EmbedOption func(*embedConfig)

func WithEmbeddingModel(model *EmbeddingModel) EmbedOption
func WithDimensions(d int) EmbedOption
```

### Image Generation & Editing

```go
type ImageGenerationProvider interface {
    DoGenerate(ctx context.Context, params *ImageGenerationParams) (*ImageResult, error)
}

type ImageEditProvider interface {
    DoEdit(ctx context.Context, params *ImageEditParams) (*ImageResult, error)
}

type ImageGenerationModel struct {
    ID       string
    Provider ImageGenerationProvider
}

type ImageEditModel struct {
    ID       string
    Provider ImageEditProvider
}

type ImageGenerationParams struct {
    Model             *ImageGenerationModel
    Prompt            string
    N                 *int
    Size              string
    Quality           string
    Style             string
    ResponseFormat    string
    Background        string
    OutputFormat      string
    OutputCompression *int
    Moderation        string
    User              string
}

type ImageEditParams struct {
    Model             *ImageEditModel
    Images            []ImageInput
    Prompt            string
    Mask              *ImageInput
    N                 *int
    Size              string
    Quality           string
    Background        string
    OutputFormat      string
    OutputCompression *int
    InputFidelity     string
    Moderation        string
    ResponseFormat    string
    User              string
}

type ImageInput struct {
    Data      []byte
    MediaType string
    Filename  string
    URL       string
    FileID    string
}

type ImageResult struct {
    Created int64
    Data    []ImageData
    Usage   ImageUsage
}

type ImageData struct {
    B64JSON       string
    URL           string
    RevisedPrompt string
}

type ImageUsage struct {
    TotalTokens       int
    InputTokens       int
    OutputTokens      int
    InputTokenDetails *ImageInputTokenDetails
}

type ImageInputTokenDetails struct {
    TextTokens  int
    ImageTokens int
}

type ImageGenerateOption func(*imageGenerateConfig)

func WithImageGenerationModel(model *ImageGenerationModel) ImageGenerateOption
func WithImagePrompt(prompt string) ImageGenerateOption
func WithImageN(n int) ImageGenerateOption
func WithImageSize(size string) ImageGenerateOption
func WithImageQuality(quality string) ImageGenerateOption
func WithImageStyle(style string) ImageGenerateOption
func WithImageResponseFormat(format string) ImageGenerateOption
func WithImageBackground(background string) ImageGenerateOption
func WithImageOutputFormat(format string) ImageGenerateOption
func WithImageOutputCompression(compression int) ImageGenerateOption
func WithImageModeration(moderation string) ImageGenerateOption
func WithImageUser(user string) ImageGenerateOption

type ImageEditOption func(*imageEditConfig)

func WithImageEditModel(model *ImageEditModel) ImageEditOption
func WithEditImages(images ...ImageInput) ImageEditOption
func WithEditPrompt(prompt string) ImageEditOption
func WithEditMask(mask *ImageInput) ImageEditOption
func WithEditN(n int) ImageEditOption
func WithEditSize(size string) ImageEditOption
func WithEditQuality(quality string) ImageEditOption
func WithEditBackground(background string) ImageEditOption
func WithEditOutputFormat(format string) ImageEditOption
func WithEditOutputCompression(compression int) ImageEditOption
func WithEditInputFidelity(fidelity string) ImageEditOption
func WithEditModeration(moderation string) ImageEditOption
func WithEditResponseFormat(format string) ImageEditOption
func WithEditUser(user string) ImageEditOption
```

## Package `provider/openai/completions`

Implements the OpenAI Chat Completions API and OpenAI-compatible `/chat/completions` backends.

```go
type Provider struct { /* unexported fields */ }

type Option func(*Provider)

func WithAPIKey(apiKey string) Option
func WithBaseURL(baseURL string) Option
func WithHTTPClient(client *http.Client) Option
func WithDeepSeekChatCompletionsCompat() Option
func New(options ...Option) *Provider

func (p *Provider) Name() string
func (p *Provider) ListModels(ctx context.Context) ([]sdk.Model, error)
func (p *Provider) Test(ctx context.Context) *sdk.ProviderTestResult
func (p *Provider) TestModel(ctx context.Context, modelID string) (*sdk.ModelTestResult, error)
func (p *Provider) ChatModel(id string) *sdk.Model
func (p *Provider) DoGenerate(ctx context.Context, params sdk.GenerateParams) (*sdk.GenerateResult, error)
func (p *Provider) DoStream(ctx context.Context, params sdk.GenerateParams) (*sdk.StreamResult, error)
```

Default option values:

- `WithBaseURL`: `https://api.openai.com/v1`
- `WithHTTPClient`: `&http.Client{}`
- `WithDeepSeekChatCompletionsCompat`: disabled. When enabled, `WithReasoningEffort("none")` sends `thinking:{type:"disabled"}` and omits `reasoning_effort`.

Discovery endpoints:

- `ListModels`: `GET /models`
- `Test`: `GET /models?limit=1`
- `TestModel`: `GET /models/{id}`

## Package `provider/openai/responses`

Implements the OpenAI Responses API with reasoning summaries, annotations, and flat input mapping.

```go
type Provider struct { /* unexported fields */ }

type Option func(*Provider)

func WithAPIKey(apiKey string) Option
func WithBaseURL(baseURL string) Option
func WithHTTPClient(client *http.Client) Option
func New(options ...Option) *Provider

func (p *Provider) Name() string
func (p *Provider) ListModels(ctx context.Context) ([]sdk.Model, error)
func (p *Provider) Test(ctx context.Context) *sdk.ProviderTestResult
func (p *Provider) TestModel(ctx context.Context, modelID string) (*sdk.ModelTestResult, error)
func (p *Provider) ChatModel(id string) *sdk.Model
func (p *Provider) DoGenerate(ctx context.Context, params sdk.GenerateParams) (*sdk.GenerateResult, error)
func (p *Provider) DoStream(ctx context.Context, params sdk.GenerateParams) (*sdk.StreamResult, error)
```

Default option values:

- `WithBaseURL`: `https://api.openai.com/v1`
- `WithHTTPClient`: `&http.Client{}`

Discovery endpoints:

- `ListModels`: `GET /models`
- `Test`: `GET /models?limit=1`
- `TestModel`: `GET /models/{id}`

Responses-specific behavior:

- assistant reasoning maps to `GenerateResult.Reasoning`
- URL citation annotations map to `GenerateResult.Sources`
- function-call outputs map to tool-call and tool-result structures

## Package `provider/openai/codex`

Implements the OpenAI Codex backend API for coding agent models. Communicates with the ChatGPT backend at `/codex/responses` using SSE streaming with Responses-style events.

```go
type ModelDescriptor struct {
    ID                string
    DisplayName       string
    SupportsToolCall  bool
    SupportsReasoning bool
    ReasoningEfforts  []string
}

func Catalog() []ModelDescriptor

type Provider struct { /* unexported fields */ }

type Option func(*Provider)

func WithAccessToken(token string) Option
func WithAPIKey(token string) Option        // alias for WithAccessToken
func WithAccountID(accountID string) Option
func WithOriginator(originator string) Option
func WithBaseURL(baseURL string) Option
func WithHTTPClient(client *http.Client) Option
func New(options ...Option) *Provider

func (p *Provider) Name() string
func (p *Provider) ListModels(ctx context.Context) ([]sdk.Model, error)
func (p *Provider) Test(ctx context.Context) *sdk.ProviderTestResult
func (p *Provider) TestModel(ctx context.Context, modelID string) (*sdk.ModelTestResult, error)
func (p *Provider) ChatModel(id string) *sdk.Model
func (p *Provider) DoGenerate(ctx context.Context, params sdk.GenerateParams) (*sdk.GenerateResult, error)
func (p *Provider) DoStream(ctx context.Context, params sdk.GenerateParams) (*sdk.StreamResult, error)
```

Default option values:

- `WithBaseURL`: `https://chatgpt.com/backend-api`
- `WithOriginator`: `"codex_cli_rs"`
- `WithHTTPClient`: `&http.Client{}`
- `WithAccountID`: auto-extracted from the access token JWT if omitted

Authentication headers:

- `Authorization: Bearer <access_token>`
- `OpenAI-Beta: responses=experimental`
- `originator: <originator>`
- `chatgpt-account-id: <account_id>` (when available)

Codex-specific behavior:

- `ListModels` returns a static catalog (no HTTP call)
- `TestModel` probes `POST /codex/responses` with a minimal request
- Messages are converted to the flat Codex input format (instructions + input items)
- Reasoning uses encrypted content: `ProviderMetadata["openai"]["reasoningEncryptedContent"]`
- Supports `ReasoningEffort` via the `reasoning.effort` request field

## Package `provider/anthropic/messages`

Implements the Anthropic Messages API.

```go
type ThinkingConfig struct {
    Type         string
    BudgetTokens int
}

type Provider struct { /* unexported fields */ }

type Option func(*Provider)

func WithAPIKey(apiKey string) Option
func WithAuthToken(token string) Option
func WithBaseURL(baseURL string) Option
func WithHTTPClient(client *http.Client) Option
func WithHeaders(headers map[string]string) Option
func WithThinking(cfg ThinkingConfig) Option
func New(options ...Option) *Provider

func (p *Provider) Name() string
func (p *Provider) ListModels(ctx context.Context) ([]sdk.Model, error)
func (p *Provider) Test(ctx context.Context) *sdk.ProviderTestResult
func (p *Provider) TestModel(ctx context.Context, modelID string) (*sdk.ModelTestResult, error)
func (p *Provider) ChatModel(id string) *sdk.Model
func (p *Provider) DoGenerate(ctx context.Context, params sdk.GenerateParams) (*sdk.GenerateResult, error)
func (p *Provider) DoStream(ctx context.Context, params sdk.GenerateParams) (*sdk.StreamResult, error)
```

Default option values:

- `WithBaseURL`: `https://api.anthropic.com/v1`
- default API version header: `2023-06-01`
- `WithHTTPClient`: `&http.Client{}`

Thinking config notes:

- `Type` supports `"enabled"`, `"adaptive"`, or `"disabled"`
- `BudgetTokens` is required when `Type == "enabled"`

Discovery endpoints:

- `ListModels`: `GET /v1/models`
- `Test`: `GET /v1/models?limit=1`
- `TestModel`: `GET /v1/models/{id}`

## Package `provider/google/generativeai`

Implements the Google Generative AI API for Gemini chat models.

```go
type Provider struct { /* unexported fields */ }

type Option func(*Provider)

func WithAPIKey(apiKey string) Option
func WithBaseURL(baseURL string) Option
func WithHTTPClient(client *http.Client) Option
func New(options ...Option) *Provider

func (p *Provider) Name() string
func (p *Provider) ListModels(ctx context.Context) ([]sdk.Model, error)
func (p *Provider) Test(ctx context.Context) *sdk.ProviderTestResult
func (p *Provider) TestModel(ctx context.Context, modelID string) (*sdk.ModelTestResult, error)
func (p *Provider) ChatModel(id string) *sdk.Model
func (p *Provider) DoGenerate(ctx context.Context, params sdk.GenerateParams) (*sdk.GenerateResult, error)
func (p *Provider) DoStream(ctx context.Context, params sdk.GenerateParams) (*sdk.StreamResult, error)
```

Default option values:

- `WithBaseURL`: `https://generativelanguage.googleapis.com/v1beta`
- `WithHTTPClient`: `&http.Client{}`

Model ID rules:

- plain names like `gemini-2.5-flash` are accepted
- full paths like `publishers/google/models/gemini-2.5-flash` are also accepted

Discovery endpoints:

- `ListModels`: `GET /v1beta/models`
- `Test`: `GET /v1beta/models?pageSize=1`
- `TestModel`: `GET /v1beta/models/{id}`

## Package `provider/openai/embedding`

Implements the OpenAI Embeddings API.

```go
type Provider struct { /* unexported fields */ }

type Option func(*Provider)

func WithAPIKey(apiKey string) Option
func WithBaseURL(baseURL string) Option
func WithHTTPClient(client *http.Client) Option
func New(options ...Option) *Provider

func (p *Provider) EmbeddingModel(id string) *sdk.EmbeddingModel
func (p *Provider) DoEmbed(ctx context.Context, params sdk.EmbedParams) (*sdk.EmbedResult, error)
```

Default option values:

- `WithBaseURL`: `https://api.openai.com/v1`
- `WithHTTPClient`: `&http.Client{}`

Behavior notes:

- `EmbeddingModel(id)` returns a model with `MaxEmbeddingsPerCall: 2048`
- `DoEmbed` calls `POST /embeddings` with `encoding_format: "float"`

## Package `provider/google/embedding`

Implements the Google embedding API.

```go
type Provider struct { /* unexported fields */ }

type Option func(*Provider)

func WithAPIKey(apiKey string) Option
func WithBaseURL(baseURL string) Option
func WithHTTPClient(client *http.Client) Option
func WithTaskType(taskType string) Option
func New(options ...Option) *Provider

func (p *Provider) EmbeddingModel(id string) *sdk.EmbeddingModel
func (p *Provider) DoEmbed(ctx context.Context, params sdk.EmbedParams) (*sdk.EmbedResult, error)
```

Default option values:

- `WithBaseURL`: `https://generativelanguage.googleapis.com/v1beta`
- `WithHTTPClient`: `&http.Client{}`

Task type values:

- `RETRIEVAL_QUERY`
- `RETRIEVAL_DOCUMENT`
- `SEMANTIC_SIMILARITY`
- `CLASSIFICATION`
- `CLUSTERING`
- `QUESTION_ANSWERING`
- `FACT_VERIFICATION`
- `CODE_RETRIEVAL_QUERY`

Behavior notes:

- `EmbeddingModel(id)` returns a model with `MaxEmbeddingsPerCall: 2048`
- single-value embedding uses `embedContent`
- multi-value embedding uses `batchEmbedContents`

## Package `provider/openai/images`

Implements the OpenAI Images API for image generation (`/images/generations`) and editing (`/images/edits`).

```go
type Provider struct { /* unexported fields */ }

type Option func(*Provider)

func WithAPIKey(apiKey string) Option
func WithBaseURL(baseURL string) Option
func WithHTTPClient(client *http.Client) Option
func New(options ...Option) *Provider

func (p *Provider) GenerationModel(id string) *sdk.ImageGenerationModel
func (p *Provider) EditModel(id string) *sdk.ImageEditModel
func (p *Provider) DoGenerate(ctx context.Context, params *sdk.ImageGenerationParams) (*sdk.ImageResult, error)
func (p *Provider) DoEdit(ctx context.Context, params *sdk.ImageEditParams) (*sdk.ImageResult, error)
```

Default option values:

- `WithBaseURL`: `https://api.openai.com/v1`
- `WithHTTPClient`: `&http.Client{}`

Supported models:

- Generation: `dall-e-2`, `dall-e-3`, `gpt-image-1`, `gpt-image-1-mini`, `gpt-image-1.5`
- Editing: `gpt-image-1`, `gpt-image-1-mini`, `gpt-image-1.5`, `dall-e-2`

Edit behavior:

- When `ImageInput.Data` (raw bytes) is provided, the request is sent as `multipart/form-data`
- When `ImageInput.URL` or `ImageInput.FileID` is provided, the request is sent as JSON

## Selection Cheatsheet

- Broad OpenAI-compatible chat API: `provider/openai/completions`
- OpenAI Responses features such as reasoning summaries or citation annotations: `provider/openai/responses`
- OpenAI Codex coding agents with encrypted reasoning: `provider/openai/codex`
- Claude and extended thinking: `provider/anthropic/messages`
- Gemini chat and tool calling: `provider/google/generativeai`
- OpenAI image generation and editing (dall-e, gpt-image): `provider/openai/images`
- OpenAI-compatible embeddings: `provider/openai/embedding`
- Gemini embeddings with task-type tuning: `provider/google/embedding`
