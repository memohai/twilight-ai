# API Reference

Complete reference for all exported types and functions in the Twilight AI SDK.

## Package `sdk`

### Client

```go
type Client struct{}

func NewClient() *Client
```

A `Client` provides text generation methods. The provider is resolved from the `Model` passed via `WithModel`.

#### Methods

```go
func (c *Client) GenerateText(ctx context.Context, options ...GenerateOption) (string, error)
```

Generates text and returns only the response string.

```go
func (c *Client) GenerateTextResult(ctx context.Context, options ...GenerateOption) (*GenerateResult, error)
```

Generates text and returns the full result including usage, steps, and metadata.

```go
func (c *Client) StreamText(ctx context.Context, options ...GenerateOption) (*StreamResult, error)
```

Returns a streaming result with a channel of `StreamPart` chunks.

#### Package-Level Functions

These use a default client instance:

```go
func GenerateText(ctx context.Context, options ...GenerateOption) (string, error)
func GenerateTextResult(ctx context.Context, options ...GenerateOption) (*GenerateResult, error)
func StreamText(ctx context.Context, options ...GenerateOption) (*StreamResult, error)
```

---

### Provider

```go
type Provider interface {
    Name() string
    ListModels(ctx context.Context) ([]Model, error)
    Test(ctx context.Context) *ProviderTestResult
    TestModel(ctx context.Context, modelID string) (*ModelTestResult, error)
    DoGenerate(ctx context.Context, params GenerateParams) (*GenerateResult, error)
    DoStream(ctx context.Context, params GenerateParams) (*StreamResult, error)
}
```

| Method | Purpose |
|--------|---------|
| `Name()` | Returns a provider identifier (e.g. `"openai-completions"`) |
| `ListModels(ctx)` | Fetches available models from the backend API |
| `Test(ctx)` | Health check: returns OK, Unhealthy, or Unreachable |
| `TestModel(ctx, id)` | Checks if a specific model ID is supported |
| `DoGenerate(ctx, params)` | Performs a single non-streaming LLM call |
| `DoStream(ctx, params)` | Performs a streaming LLM call |

#### ProviderStatus

```go
type ProviderStatus string

const (
    ProviderStatusOK          ProviderStatus = "ok"          // Connected and healthy
    ProviderStatusUnhealthy   ProviderStatus = "unhealthy"   // Connected but health check failed
    ProviderStatusUnreachable ProviderStatus = "unreachable" // Cannot connect
)
```

#### ProviderTestResult

```go
type ProviderTestResult struct {
    Status  ProviderStatus
    Message string
    Error   error
}
```

#### ModelTestResult

```go
type ModelTestResult struct {
    Supported bool
    Message   string
}
```

### Model

```go
type Model struct {
    ID          string
    DisplayName string
    Provider    Provider
    Type        ModelType
    MaxTokens   int
}

type ModelType string
const ModelTypeChat ModelType = "chat"
```

#### Methods

```go
func (m *Model) Test(ctx context.Context) (*ModelTestResult, error)
```

Checks whether this model is supported by its provider. Delegates to `Provider.TestModel`.

---

### Messages

```go
type Message struct {
    Role    MessageRole
    Content []MessagePart
}
```

#### MessageRole

| Constant | Value |
|----------|-------|
| `MessageRoleUser` | `"user"` |
| `MessageRoleAssistant` | `"assistant"` |
| `MessageRoleSystem` | `"system"` |
| `MessageRoleTool` | `"tool"` |

#### Message Constructors

```go
func UserMessage(text string, extra ...MessagePart) Message
func SystemMessage(text string) Message
func AssistantMessage(text string) Message
func ToolMessage(results ...ToolResultPart) Message
```

`UserMessage` accepts optional extra parts (e.g. `ImagePart`) after the text.

#### MessagePart Interface

```go
type MessagePart interface {
    PartType() MessagePartType
}
```

#### Part Types

```go
type TextPart struct {
    Text string
}

type ReasoningPart struct {
    Text      string
    Signature string  // optional
}

type ImagePart struct {
    Image     string  // URL or base64
    MediaType string  // optional, e.g. "image/png"
}

type FilePart struct {
    Data      string
    MediaType string  // optional
    Filename  string  // optional
}

type ToolCallPart struct {
    ToolCallID string
    ToolName   string
    Input      any
}

type ToolResultPart struct {
    ToolCallID string
    ToolName   string
    Result     any
    IsError    bool   // optional
}
```

`Message` supports full JSON serialization with automatic type discrimination.

---

### Generation

#### GenerateParams

```go
type GenerateParams struct {
    Model            *Model
    System           string
    Messages         []Message
    Tools            []Tool
    ToolChoice       any              // "auto", "none", "required"
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
```

#### GenerateResult

```go
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

#### StepResult

```go
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
```

#### FinishReason

| Constant | Value | Description |
|----------|-------|-------------|
| `FinishReasonStop` | `"stop"` | Normal completion |
| `FinishReasonLength` | `"length"` | Max tokens reached |
| `FinishReasonContentFilter` | `"content-filter"` | Content filter triggered |
| `FinishReasonToolCalls` | `"tool-calls"` | Model wants to call tools |
| `FinishReasonError` | `"error"` | An error occurred |
| `FinishReasonOther` | `"other"` | Provider-specific reason |
| `FinishReasonUnknown` | `"unknown"` | Unknown reason |

#### ResponseFormat

```go
type ResponseFormat struct {
    Type       ResponseFormatType
    JSONSchema any                 // required when Type is json_schema
}

type ResponseFormatType string
const (
    ResponseFormatText       ResponseFormatType = "text"
    ResponseFormatJSONObject ResponseFormatType = "json_object"
    ResponseFormatJSONSchema ResponseFormatType = "json_schema"
)
```

---

### Options

All options are of type `GenerateOption` (`func(*generateConfig)`).

#### Provider-Level Options

| Function | Description |
|----------|-------------|
| `WithModel(model *Model)` | **Required.** The model to use |
| `WithMessages(msgs []Message)` | Chat messages |
| `WithSystem(text string)` | System prompt |
| `WithTools(tools []Tool)` | Tool definitions |
| `WithToolChoice(choice any)` | `"auto"`, `"none"`, `"required"` |
| `WithResponseFormat(rf ResponseFormat)` | Response format constraint |
| `WithTemperature(t float64)` | Sampling temperature |
| `WithTopP(topP float64)` | Nucleus sampling |
| `WithMaxTokens(n int)` | Maximum output tokens |
| `WithStopSequences(s []string)` | Stop sequences |
| `WithFrequencyPenalty(p float64)` | Frequency penalty |
| `WithPresencePenalty(p float64)` | Presence penalty |
| `WithSeed(s int)` | Random seed for reproducibility |
| `WithReasoningEffort(effort string)` | Reasoning effort level |

#### Orchestration Options

| Function | Description |
|----------|-------------|
| `WithMaxSteps(n int)` | `0` = single call (default), `N` = up to N calls, `-1` = unlimited |
| `WithOnFinish(fn func(*GenerateResult))` | Called when all steps complete |
| `WithOnStep(fn func(*StepResult) *GenerateParams)` | Called after each step; return non-nil to override next step |
| `WithPrepareStep(fn func(*GenerateParams) *GenerateParams)` | Called before each step (from step 2); can modify params |
| `WithApprovalHandler(fn func(ctx, ToolCall) (bool, error))` | Approval for tools with `RequireApproval` |

---

### Tools

```go
type Tool struct {
    Name            string
    Description     string
    Parameters      any              // JSON Schema
    Execute         ToolExecuteFunc
    RequireApproval bool
}

type ToolExecuteFunc func(ctx *ToolExecContext, input any) (any, error)

type ToolExecContext struct {
    context.Context
    ToolCallID   string
    ToolName     string
    SendProgress func(content any) // nil outside streaming mode
}
```

#### ToolCall & ToolResult

```go
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

Behavior notes:

- `CreateMCPClient` performs the MCP handshake and returns a ready-to-use client.
- When `Transport` is non-nil, `Type`, `URL`, and `Headers` are ignored.
- `MCPTransportHTTP` uses the official MCP Go SDK's streamable HTTP client transport.
- `MCPTransportSSE` uses the official MCP Go SDK's SSE client transport.
- For stdio, callers should create an MCP transport themselves, such as `mcp.CommandTransport`, and pass it via `Transport`.
- `Tools(ctx)` converts remote `mcp.Tool` definitions into `sdk.Tool` values.
- Converted tools use the MCP server's `InputSchema` as `Parameters` and call `tools/call` inside `Execute`.

---

### Streaming

#### StreamResult

```go
type StreamResult struct {
    Stream   <-chan StreamPart
    Steps    []StepResult  // populated after stream consumed
    Messages []Message     // populated after stream consumed
}

func (sr *StreamResult) Text() (string, error)
func (sr *StreamResult) ToResult() (*GenerateResult, error)
```

#### StreamPart Interface

```go
type StreamPart interface {
    Type() StreamPartType
}
```

#### All StreamPart Types

**Text:**

| Type | Key Fields |
|------|-----------|
| `*TextStartPart` | `ID` |
| `*TextDeltaPart` | `ID`, `Text` |
| `*TextEndPart` | `ID` |

**Reasoning:**

| Type | Key Fields |
|------|-----------|
| `*ReasoningStartPart` | `ID` |
| `*ReasoningDeltaPart` | `ID`, `Text` |
| `*ReasoningEndPart` | `ID` |

**Tool Input:**

| Type | Key Fields |
|------|-----------|
| `*ToolInputStartPart` | `ID`, `ToolName` |
| `*ToolInputDeltaPart` | `ID`, `Delta` |
| `*ToolInputEndPart` | `ID` |

**Tool Execution:**

| Type | Key Fields |
|------|-----------|
| `*StreamToolCallPart` | `ToolCallID`, `ToolName`, `Input` |
| `*StreamToolResultPart` | `ToolCallID`, `ToolName`, `Input`, `Output` |
| `*StreamToolErrorPart` | `ToolCallID`, `ToolName`, `Error` |
| `*ToolOutputDeniedPart` | `ToolCallID`, `ToolName` |
| `*ToolApprovalRequestPart` | `ApprovalID`, `ToolCallID`, `ToolName`, `Input` |
| `*ToolProgressPart` | `ToolCallID`, `ToolName`, `Content` |

**Sources & Files:**

| Type | Key Fields |
|------|-----------|
| `*StreamSourcePart` | `Source` |
| `*StreamFilePart` | `File` |

**Lifecycle:**

| Type | Key Fields |
|------|-----------|
| `*StartPart` | — |
| `*FinishPart` | `FinishReason`, `RawFinishReason`, `TotalUsage` |
| `*StartStepPart` | — |
| `*FinishStepPart` | `FinishReason`, `RawFinishReason`, `Usage`, `Response` |
| `*ErrorPart` | `Error` |
| `*AbortPart` | `Reason` |
| `*RawPart` | `RawValue` |

---

### Usage

```go
type Usage struct {
    InputTokens         int
    OutputTokens        int
    TotalTokens         int
    ReasoningTokens     int
    CachedInputTokens   int
    InputTokenDetails   InputTokenDetail
    OutputTokenDetails  OutputTokenDetail
}

type InputTokenDetail struct {
    CacheReadTokens    int
    CacheCreationTokens int
}

type OutputTokenDetail struct {
    TextTokens      int
    ReasoningTokens int
    AudioTokens     int
}
```

### Source

```go
type Source struct {
    SourceType       string
    ID               string
    URL              string
    Title            string
    ProviderMetadata map[string]any
}
```

### GeneratedFile

```go
type GeneratedFile struct {
    Data      string
    MediaType string
}
```

### ResponseMetadata

```go
type ResponseMetadata struct {
    ID        string
    ModelID   string
    Timestamp time.Time
    Headers   map[string]string
}
```

---

### Embedding

#### EmbeddingProvider

```go
type EmbeddingProvider interface {
    DoEmbed(ctx context.Context, params EmbedParams) (*EmbedResult, error)
}
```

The interface that embedding backends must implement.

#### EmbeddingModel

```go
type EmbeddingModel struct {
    ID                   string
    Provider             EmbeddingProvider
    MaxEmbeddingsPerCall int
}
```

Represents an embedding model bound to an `EmbeddingProvider`. `MaxEmbeddingsPerCall` indicates the maximum number of input values per single API call (typically 2048).

#### EmbedParams

```go
type EmbedParams struct {
    Model      *EmbeddingModel
    Values     []string
    Dimensions *int
}
```

| Field | Description |
|-------|-------------|
| `Model` | **Required.** The embedding model to use |
| `Values` | Input texts to embed |
| `Dimensions` | Optional output dimensionality (not all models support this) |

#### EmbedResult

```go
type EmbedResult struct {
    Embeddings [][]float64
    Usage      EmbeddingUsage
}
```

| Field | Description |
|-------|-------------|
| `Embeddings` | One `[]float64` vector per input value |
| `Usage` | Token usage for the request |

#### EmbeddingUsage

```go
type EmbeddingUsage struct {
    Tokens int
}
```

#### Embed Options

All options are of type `EmbedOption` (`func(*embedConfig)`).

| Function | Description |
|----------|-------------|
| `WithEmbeddingModel(model *EmbeddingModel)` | **Required.** The embedding model to use |
| `WithDimensions(d int)` | Output dimensionality (model-dependent) |

#### Client Methods

```go
func (c *Client) Embed(ctx context.Context, value string, options ...EmbedOption) ([]float64, error)
func (c *Client) EmbedMany(ctx context.Context, values []string, options ...EmbedOption) (*EmbedResult, error)
```

| Method | Description |
|--------|-------------|
| `Embed` | Generates an embedding for a single string; returns the vector |
| `EmbedMany` | Generates embeddings for multiple strings; returns the full result |

#### Package-Level Functions

```go
func Embed(ctx context.Context, value string, options ...EmbedOption) ([]float64, error)
func EmbedMany(ctx context.Context, values []string, options ...EmbedOption) (*EmbedResult, error)
```

These use the default client instance, equivalent to `client.Embed` and `client.EmbedMany`.

---

### Speech

#### SpeechProvider

```go
type SpeechProvider interface {
    DoSynthesize(ctx context.Context, params SpeechParams) (*SpeechResult, error)
    DoStream(ctx context.Context, params SpeechParams) (*SpeechStreamResult, error)
}
```

The interface that speech synthesis backends must implement.

#### SpeechModel

```go
type SpeechModel struct {
    ID       string
    Provider SpeechProvider
}
```

Represents a speech model bound to a `SpeechProvider`.

#### SpeechParams

```go
type SpeechParams struct {
    Model  *SpeechModel
    Text   string
    Config map[string]any
}
```

| Field | Description |
|-------|-------------|
| `Model` | **Required.** The speech model to use |
| `Text` | **Required.** The text to synthesize |
| `Config` | Provider-specific configuration (e.g. voice, format, speed) |

#### SpeechResult

```go
type SpeechResult struct {
    Audio       []byte
    ContentType string
}
```

| Field | Description |
|-------|-------------|
| `Audio` | Raw audio bytes |
| `ContentType` | MIME type (e.g. `audio/mpeg`) |

#### SpeechStreamResult

```go
type SpeechStreamResult struct {
    Stream      <-chan []byte
    ContentType string
}

func (r *SpeechStreamResult) Bytes() ([]byte, error)
```

| Field/Method | Description |
|-------|-------------|
| `Stream` | Channel that yields raw audio chunks; closed when done |
| `ContentType` | MIME type (e.g. `audio/mpeg`) |
| `Bytes()` | Consumes the stream and returns concatenated audio data |

#### Speech Options

All options are of type `SpeechOption` (`func(*speechConfig)`).

| Function | Description |
|----------|-------------|
| `WithSpeechModel(model *SpeechModel)` | **Required.** The speech model to use |
| `WithText(text string)` | **Required.** The text to synthesize |
| `WithSpeechConfig(cfg map[string]any)` | Provider-specific configuration |

#### Client Methods

```go
func (c *Client) GenerateSpeech(ctx context.Context, options ...SpeechOption) (*SpeechResult, error)
func (c *Client) StreamSpeech(ctx context.Context, options ...SpeechOption) (*SpeechStreamResult, error)
```

| Method | Description |
|--------|-------------|
| `GenerateSpeech` | Synthesizes speech; returns complete audio |
| `StreamSpeech` | Synthesizes speech; returns streaming audio chunks |

#### Package-Level Functions

```go
func GenerateSpeech(ctx context.Context, options ...SpeechOption) (*SpeechResult, error)
func StreamSpeech(ctx context.Context, options ...SpeechOption) (*SpeechStreamResult, error)
```

These use the default client instance.

---

## Package `provider/edge/speech`

### Provider

```go
type Provider struct { /* unexported */ }

func New(options ...Option) *Provider
```

Implements `sdk.SpeechProvider`. Uses Microsoft Edge's built-in TTS via WebSocket. No API key required.

#### Options

```go
type Option func(*Provider)

func WithBaseURL(url string) Option
```

| Option | Default | Description |
|--------|---------|-------------|
| `WithBaseURL(url)` | Bing WSS endpoint | Override the WebSocket endpoint (for testing) |

#### Methods

```go
func (p *Provider) SpeechModel(id string) *sdk.SpeechModel
func (p *Provider) DoSynthesize(ctx context.Context, params sdk.SpeechParams) (*sdk.SpeechResult, error)
func (p *Provider) DoStream(ctx context.Context, params sdk.SpeechParams) (*sdk.SpeechStreamResult, error)
```

| Method | Description |
|--------|-------------|
| `SpeechModel(id)` | Creates a `SpeechModel` bound to this provider. Default model: `edge-read-aloud` |
| `DoSynthesize` | Synthesizes complete audio via WebSocket |
| `DoStream` | Synthesizes streaming audio chunks via WebSocket |

#### Configuration Keys

The Edge provider reads these keys from `SpeechParams.Config`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `voice` | `string` | `en-US-EmmaMultilingualNeural` | Voice ID |
| `language` | `string` | Auto-detected | BCP-47 language tag |
| `format` | `string` | `audio-24khz-48kbitrate-mono-mp3` | Output format |
| `speed` | `float64` | `0` | Speech rate (1.0 = normal) |
| `pitch` | `float64` | `0` | Pitch in Hz |

#### Package-Level Variables

```go
var EdgeTTSVoices map[string][]string  // language tag → voice IDs
```

#### Helper Functions

```go
func LookupVoiceLang(voiceID string) (string, bool)
```

Returns the language tag for a voice ID, or `("", false)` if unknown.

---

## Package `provider/openai/embedding`

### Provider

```go
type Provider struct { /* unexported */ }

func New(options ...Option) *Provider
```

Implements `sdk.EmbeddingProvider`. Uses the OpenAI Embeddings API (`/embeddings`).

#### Options

```go
type Option func(*Provider)

func WithAPIKey(apiKey string) Option
func WithBaseURL(baseURL string) Option
func WithHTTPClient(client *http.Client) Option
```

| Option | Default | Description |
|--------|---------|-------------|
| `WithAPIKey(key)` | `""` | API key sent as `Authorization: Bearer <key>` |
| `WithBaseURL(url)` | `https://api.openai.com/v1` | Base URL for API requests |
| `WithHTTPClient(client)` | `&http.Client{}` | Custom HTTP client |

#### Methods

```go
func (p *Provider) EmbeddingModel(id string) *sdk.EmbeddingModel
func (p *Provider) DoEmbed(ctx context.Context, params sdk.EmbedParams) (*sdk.EmbedResult, error)
```

| Method | Description |
|--------|-------------|
| `EmbeddingModel(id)` | Creates an `EmbeddingModel` bound to this provider (MaxEmbeddingsPerCall: 2048) |
| `DoEmbed(ctx, params)` | Sends a `POST /embeddings` request with `encoding_format: "float"` |

#### Supported Models

Any model available via the OpenAI `/embeddings` endpoint, including:
- `text-embedding-3-small`
- `text-embedding-3-large`
- `text-embedding-ada-002`

---

## Package `provider/google/embedding`

### Provider

```go
type Provider struct { /* unexported */ }

func New(options ...Option) *Provider
```

Implements `sdk.EmbeddingProvider`. Uses the Google Generative AI Embedding API.

#### Options

```go
type Option func(*Provider)

func WithAPIKey(apiKey string) Option
func WithBaseURL(baseURL string) Option
func WithHTTPClient(client *http.Client) Option
func WithTaskType(taskType string) Option
```

| Option | Default | Description |
|--------|---------|-------------|
| `WithAPIKey(key)` | `""` | API key sent as `x-goog-api-key` header |
| `WithBaseURL(url)` | `https://generativelanguage.googleapis.com/v1beta` | Base URL |
| `WithHTTPClient(client)` | `&http.Client{}` | Custom HTTP client |
| `WithTaskType(taskType)` | `""` | Default task type for all requests |

#### Task Types

| Value | Use Case |
|-------|----------|
| `RETRIEVAL_QUERY` | Query text for search/retrieval |
| `RETRIEVAL_DOCUMENT` | Document text being indexed |
| `SEMANTIC_SIMILARITY` | Comparing text similarity |
| `CLASSIFICATION` | Text classification |
| `CLUSTERING` | Text clustering |
| `QUESTION_ANSWERING` | Question answering |
| `FACT_VERIFICATION` | Fact verification |
| `CODE_RETRIEVAL_QUERY` | Code search queries |

#### Methods

```go
func (p *Provider) EmbeddingModel(id string) *sdk.EmbeddingModel
func (p *Provider) DoEmbed(ctx context.Context, params sdk.EmbedParams) (*sdk.EmbedResult, error)
```

| Method | Description |
|--------|-------------|
| `EmbeddingModel(id)` | Creates an `EmbeddingModel` bound to this provider (MaxEmbeddingsPerCall: 2048) |
| `DoEmbed(ctx, params)` | Single value: `embedContent`; multiple values: `batchEmbedContents` |

#### Supported Models

- `gemini-embedding-001`
- `text-embedding-004`

---

## Package `provider/openai/completions`

### Provider

```go
type Provider struct { /* unexported */ }

func New(options ...Option) *Provider
```

Implements `sdk.Provider`. Uses the OpenAI Chat Completions API (`/chat/completions`).

#### Options

```go
type Option func(*Provider)

func WithAPIKey(apiKey string) Option
func WithBaseURL(baseURL string) Option
func WithHTTPClient(client *http.Client) Option
```

#### Methods

```go
func (p *Provider) Name() string                  // "openai-completions"
func (p *Provider) ChatModel(id string) *sdk.Model
func (p *Provider) ListModels(ctx context.Context) ([]sdk.Model, error)
func (p *Provider) Test(ctx context.Context) *sdk.ProviderTestResult
func (p *Provider) TestModel(ctx context.Context, modelID string) (*sdk.ModelTestResult, error)
func (p *Provider) DoGenerate(ctx, params) (*sdk.GenerateResult, error)
func (p *Provider) DoStream(ctx, params) (*sdk.StreamResult, error)
```

| Method | API Endpoint |
|--------|-------------|
| `ListModels` | `GET /models` |
| `Test` | `GET /models?limit=1` |
| `TestModel` | `GET /models/{id}` |

---

## Package `provider/openai/responses`

### Provider

```go
type Provider struct { /* unexported */ }

func New(options ...Option) *Provider
```

Implements `sdk.Provider`. Uses the OpenAI Responses API (`/responses`). Supports reasoning models (o3, o4-mini) with first-class reasoning summaries, URL citation annotations, and a flat input format.

#### Options

```go
type Option func(*Provider)

func WithAPIKey(apiKey string) Option
func WithBaseURL(baseURL string) Option
func WithHTTPClient(client *http.Client) Option
```

#### Methods

```go
func (p *Provider) Name() string                  // "openai-responses"
func (p *Provider) ChatModel(id string) *sdk.Model
func (p *Provider) ListModels(ctx context.Context) ([]sdk.Model, error)
func (p *Provider) Test(ctx context.Context) *sdk.ProviderTestResult
func (p *Provider) TestModel(ctx context.Context, modelID string) (*sdk.ModelTestResult, error)
func (p *Provider) DoGenerate(ctx, params) (*sdk.GenerateResult, error)
func (p *Provider) DoStream(ctx, params) (*sdk.StreamResult, error)
```

| Method | API Endpoint |
|--------|-------------|
| `ListModels` | `GET /models` |
| `Test` | `GET /models?limit=1` |
| `TestModel` | `GET /models/{id}` |

#### Responses API-Specific Behavior

**Input Conversion**: The provider converts `sdk.Message` types into the Responses API's flat input format:

| SDK Message | Responses Input Type |
|-------------|---------------------|
| System message | `{ "type": "message", "role": "system" }` |
| User message (text) | `{ "type": "message", "role": "user" }` |
| User message (image) | Content part with `{ "type": "input_image" }` |
| Assistant message | `{ "type": "message", "role": "assistant" }` |
| Assistant reasoning | `{ "type": "reasoning" }` item |
| Tool call | `{ "type": "function_call" }` |
| Tool result | `{ "type": "function_call_output" }` |

**Output Parsing**: Responses API output items are mapped to SDK types:

| Responses Output | SDK Result |
|-----------------|------------|
| `message` with text content | `GenerateResult.Text` |
| `reasoning` | `GenerateResult.Reasoning` |
| `function_call` | `GenerateResult.ToolCalls` |
| URL citation annotations | `GenerateResult.Sources` |

**Finish Reason Mapping**:

| API Condition | SDK FinishReason |
|--------------|-----------------|
| No `incomplete_details` | `stop` |
| `incomplete_details.reason == "max_output_tokens"` | `length` |
| `incomplete_details.reason == "content_filter"` | `content-filter` |
| Has function calls | `tool-calls` |

**Streaming Events**: The provider handles these SSE event types:

| SSE Event | SDK StreamPart |
|-----------|---------------|
| `response.output_text.delta` | `TextDeltaPart` |
| `response.reasoning_summary_text.delta` | `ReasoningDeltaPart` |
| `response.function_call_arguments.delta` | `ToolInputDeltaPart` |
| `response.output_item.done` (function_call) | `ToolInputEndPart` |
| `response.output_text.annotation.added` (url_citation) | `StreamSourcePart` |
| `response.completed` / `response.incomplete` | `FinishStepPart` + `FinishPart` |
