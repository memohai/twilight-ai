# Providers

A **Provider** is the abstraction that connects the SDK to an AI backend. It handles HTTP communication, request/response mapping, and streaming protocol details.

## The Provider Interface

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
| `Name()` | Returns a human-readable provider identifier (e.g. `"openai-completions"`) |
| `ListModels(ctx)` | Fetches available models from the backend API |
| `Test(ctx)` | Health check returning one of three states (see below) |
| `TestModel(ctx, id)` | Checks whether a specific model ID is supported |
| `DoGenerate()` | Performs a single non-streaming LLM call |
| `DoStream()` | Performs a streaming LLM call, returning a channel of `StreamPart` |

The SDK never calls a provider directly — it goes through the `Client` which adds orchestration (tool loop, callbacks, multi-step). The `Model` struct carries a reference to its provider:

```go
type Model struct {
    ID          string
    DisplayName string
    Provider    Provider
    Type        ModelType   // "chat"
    MaxTokens   int
}
```

`Model` also has a `Test(ctx)` method that delegates to `Provider.TestModel`.

### Provider Health Check

`Test(ctx)` returns a `*ProviderTestResult` with one of three statuses:

| Status | Meaning |
|--------|---------|
| `ProviderStatusOK` | Connected and API key is valid |
| `ProviderStatusUnhealthy` | TCP connection succeeded but API returned an error (e.g. 401/403 auth failure) |
| `ProviderStatusUnreachable` | Cannot establish a network connection to the endpoint |

```go
result := provider.Test(ctx)
if result.Status != sdk.ProviderStatusOK {
    log.Fatalf("provider issue: %s (error: %v)", result.Message, result.Error)
}
```

### Model Discovery

`ListModels(ctx)` returns all models available from the provider. Each returned `Model` is bound to the provider and ready for use:

```go
models, err := provider.ListModels(ctx)
for _, m := range models {
    fmt.Printf("%-40s %s\n", m.ID, m.DisplayName)
}
```

To check a single model without listing all:

```go
model := provider.ChatModel("gpt-4o")
result, err := model.Test(ctx)
if result.Supported {
    // safe to use this model
}
```

## OpenAI Completions Provider

The `provider/openai/completions` package provides an implementation for the OpenAI Chat Completions API (`/chat/completions`).

### Basic Usage

```go
import "github.com/memohai/twilight-ai/provider/openai/completions"

provider := completions.New(
    completions.WithAPIKey("sk-..."),
)
model := provider.ChatModel("gpt-4o-mini")
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `WithAPIKey(key)` | `""` | API key sent as `Authorization: Bearer <key>` |
| `WithBaseURL(url)` | `https://api.openai.com/v1` | Base URL for API requests |
| `WithHTTPClient(client)` | `&http.Client{}` | Custom HTTP client (for proxies, timeouts, etc.) |

### API Endpoints for Discovery

| Method | API Endpoint |
|--------|-------------|
| `ListModels` | `GET /models` |
| `Test` | `GET /models?limit=1` |
| `TestModel` | `GET /models/{id}` |

### OpenAI-Compatible Providers

Any service that implements the OpenAI Chat Completions API works out of the box:

```go
// DeepSeek
provider := completions.New(
    completions.WithAPIKey("your-deepseek-key"),
    completions.WithBaseURL("https://api.deepseek.com"),
)

// Groq
provider := completions.New(
    completions.WithAPIKey("your-groq-key"),
    completions.WithBaseURL("https://api.groq.com/openai/v1"),
)

// Azure OpenAI
provider := completions.New(
    completions.WithAPIKey("your-azure-key"),
    completions.WithBaseURL("https://your-resource.openai.azure.com/openai/deployments/gpt-4o"),
)

// Local (Ollama, vLLM, etc.)
provider := completions.New(
    completions.WithBaseURL("http://localhost:11434/v1"),
)
```

### Supported Features

| Feature | Supported |
|---------|-----------|
| Chat completions | ✅ |
| Streaming (SSE) | ✅ |
| Tool/function calling | ✅ |
| Vision (image inputs) | ✅ |
| Reasoning content (o1, DeepSeek-R1) | ✅ |
| JSON mode / JSON Schema | ✅ |
| Token usage reporting | ✅ |
| Cached token details | ✅ |
| ListModels / Test / TestModel | ✅ |

### Custom HTTP Client

Use `WithHTTPClient` for custom timeouts, proxies, or TLS settings:

```go
provider := completions.New(
    completions.WithAPIKey("sk-..."),
    completions.WithHTTPClient(&http.Client{
        Timeout: 120 * time.Second,
        Transport: &http.Transport{
            Proxy: http.ProxyFromEnvironment,
        },
    }),
)
```

## OpenAI Responses Provider

The `provider/openai/responses` package provides an implementation for the OpenAI Responses API (`/responses`). This is OpenAI's newer API that offers first-class reasoning support, URL citation annotations, and a flat input format.

### When to Use Responses vs Completions

| | Chat Completions | Responses |
|--|---|---|
| **Endpoint** | `/chat/completions` | `/responses` |
| **Reasoning models** | Basic support (`reasoning_content` field) | First-class (`reasoning` output items with summaries) |
| **Citations** | Not supported | URL citations via annotations |
| **Input format** | Nested `messages` array | Flat `input` array |
| **Compatibility** | Broad (DeepSeek, Groq, Ollama, etc.) | OpenAI and OpenRouter |

Use **Completions** when you need broad compatibility with OpenAI-compatible endpoints. Use **Responses** when you want native reasoning model support (o3, o4-mini) or URL citation annotations.

### Basic Usage

```go
import "github.com/memohai/twilight-ai/provider/openai/responses"

provider := responses.New(
    responses.WithAPIKey("sk-..."),
)
model := provider.ChatModel("gpt-4o-mini")
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `WithAPIKey(key)` | `""` | API key sent as `Authorization: Bearer <key>` |
| `WithBaseURL(url)` | `https://api.openai.com/v1` | Base URL for API requests |
| `WithHTTPClient(client)` | `&http.Client{}` | Custom HTTP client |

### API Endpoints for Discovery

| Method | API Endpoint |
|--------|-------------|
| `ListModels` | `GET /models` |
| `Test` | `GET /models?limit=1` |
| `TestModel` | `GET /models/{id}` |

### Using with OpenRouter

OpenRouter supports the Responses API as a beta feature:

```go
provider := responses.New(
    responses.WithAPIKey("sk-or-v1-..."),
    responses.WithBaseURL("https://openrouter.ai/api/v1"),
)
model := provider.ChatModel("openai/o4-mini")
```

### Reasoning Models

Reasoning models (o3, o4-mini) return both reasoning summaries and the final answer:

```go
effort := "medium"
result, _ := sdk.GenerateTextResult(ctx,
    sdk.WithModel(provider.ChatModel("openai/o4-mini")),
    sdk.WithMessages([]sdk.Message{
        sdk.UserMessage("What is 15 * 37? Think step by step."),
    }),
    sdk.WithReasoningEffort(&effort),
)
fmt.Println(result.Reasoning)  // model's reasoning summary
fmt.Println(result.Text)       // final answer: "555"
```

In streaming mode, reasoning arrives as `ReasoningStartPart` / `ReasoningDeltaPart` / `ReasoningEndPart` before the text content.

### Supported Features

| Feature | Supported |
|---------|-----------|
| Text generation | ✅ |
| Streaming (SSE) | ✅ |
| Tool/function calling | ✅ |
| Vision (image inputs) | ✅ |
| Reasoning summaries (o3, o4-mini) | ✅ |
| URL citation annotations | ✅ |
| JSON mode / JSON Schema | ✅ |
| Token usage reporting | ✅ |
| Cached / reasoning token details | ✅ |
| ListModels / Test / TestModel | ✅ |

## Anthropic Provider

The `provider/anthropic/messages` package implements the [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) for Claude models.

### Basic Usage

```go
import "github.com/memohai/twilight-ai/provider/anthropic/messages"

provider := messages.New(
    messages.WithAPIKey("sk-ant-..."),
)
model := provider.ChatModel("claude-sonnet-4-20250514")
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `WithAPIKey(key)` | `""` | API key sent as `x-api-key` header |
| `WithAuthToken(token)` | `""` | OAuth token sent as `Authorization: Bearer <token>` |
| `WithBaseURL(url)` | `https://api.anthropic.com` | Base URL for API requests |
| `WithHTTPClient(client)` | `&http.Client{}` | Custom HTTP client |
| `WithThinking(config)` | `nil` | Enable extended thinking for reasoning |

### Extended Thinking

Claude supports [extended thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking) (chain-of-thought reasoning):

```go
provider := messages.New(
    messages.WithAPIKey("sk-ant-..."),
    messages.WithThinking(messages.ThinkingConfig{
        Type:         "enabled",
        BudgetTokens: 10000,
    }),
)
```

When enabled, the model's internal reasoning appears in `result.Reasoning` (non-streaming) or as `ReasoningStartPart` / `ReasoningDeltaPart` / `ReasoningEndPart` events (streaming).

### Supported Features

| Feature | Supported |
|---------|-----------|
| Text generation | ✅ |
| Streaming (SSE) | ✅ |
| Tool/function calling | ✅ |
| Vision (image inputs) | ✅ |
| Extended thinking | ✅ |
| Token usage reporting | ✅ |
| Cached token details | ✅ |
| ListModels / Test / TestModel | ✅ |

### API Endpoints for Discovery

| Method | API Endpoint |
|--------|-------------|
| `ListModels` | `GET /v1/models` |
| `Test` | `GET /v1/models?limit=1` |
| `TestModel` | `GET /v1/models/{id}` |

---

## Google Gemini Provider

The `provider/google/generativeai` package implements the [Google Generative AI API](https://ai.google.dev/api) for Gemini models.

### Basic Usage

```go
import "github.com/memohai/twilight-ai/provider/google/generativeai"

provider := generativeai.New(
    generativeai.WithAPIKey("AIza..."),
)
model := provider.ChatModel("gemini-2.5-flash")
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `WithAPIKey(key)` | `""` | API key sent as `x-goog-api-key` header |
| `WithBaseURL(url)` | `https://generativelanguage.googleapis.com/v1beta` | Base URL |
| `WithHTTPClient(client)` | `&http.Client{}` | Custom HTTP client |

### Model ID

The model ID can be a simple name or a full resource path:

```go
// Simple name — resolved to "models/gemini-2.5-flash"
model := provider.ChatModel("gemini-2.5-flash")

// Full path — used as-is
model := provider.ChatModel("publishers/google/models/gemini-2.5-flash")
```

### API Endpoints

| Operation | Endpoint |
|-----------|----------|
| Non-streaming | `POST {baseURL}/models/{modelId}:generateContent` |
| Streaming | `POST {baseURL}/models/{modelId}:streamGenerateContent?alt=sse` |

### How Messages Are Mapped

The provider automatically converts SDK messages to Google's format:

| SDK | Google API |
|-----|-----------|
| `System` param | `systemInstruction` field (separate from `contents`) |
| User message | `{role: "user", parts: [{text: "..."}, ...]}` |
| Assistant message | `{role: "model", parts: [{text: "..."}, {functionCall: ...}]}` |
| Tool result message | `{role: "user", parts: [{functionResponse: {name, response}}]}` |

### Tool Choice Mapping

| SDK `ToolChoice` | Google `functionCallingConfig.mode` |
|------------------|-------------------------------------|
| `"auto"` | `AUTO` |
| `"none"` | `NONE` |
| `"required"` | `ANY` |

### Thinking / Reasoning

Gemini 2.5+ models support thinking (reasoning). The model returns parts with `thought: true` which the provider maps to `Reasoning` in the result:

```go
provider := generativeai.New(generativeai.WithAPIKey("AIza..."))
model := provider.ChatModel("gemini-2.5-flash")

result, _ := sdk.GenerateTextResult(ctx,
    sdk.WithModel(model),
    sdk.WithMessages([]sdk.Message{
        sdk.UserMessage("What is 15 * 37? Think step by step."),
    }),
)
fmt.Println(result.Reasoning) // model's thinking process
fmt.Println(result.Text)      // final answer
```

### Supported Features

| Feature | Supported |
|---------|-----------|
| Text generation | ✅ |
| Streaming (SSE) | ✅ |
| Tool/function calling | ✅ |
| Vision (image inputs) | ✅ |
| Thinking / Reasoning (Gemini 2.5+) | ✅ |
| JSON mode | ✅ |
| Token usage reporting | ✅ |
| Cached content token details | ✅ |
| ListModels / Test / TestModel | ✅ |

### API Endpoints for Discovery

| Method | API Endpoint |
|--------|-------------|
| `ListModels` | `GET /v1beta/models` |
| `Test` | `GET /v1beta/models?pageSize=1` |
| `TestModel` | `GET /v1beta/models/{id}` |

---

## Embedding Providers

Embedding providers implement the `sdk.EmbeddingProvider` interface and are separate from chat providers. They generate vector representations of text for use in search, retrieval, clustering, and other similarity-based tasks.

```go
type EmbeddingProvider interface {
    DoEmbed(ctx context.Context, params EmbedParams) (*EmbedResult, error)
}
```

### OpenAI Embedding Provider

The `provider/openai/embedding` package provides text embeddings via the OpenAI `/embeddings` endpoint.

#### Basic Usage

```go
import (
    "github.com/memohai/twilight-ai/provider/openai/embedding"
    "github.com/memohai/twilight-ai/sdk"
)

provider := embedding.New(
    embedding.WithAPIKey("sk-..."),
)
model := provider.EmbeddingModel("text-embedding-3-small")

// Single embedding
vec, err := sdk.Embed(ctx, "Hello world", sdk.WithEmbeddingModel(model))

// Batch embeddings
result, err := sdk.EmbedMany(ctx,
    []string{"Hello", "World"},
    sdk.WithEmbeddingModel(model),
)
```

#### Custom Dimensions

Models like `text-embedding-3-small` and `text-embedding-3-large` support custom output dimensions:

```go
vec, err := sdk.Embed(ctx, "Hello world",
    sdk.WithEmbeddingModel(model),
    sdk.WithDimensions(256),
)
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `WithAPIKey(key)` | `""` | API key sent as `Authorization: Bearer <key>` |
| `WithBaseURL(url)` | `https://api.openai.com/v1` | Base URL for API requests |
| `WithHTTPClient(client)` | `&http.Client{}` | Custom HTTP client |

#### OpenAI-Compatible Endpoints

Any service that implements the OpenAI Embeddings API works:

```go
// Ollama
provider := embedding.New(
    embedding.WithBaseURL("http://localhost:11434/v1"),
)
model := provider.EmbeddingModel("nomic-embed-text")
```

### Google Embedding Provider

The `provider/google/embedding` package provides text embeddings via the Google Generative AI API.

#### Basic Usage

```go
import (
    "github.com/memohai/twilight-ai/provider/google/embedding"
    "github.com/memohai/twilight-ai/sdk"
)

provider := embedding.New(
    embedding.WithAPIKey("AIza..."),
)
model := provider.EmbeddingModel("gemini-embedding-001")

vec, err := sdk.Embed(ctx, "Hello world", sdk.WithEmbeddingModel(model))
```

#### Task Types

Google embedding models support a `taskType` parameter to optimize the embedding for a specific use case:

```go
provider := embedding.New(
    embedding.WithAPIKey("AIza..."),
    embedding.WithTaskType("RETRIEVAL_DOCUMENT"),
)
```

| Task Type | Use Case |
|-----------|----------|
| `RETRIEVAL_QUERY` | Query text for search/retrieval |
| `RETRIEVAL_DOCUMENT` | Document text being indexed |
| `SEMANTIC_SIMILARITY` | Comparing text similarity |
| `CLASSIFICATION` | Text classification |
| `CLUSTERING` | Text clustering |
| `QUESTION_ANSWERING` | Question answering |
| `FACT_VERIFICATION` | Fact verification |
| `CODE_RETRIEVAL_QUERY` | Code search queries |

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `WithAPIKey(key)` | `""` | API key sent as `x-goog-api-key` header |
| `WithBaseURL(url)` | `https://generativelanguage.googleapis.com/v1beta` | Base URL |
| `WithHTTPClient(client)` | `&http.Client{}` | Custom HTTP client |
| `WithTaskType(taskType)` | `""` | Default task type for all requests |

#### API Endpoints

| Scenario | Endpoint |
|----------|----------|
| Single value | `POST {baseURL}/models/{modelId}:embedContent` |
| Multiple values | `POST {baseURL}/models/{modelId}:batchEmbedContents` |

The provider automatically selects the optimal endpoint based on the number of input values.

---

## Speech Providers

Speech providers implement the `sdk.SpeechProvider` interface and are separate from chat and embedding providers. They convert text into audio.

```go
type SpeechProvider interface {
    DoSynthesize(ctx context.Context, params SpeechParams) (*SpeechResult, error)
    DoStream(ctx context.Context, params SpeechParams) (*SpeechStreamResult, error)
}
```

Speech uses an open-ended `Config map[string]any` in `SpeechParams`, allowing each provider to define its own configuration keys.

### Edge TTS Provider

The `provider/edge/speech` package provides free speech synthesis via Microsoft Edge's built-in TTS service. No API key required.

#### Basic Usage

```go
import (
    "github.com/memohai/twilight-ai/provider/edge/speech"
    "github.com/memohai/twilight-ai/sdk"
)

provider := speech.New()
model := provider.SpeechModel("edge-read-aloud")

result, err := sdk.GenerateSpeech(ctx,
    sdk.WithSpeechModel(model),
    sdk.WithText("Hello, world!"),
    sdk.WithSpeechConfig(map[string]any{
        "voice": "en-US-EmmaMultilingualNeural",
        "speed": 1.0,
    }),
)
```

#### Configuration Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `voice` | `string` | `en-US-EmmaMultilingualNeural` | Edge TTS voice ID |
| `language` | `string` | Auto-detected from voice | BCP-47 language tag |
| `format` | `string` | `audio-24khz-48kbitrate-mono-mp3` | Output audio format |
| `speed` | `float64` | `0` (server default) | Speech rate (1.0 = normal) |
| `pitch` | `float64` | `0` | Pitch adjustment in Hz |

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `WithBaseURL(url)` | Bing WSS endpoint | Override WebSocket endpoint (for testing) |

See [Speech](speech.md) for complete documentation including voices, streaming, and custom provider implementation.

---

## Implementing a Custom Provider

To add support for a new AI backend, implement the `sdk.Provider` interface:

```go
package myprovider

import (
    "context"
    "github.com/memohai/twilight-ai/sdk"
)

type MyProvider struct {
    apiKey string
}

func New(apiKey string) *MyProvider {
    return &MyProvider{apiKey: apiKey}
}

func (p *MyProvider) Name() string {
    return "my-provider"
}

func (p *MyProvider) ListModels(ctx context.Context) ([]sdk.Model, error) {
    // Fetch models from your backend's API
    return []sdk.Model{
        {ID: "my-model-v1", Provider: p, Type: sdk.ModelTypeChat},
    }, nil
}

func (p *MyProvider) Test(ctx context.Context) *sdk.ProviderTestResult {
    // Try a lightweight API call to verify connectivity
    _, err := p.ListModels(ctx)
    if err != nil {
        return &sdk.ProviderTestResult{
            Status:  sdk.ProviderStatusUnreachable,
            Message: err.Error(),
            Error:   err,
        }
    }
    return &sdk.ProviderTestResult{Status: sdk.ProviderStatusOK, Message: "ok"}
}

func (p *MyProvider) TestModel(ctx context.Context, modelID string) (*sdk.ModelTestResult, error) {
    // Check if a specific model exists
    models, err := p.ListModels(ctx)
    if err != nil {
        return nil, err
    }
    for _, m := range models {
        if m.ID == modelID {
            return &sdk.ModelTestResult{Supported: true, Message: "supported"}, nil
        }
    }
    return &sdk.ModelTestResult{Supported: false, Message: "model not found"}, nil
}

func (p *MyProvider) ChatModel(id string) *sdk.Model {
    return &sdk.Model{ID: id, Provider: p, Type: sdk.ModelTypeChat}
}

func (p *MyProvider) DoGenerate(ctx context.Context, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
    // Make HTTP request to your backend...
    return &sdk.GenerateResult{
        Text:         "response text",
        FinishReason: sdk.FinishReasonStop,
    }, nil
}

func (p *MyProvider) DoStream(ctx context.Context, params sdk.GenerateParams) (*sdk.StreamResult, error) {
    ch := make(chan sdk.StreamPart, 64)

    go func() {
        defer close(ch)
        ch <- &sdk.StartPart{}
        ch <- &sdk.StartStepPart{}
        ch <- &sdk.TextStartPart{}
        ch <- &sdk.TextDeltaPart{Text: "Hello"}
        ch <- &sdk.TextEndPart{}
        ch <- &sdk.FinishStepPart{FinishReason: sdk.FinishReasonStop}
        ch <- &sdk.FinishPart{FinishReason: sdk.FinishReasonStop}
    }()

    return &sdk.StreamResult{Stream: ch}, nil
}
```

Then use it exactly like the built-in provider:

```go
provider := myprovider.New("my-key")
model := provider.ChatModel("my-model-v1")

text, err := sdk.GenerateText(ctx,
    sdk.WithModel(model),
    sdk.WithMessages([]sdk.Message{sdk.UserMessage("Hello")}),
)
```

## Next Steps

- [Embeddings](embeddings.md) — generate vector embeddings with OpenAI and Google
- [Speech](speech.md) — speech synthesis with Edge TTS and custom providers
- [Tool Calling](tools.md) — define tools and enable multi-step execution
- [Streaming](streaming.md) — understand StreamPart types
- [API Reference](api-reference.md) — complete type and function reference
