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

## OpenAI Codex Provider

The `provider/openai/codex` package provides an implementation for the [OpenAI Codex](https://openai.com/index/introducing-codex/) backend — a cloud-based coding agent powered by reasoning models. It communicates with the ChatGPT backend API via the Responses-style event stream at `/codex/responses`.

### When to Use Codex vs Completions / Responses

| | Chat Completions | Responses | Codex |
|--|---|---|---|
| **Endpoint** | `/chat/completions` | `/responses` | `/codex/responses` (ChatGPT backend) |
| **Authentication** | API key | API key | ChatGPT access token + account ID |
| **Reasoning models** | Basic | First-class | Native (encrypted reasoning content) |
| **Target use-case** | General chat | General + reasoning | Coding agents |
| **Compatibility** | Broad | OpenAI / OpenRouter | OpenAI Codex only |

Use **Codex** when you have a ChatGPT access token and want to leverage Codex-specific models (gpt-5.x-codex series) with encrypted reasoning support.

### Basic Usage

```go
import "github.com/memohai/twilight-ai/provider/openai/codex"

provider := codex.New(
    codex.WithAccessToken("eyJhbGci..."),
    codex.WithAccountID("acct_123"),
)
model := provider.ChatModel("gpt-5.2-codex")
```

The account ID is optional — if omitted, the provider extracts it from the JWT access token's `https://api.openai.com/auth` claim automatically.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `WithAccessToken(token)` | `""` | ChatGPT access token (JWT) sent as `Authorization: Bearer <token>` |
| `WithAPIKey(token)` | `""` | Alias for `WithAccessToken` for migration convenience |
| `WithAccountID(id)` | Auto-extracted from token | ChatGPT account ID sent as `chatgpt-account-id` header |
| `WithOriginator(name)` | `"codex_cli_rs"` | Originator identifier sent in the `originator` header |
| `WithBaseURL(url)` | `https://chatgpt.com/backend-api` | Base URL for API requests |
| `WithHTTPClient(client)` | `&http.Client{}` | Custom HTTP client |

### Available Models

The provider includes a static model catalog accessible via `codex.Catalog()`:

| Model | Reasoning Efforts |
|-------|-------------------|
| `gpt-5.2` | none, low, medium, high, xhigh |
| `gpt-5.2-codex` | low, medium, high, xhigh |
| `gpt-5.1-codex-max` | low, medium, high, xhigh |
| `gpt-5.1-codex` | low, medium, high |
| `gpt-5.1-codex-mini` | medium, high |
| `gpt-5.1` | none, low, medium, high |

### Reasoning

Codex models support reasoning with encrypted content preservation. When the model returns reasoning, the encrypted content is passed through via `ProviderMetadata` so it can be sent back in follow-up turns:

```go
effort := "high"
result, _ := sdk.GenerateTextResult(ctx,
    sdk.WithModel(provider.ChatModel("gpt-5.2-codex")),
    sdk.WithMessages([]sdk.Message{
        sdk.UserMessage("Refactor this function to use generics."),
    }),
    sdk.WithReasoningEffort(&effort),
)
fmt.Println(result.Reasoning) // reasoning summary
fmt.Println(result.Text)      // final answer
```

In streaming mode, reasoning arrives as `ReasoningStartPart` / `ReasoningDeltaPart` / `ReasoningEndPart` with encrypted content in `ProviderMetadata["openai"]["reasoningEncryptedContent"]`.

### Message Mapping

The Codex API uses a flat input format (similar to Responses). The provider converts SDK messages automatically:

| SDK Message | Codex Input |
|-------------|-------------|
| System message / `System` param | `instructions` field (joined with `\n\n`) |
| User message (text) | `{role: "user", content: [{type: "input_text", ...}]}` |
| User message (image) | `{role: "user", content: [{type: "input_image", ...}]}` |
| Assistant message (text) | `{role: "assistant", content: [{type: "output_text", ...}]}` |
| Assistant reasoning | `{type: "reasoning", summary: [...], encrypted_content: "..."}` |
| Tool call | `{type: "function_call", call_id, name, arguments}` |
| Tool result | `{type: "function_call_output", call_id, output}` |

### Supported Features

| Feature | Supported |
|---------|-----------|
| Text generation | ✅ |
| Streaming (SSE) | ✅ |
| Tool/function calling | ✅ |
| Vision (image inputs) | ✅ |
| Reasoning (encrypted content) | ✅ |
| Reasoning effort control | ✅ |
| JSON mode / JSON Schema | ✅ |
| Token usage reporting | ✅ |
| Cached / reasoning token details | ✅ |
| ListModels (static catalog) | ✅ |
| Test / TestModel | ✅ |

## GitHub Copilot Provider

The `provider/github/copilot` package implements GitHub Copilot's documented chat completions endpoint for Copilot agents/extensions at `https://api.githubcopilot.com/chat/completions`.

### When to Use Copilot vs Completions

| | GitHub Copilot | OpenAI Completions |
|--|---|---|
| **Endpoint** | `api.githubcopilot.com/chat/completions` | `/chat/completions` |
| **Authentication** | GitHub token issued to your Copilot agent/extension user context | API key |
| **Model discovery** | Static local catalog (`copilot-auto`) | `GET /models` |
| **Target use-case** | Copilot agents / extensions running inside GitHub's ecosystem | General OpenAI-compatible backends |

Use **GitHub Copilot** when your code is running as a Copilot agent/extension and GitHub has already provided the user-scoped token you must pass through to the Copilot LLM endpoint. Use **OpenAI Completions** for direct OpenAI or OpenAI-compatible integrations.

This is not a generic PAT-based GitHub Models integration.

### Basic Usage

```go
import "github.com/memohai/twilight-ai/provider/github/copilot"

provider := copilot.New(
    // Pass through the inbound X-GitHub-Token value from GitHub.
    copilot.WithGitHubToken(r.Header.Get("X-GitHub-Token")),
)
model := provider.ChatModel(copilot.AutoModel)
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `WithGitHubToken(token)` | `""` | User-scoped GitHub token sent as `Authorization: Bearer <token>` |
| `WithAPIKey(token)` | `""` | Alias for `WithGitHubToken` for migration convenience |
| `WithBaseURL(url)` | `https://api.githubcopilot.com` | Base URL for API requests |
| `WithHTTPClient(client)` | `&http.Client{}` | Custom HTTP client |

### Model Catalog

GitHub's documented Copilot LLM endpoint for agents/extensions does not expose a public `/models` discovery API. The provider therefore exposes a single static catalog entry that means "let GitHub choose":

| Model | Meaning |
|-------|---------|
| `copilot-auto` (`copilot.AutoModel`) | Let GitHub choose the backing Copilot model |

If GitHub documents or injects an explicit upstream model ID in your runtime, you can still pass that value to `provider.ChatModel("...")` and `TestModel("...")`. `AutoModel` is the safe default.

### Request Mapping

The provider reuses the same OpenAI-compatible chat-completions mapping as Twilight AI's OpenAI provider:

| SDK Message | Copilot API |
|-------------|-------------|
| System message / `System` param | `{"role":"system","content":"..."}` |
| User message | OpenAI-compatible `messages` entry |
| Assistant reasoning | `reasoning_content` field |
| Tool call | `tool_calls` |
| Tool result | `{"role":"tool","tool_call_id":...}` |

### Supported Features

| Feature | Supported |
|---------|-----------|
| Text generation | ✅ |
| Streaming (SSE) | ✅ |
| Tool/function calling | ✅ |
| Vision (image inputs) | ✅ |
| Reasoning content passthrough | ✅ |
| JSON mode / JSON Schema | ✅ |
| Token usage reporting | ✅ |
| ListModels (static catalog) | ✅ |
| Test / TestModel | ✅ |

### Limitations

- This provider is for GitHub Copilot agent / extension runtimes, not for GitHub Models.
- The provider intentionally exposes one sentinel local model because GitHub does not document a public model discovery API for this endpoint.
- `Test` / `TestModel` are implemented via a minimal probe request because there is no documented public model metadata API.

---

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

## Image Providers

Image providers implement the `sdk.ImageGenerationProvider` and/or `sdk.ImageEditProvider` interfaces and are separate from chat, embedding, and speech providers.

```go
type ImageGenerationProvider interface {
    DoGenerate(ctx context.Context, params *ImageGenerationParams) (*ImageResult, error)
}

type ImageEditProvider interface {
    DoEdit(ctx context.Context, params *ImageEditParams) (*ImageResult, error)
}
```

### OpenAI Images Provider

The `provider/openai/images` package provides image generation and editing via the OpenAI Images API (`/images/generations` and `/images/edits`).

#### Basic Usage

```go
import (
    "github.com/memohai/twilight-ai/provider/openai/images"
    "github.com/memohai/twilight-ai/sdk"
)

provider := images.New(
    images.WithAPIKey("sk-..."),
)

// Generation
genModel := provider.GenerationModel("gpt-image-1")
result, err := sdk.GenerateImage(ctx,
    sdk.WithImageGenerationModel(genModel),
    sdk.WithImagePrompt("A sunset over mountains"),
    sdk.WithImageSize("1024x1024"),
)

// Editing
editModel := provider.EditModel("gpt-image-1")
result, err := sdk.EditImage(ctx,
    sdk.WithImageEditModel(editModel),
    sdk.WithEditPrompt("Add a rainbow"),
    sdk.WithEditImages(sdk.ImageInput{
        Data:     pngBytes,
        Filename: "photo.png",
    }),
)
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `WithAPIKey(key)` | `""` | API key sent as `Authorization: Bearer <key>` |
| `WithBaseURL(url)` | `https://api.openai.com/v1` | Base URL for API requests |
| `WithHTTPClient(client)` | `&http.Client{}` | Custom HTTP client |

#### Supported Models

| Model | Generation | Editing | Notes |
|-------|-----------|---------|-------|
| `dall-e-2` | Yes | Yes | Legacy; `size`: 256/512/1024; `n`: 1-10 |
| `dall-e-3` | Yes | No | `n` must be 1; supports `style` (vivid/natural) |
| `gpt-image-1` | Yes | Yes | GPT Image; supports `background`, `output_format`, `moderation` |
| `gpt-image-1-mini` | Yes | Yes | Smaller GPT Image variant |
| `gpt-image-1.5` | Yes | Yes | Latest GPT Image |

#### Generation Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `Prompt` | string | **Required.** Text description of the desired image |
| `N` | *int | Number of images (1-10; dall-e-3 only supports 1) |
| `Size` | string | Image size (e.g. `"1024x1024"`, `"1536x1024"`) |
| `Quality` | string | `"auto"`, `"low"`, `"medium"`, `"high"`, `"standard"`, `"hd"` |
| `Style` | string | dall-e-3 only: `"vivid"`, `"natural"` |
| `ResponseFormat` | string | dall-e-2/3: `"url"`, `"b64_json"` |
| `Background` | string | GPT Image: `"transparent"`, `"opaque"`, `"auto"` |
| `OutputFormat` | string | GPT Image: `"png"`, `"jpeg"`, `"webp"` |
| `OutputCompression` | *int | GPT Image, jpeg/webp: 0-100 |
| `Moderation` | string | GPT Image: `"low"`, `"auto"` |

#### Edit Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `Images` | []ImageInput | Source images (up to 16 for GPT Image) |
| `Prompt` | string | **Required.** Edit description |
| `Mask` | *ImageInput | Mask image (transparent regions = edit area) |
| `InputFidelity` | string | GPT Image: `"high"`, `"low"` |
| Other params | — | Same as generation: `N`, `Size`, `Quality`, `Background`, `OutputFormat`, `OutputCompression`, `Moderation` |

#### Edit Input Modes

The provider automatically selects the request format based on the `ImageInput` fields:

| `ImageInput` field set | Request format | Use case |
|----------------------|----------------|----------|
| `Data` ([]byte) | `multipart/form-data` | File upload (local images) |
| `URL` or `FileID` | JSON body | URL/file-ID reference (GPT Image) |

#### OpenAI-Compatible Endpoints

Any service implementing the OpenAI Images API works with `WithBaseURL`:

```go
provider := images.New(
    images.WithAPIKey("your-key"),
    images.WithBaseURL("https://your-compatible-api.com/v1"),
)
```

See [Images](images.md) for complete documentation.

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

---

### OpenAI TTS Provider

The `provider/openai/speech` package targets the `/audio/speech` endpoint. It works with the official OpenAI TTS API and any OpenAI-compatible proxy (OpenRouter, CometAPI, Player2, Index-TTS vLLM, unspeech, etc.).

#### Basic Usage

```go
import "github.com/memohai/twilight-ai/provider/openai/speech"

provider := speech.New(
    speech.WithAPIKey("sk-..."),
)
model := provider.SpeechModel("tts-1")

result, err := sdk.GenerateSpeech(ctx,
    sdk.WithSpeechModel(model),
    sdk.WithText("Hello, world!"),
    sdk.WithSpeechConfig(map[string]any{
        "voice": "alloy",
    }),
)
```

#### Configuration Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `voice` | `string` | `alloy` | Voice ID: alloy/ash/ballad/coral/echo/fable/onyx/nova/shimmer |
| `response_format` | `string` | `mp3` | Output format: mp3/opus/aac/flac/wav/pcm |
| `speed` | `float64` | `0` (server default) | Speech rate 0.25–4.0 |
| `instructions` | `string` | `""` | Style instructions (gpt-4o-mini-tts only) |

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `WithAPIKey(key)` | `""` | API key for `Authorization: Bearer` |
| `WithBaseURL(url)` | `https://api.openai.com/v1` | Override for proxies or testing |
| `WithHTTPClient(client)` | `&http.Client{}` | Custom HTTP client |

#### OpenAI-Compatible Endpoints

```go
// OpenRouter
provider := speech.New(
    speech.WithAPIKey("sk-or-v1-..."),
    speech.WithBaseURL("https://openrouter.ai/api/v1"),
)

// Local Index-TTS via vLLM
provider := speech.New(
    speech.WithBaseURL("http://localhost:8000/v1"),
)
```

---

### ElevenLabs TTS Provider

The `provider/elevenlabs/speech` package targets the `/v1/text-to-speech/{voice_id}` (full) and `/v1/text-to-speech/{voice_id}/stream` (streaming) endpoints.

#### Basic Usage

```go
import "github.com/memohai/twilight-ai/provider/elevenlabs/speech"

provider := speech.New(
    speech.WithAPIKey("your-elevenlabs-key"),
)
model := provider.SpeechModel("elevenlabs-tts")

result, err := sdk.GenerateSpeech(ctx,
    sdk.WithSpeechModel(model),
    sdk.WithText("Hello!"),
    sdk.WithSpeechConfig(map[string]any{
        "voice_id": "21m00Tcm4TlvDq8ikWAM",
        "model_id": "eleven_multilingual_v2",
    }),
)
```

#### Configuration Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `voice_id` | `string` | `""` | **Required.** ElevenLabs voice ID |
| `model_id` | `string` | `eleven_multilingual_v2` | Model: eleven_turbo_v2_5/eleven_v3/… |
| `stability` | `float64` | `0.5` | Voice stability 0–1 |
| `similarity_boost` | `float64` | `0.75` | Voice similarity boost 0–1 |
| `output_format` | `string` | `mp3_44100_128` | Output format: mp3_44100_128/pcm_16000/… |
| `speed` | `float64` | `0` (server default) | Speech rate 0.25–4.0 |

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `WithAPIKey(key)` | `""` | API key sent as `xi-api-key` header |
| `WithBaseURL(url)` | `https://api.elevenlabs.io` | Override base URL |
| `WithHTTPClient(client)` | `&http.Client{}` | Custom HTTP client |

---

### Deepgram TTS Provider

The `provider/deepgram/speech` package targets the `POST /v1/speak` endpoint using `Authorization: Token` authentication.

#### Basic Usage

```go
import "github.com/memohai/twilight-ai/provider/deepgram/speech"

provider := speech.New(
    speech.WithAPIKey("your-deepgram-key"),
)
model := provider.SpeechModel("deepgram-tts")

result, err := sdk.GenerateSpeech(ctx,
    sdk.WithSpeechModel(model),
    sdk.WithText("Hello!"),
    sdk.WithSpeechConfig(map[string]any{
        "model": "aura-2-asteria-en",
    }),
)
```

#### Configuration Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | `string` | `aura-2-asteria-en` | Voice model: aura-2-*/aura-* series |
| `encoding` | `string` | `""` | Audio encoding: linear16/mulaw/alaw |
| `sample_rate` | `int` | `0` (server default) | Sample rate in Hz |
| `container` | `string` | `""` | Container format: wav/none |

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `WithAPIKey(key)` | `""` | API key sent as `Authorization: Token` |
| `WithBaseURL(url)` | `https://api.deepgram.com` | Override base URL |
| `WithHTTPClient(client)` | `&http.Client{}` | Custom HTTP client |

---

### MiniMax TTS Provider

The `provider/minimax/speech` package targets the `POST /v1/t2a_v2` endpoint. The API returns audio as a hex-encoded string inside a JSON response; this provider decodes it automatically.

#### Basic Usage

```go
import "github.com/memohai/twilight-ai/provider/minimax/speech"

provider := speech.New(
    speech.WithAPIKey("your-minimax-key"),
)
model := provider.SpeechModel("minimax-tts")

result, err := sdk.GenerateSpeech(ctx,
    sdk.WithSpeechModel(model),
    sdk.WithText("你好，世界！"),
    sdk.WithSpeechConfig(map[string]any{
        "voice_id": "Chinese_narrator_female",
        "model":    "speech-2.8-hd",
    }),
)
```

#### Configuration Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `voice_id` | `string` | `English_expressive_narrator` | Voice ID |
| `model` | `string` | `speech-2.8-hd` | Model version |
| `speed` | `float64` | `0` (server default) | Speech speed |
| `vol` | `float64` | `0` (server default) | Volume |
| `pitch` | `int` | `0` | Pitch adjustment |
| `output_format` | `string` | `mp3` | Output format: mp3/pcm/flac/wav |
| `sample_rate` | `int` | `32000` | Sample rate in Hz |

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `WithAPIKey(key)` | `""` | API key for `Authorization: Bearer` |
| `WithBaseURL(url)` | `https://api.minimax.io` | Override base URL |
| `WithHTTPClient(client)` | `&http.Client{}` | Custom HTTP client |

> **Note:** MiniMax streaming TTS is a paid-tier API feature. `DoStream` returns the fully synthesized audio as a single chunk (equivalent to `DoSynthesize`).

---

### Alibaba Cloud DashScope CosyVoice Provider

The `provider/alibabacloud/speech` package implements the DashScope CosyVoice WebSocket protocol (`wss://dashscope.aliyuncs.com/api-ws/v1/inference/`). It uses the `run-task` / `continue-task` / `finish-task` message flow with Bearer authentication during the WebSocket handshake.

#### Basic Usage

```go
import "github.com/memohai/twilight-ai/provider/alibabacloud/speech"

provider := speech.New(
    speech.WithAPIKey("your-dashscope-api-key"),
)
model := provider.SpeechModel("cosyvoice-tts")

result, err := sdk.GenerateSpeech(ctx,
    sdk.WithSpeechModel(model),
    sdk.WithText("你好，世界！"),
    sdk.WithSpeechConfig(map[string]any{
        "model": "cosyvoice-v1",
        "voice": "longanyang",
    }),
)
```

#### Configuration Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | `string` | `cosyvoice-v1` | CosyVoice model: cosyvoice-v1/cosyvoice-v2/cosyvoice-v3-flash/… |
| `voice` | `string` | `""` | Voice ID (system voice or custom clone ID) |
| `format` | `string` | `mp3` | Output format: mp3/wav/pcm/opus |
| `sample_rate` | `int` | `22050` | Sample rate in Hz |
| `volume` | `int` | `0` (server default) | Volume 0–100 |
| `rate` | `float64` | `0` (server default) | Speech rate 0.5–2.0 |
| `pitch` | `float64` | `0` (server default) | Pitch multiplier 0.5–2.0 |

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `WithAPIKey(key)` | `""` | DashScope API key (Bearer auth during WS handshake) |
| `WithBaseURL(url)` | `wss://dashscope.aliyuncs.com/api-ws/v1/inference/` | Override WebSocket endpoint |

---

### Volcengine SAMI TTS Provider

The `provider/volcengine/speech` package implements the Volcengine SAMI (Speech Audio Machine Intelligence) TTS API. Authentication uses a two-step process: first obtain a bearer token via `open.volcengineapi.com/GetToken` (signed with Volcengine V4 HMAC-SHA256), then call `POST https://sami.bytedance.com/api/v1/invoke`.

#### Basic Usage

```go
import "github.com/memohai/twilight-ai/provider/volcengine/speech"

provider := speech.New(
    speech.WithAccessKey("your-access-key"),
    speech.WithSecretKey("your-secret-key"),
    speech.WithAppKey("your-app-key"),
)
model := provider.SpeechModel("sami-tts")

result, err := sdk.GenerateSpeech(ctx,
    sdk.WithSpeechModel(model),
    sdk.WithText("你好，世界！"),
    sdk.WithSpeechConfig(map[string]any{
        "speaker": "zh_female_qingxin",
    }),
)
```

Tokens obtained via `GetToken` are cached for their full validity period (1 hour) and reused automatically.

#### Configuration Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `speaker` | `string` | `""` | **Required.** SAMI voice speaker ID |
| `encoding` | `string` | `mp3` | Output format: mp3/wav/aac |
| `sample_rate` | `int` | `24000` | Sample rate in Hz |
| `speech_rate` | `int` | `0` | Speech rate [-50, 100] (100 = 2× speed) |
| `pitch_rate` | `int` | `0` | Pitch adjustment [-12, 12] |

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `WithAccessKey(key)` | `""` | Volcengine AccessKeyID |
| `WithSecretKey(key)` | `""` | Volcengine SecretAccessKey |
| `WithAppKey(key)` | `""` | SAMI Application AppKey |
| `WithBaseURL(url)` | `https://sami.bytedance.com` | Override SAMI base URL |
| `WithHTTPClient(client)` | `&http.Client{}` | Custom HTTP client |
| `WithToken(token)` | `""` | Inject a static pre-obtained token (skips GetToken; useful for testing) |

> **Note:** SAMI streaming is not exposed via a public non-SDK endpoint. `DoStream` wraps `DoSynthesize` and returns the fully synthesized audio as a single chunk.

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

- [Images](images.md) — generate and edit images with OpenAI image models
- [Embeddings](embeddings.md) — generate vector embeddings with OpenAI and Google
- [Speech](speech.md) — speech synthesis with Edge TTS and custom providers
- [Tool Calling](tools.md) — define tools and enable multi-step execution
- [Streaming](streaming.md) — understand StreamPart types
- [API Reference](api-reference.md) — complete type and function reference
