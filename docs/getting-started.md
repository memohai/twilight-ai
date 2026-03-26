# Getting Started

This guide walks you through installing Twilight AI and making your first LLM request.

## Prerequisites

- Go 1.25 or later
- An API key from OpenAI, Anthropic, Google, or any OpenAI-compatible provider

## Installation

```bash
go get github.com/memohai/twilight-ai
```

## Setup

### 1. Create a Provider

A **Provider** is the bridge between the SDK and a specific AI backend. The SDK ships with two OpenAI providers:

**Chat Completions** — broad compatibility with OpenAI and all OpenAI-compatible APIs:

```go
import "github.com/memohai/twilight-ai/provider/openai/completions"

provider := completions.New(
    completions.WithAPIKey("sk-..."),
)
```

**Responses** — OpenAI's newer API with native reasoning and citation support:

```go
import "github.com/memohai/twilight-ai/provider/openai/responses"

provider := responses.New(
    responses.WithAPIKey("sk-..."),
)
```

For OpenAI-compatible endpoints, add a custom base URL:

```go
// Completions API
provider := completions.New(
    completions.WithAPIKey("your-key"),
    completions.WithBaseURL("https://api.deepseek.com/v1"),
)

// Responses API (e.g. via OpenRouter)
provider := responses.New(
    responses.WithAPIKey("sk-or-v1-..."),
    responses.WithBaseURL("https://openrouter.ai/api/v1"),
)
```

### 2. Get a Model

```go
model := provider.ChatModel("gpt-4o-mini")
```

The model carries a reference to its provider, so the SDK knows which backend to call.

### 3. Generate Text

The simplest way to get a response:

```go
import "github.com/memohai/twilight-ai/sdk"

text, err := sdk.GenerateText(ctx,
    sdk.WithModel(model),
    sdk.WithMessages([]sdk.Message{
        sdk.UserMessage("What is the capital of France?"),
    }),
)
// text == "The capital of France is Paris."
```

### 4. Get the Full Result

If you need token usage, finish reason, or other metadata:

```go
result, err := sdk.GenerateTextResult(ctx,
    sdk.WithModel(model),
    sdk.WithSystem("You are a helpful assistant."),
    sdk.WithMessages([]sdk.Message{
        sdk.UserMessage("What is the capital of France?"),
    }),
)

fmt.Println(result.Text)                      // response text
fmt.Println(result.Usage.TotalTokens)         // token count
fmt.Println(result.FinishReason)              // "stop"
fmt.Println(result.Response.ModelID)          // "gpt-4o-mini"
```

### 5. Stream the Response

For real-time output, use `StreamText`:

```go
sr, err := sdk.StreamText(ctx,
    sdk.WithModel(model),
    sdk.WithMessages([]sdk.Message{
        sdk.UserMessage("Tell me a short story."),
    }),
)
if err != nil {
    log.Fatal(err)
}

for part := range sr.Stream {
    switch p := part.(type) {
    case *sdk.TextDeltaPart:
        fmt.Print(p.Text)
    case *sdk.ErrorPart:
        log.Fatal(p.Error)
    }
}
fmt.Println() // newline after streaming
```

Or use the convenience helper to collect all text at once:

```go
sr, err := sdk.StreamText(ctx, sdk.WithModel(model), ...)
text, err := sr.Text()
```

### 6. Test Provider Connectivity

Before making generation requests, you can verify that your API key and endpoint are working:

```go
result := provider.Test(ctx)
switch result.Status {
case sdk.ProviderStatusOK:
    fmt.Println("Ready to go!")
case sdk.ProviderStatusUnhealthy:
    fmt.Printf("Connected but auth failed: %s\n", result.Message)
case sdk.ProviderStatusUnreachable:
    fmt.Printf("Cannot reach endpoint: %s\n", result.Message)
}
```

### 7. Discover Available Models

List all models your API key has access to:

```go
models, err := provider.ListModels(ctx)
for _, m := range models {
    fmt.Println(m.ID)
}
```

Or check a single model:

```go
model := provider.ChatModel("gpt-4o")
testResult, _ := model.Test(ctx)
fmt.Println(testResult.Supported) // true or false
```

### 6. Generate Speech

Speech synthesis converts text to audio. Speech providers are separate from chat providers:

```go
import "github.com/memohai/twilight-ai/provider/edge/speech"

speechProvider := speech.New()
speechModel := speechProvider.SpeechModel("edge-read-aloud")

result, err := sdk.GenerateSpeech(ctx,
    sdk.WithSpeechModel(speechModel),
    sdk.WithText("Hello, this is a speech test."),
    sdk.WithSpeechConfig(map[string]any{
        "voice": "en-US-EmmaMultilingualNeural",
    }),
)
// result.Audio is []byte (raw audio data)
// result.ContentType is "audio/mpeg"

// Or stream for low-latency playback:
sr, err := sdk.StreamSpeech(ctx,
    sdk.WithSpeechModel(speechModel),
    sdk.WithText("Streaming speech synthesis."),
)
for chunk := range sr.Stream {
    // write each chunk...
}
```

Edge TTS is free and requires no API key. See [Speech](speech.md) for all available voices and configuration options.

### 7. Generate Embeddings

Embeddings convert text into numeric vectors for search, retrieval, and similarity comparison. Embedding providers are separate from chat providers:

```go
import "github.com/memohai/twilight-ai/provider/openai/embedding"

embProvider := embedding.New(
    embedding.WithAPIKey("sk-..."),
)
embModel := embProvider.EmbeddingModel("text-embedding-3-small")

vec, err := sdk.Embed(ctx, "What is the capital of France?",
    sdk.WithEmbeddingModel(embModel),
)
// vec is a []float64 vector

// Embed multiple texts at once
result, err := sdk.EmbedMany(ctx, []string{"Paris", "London", "Berlin"},
    sdk.WithEmbeddingModel(embModel),
)
// result.Embeddings[0] is the vector for "Paris"
// result.Usage.Tokens reports total tokens consumed
```

For Google embeddings, use the Google embedding provider:

```go
import "github.com/memohai/twilight-ai/provider/google/embedding"

embProvider := embedding.New(
    embedding.WithAPIKey("AIza..."),
    embedding.WithTaskType("RETRIEVAL_QUERY"),
)
embModel := embProvider.EmbeddingModel("gemini-embedding-001")

vec, err := sdk.Embed(ctx, "search query here",
    sdk.WithEmbeddingModel(embModel),
)
```

## Using a Client Instance

The package-level functions (`sdk.GenerateText`, `sdk.StreamText`) use a default client. You can also create your own:

```go
client := sdk.NewClient()
text, err := client.GenerateText(ctx, sdk.WithModel(model), ...)
```

This is useful when you need multiple clients with different configurations.

## Environment Variables

The SDK itself does not read environment variables, but the test suite supports a `.env` file with:

```
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

## Next Steps

- [Providers](providers.md) — learn about the Provider interface and OpenAI options
- [Embeddings](embeddings.md) — generate vector embeddings with OpenAI and Google
- [Speech](speech.md) — speech synthesis with Edge TTS and custom providers
- [Tool Calling](tools.md) — define tools and enable multi-step execution
- [Streaming](streaming.md) — understand StreamPart types and advanced patterns
- [API Reference](api-reference.md) — complete type and function reference
