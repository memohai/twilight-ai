# Speech Synthesis

Speech synthesis converts text into audio. Twilight AI provides a unified speech API with an open-ended configuration model that lets each provider define its own parameters.

## Core Concepts

| Concept | Description |
|---------|-------------|
| `SpeechProvider` | Interface that speech backends implement |
| `SpeechModel` | A model bound to a provider, created via `provider.SpeechModel(id)` |
| `GenerateSpeech` | Synthesize complete audio in one call |
| `StreamSpeech` | Synthesize audio as a stream of chunks |

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"

    "github.com/memohai/twilight-ai/provider/edge/speech"
    "github.com/memohai/twilight-ai/sdk"
)

func main() {
    provider := speech.New()
    model := provider.SpeechModel("edge-read-aloud")

    result, err := sdk.GenerateSpeech(context.Background(),
        sdk.WithSpeechModel(model),
        sdk.WithText("Hello, this is a speech synthesis test."),
        sdk.WithSpeechConfig(map[string]any{
            "voice": "en-US-EmmaMultilingualNeural",
            "speed": 1.0,
        }),
    )
    if err != nil {
        log.Fatal(err)
    }

    if err := os.WriteFile("output.mp3", result.Audio, 0644); err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Saved %d bytes (%s)\n", len(result.Audio), result.ContentType)
}
```

## Generate Speech

`sdk.GenerateSpeech` synthesizes the entire audio and returns it as a byte slice:

```go
result, err := sdk.GenerateSpeech(ctx,
    sdk.WithSpeechModel(model),
    sdk.WithText("The quick brown fox jumps over the lazy dog."),
    sdk.WithSpeechConfig(map[string]any{
        "voice":  "en-US-JennyNeural",
        "format": "audio-24khz-96kbitrate-mono-mp3",
    }),
)
// result.Audio       — raw audio bytes
// result.ContentType — MIME type, e.g. "audio/mpeg"
```

## Stream Speech

`sdk.StreamSpeech` returns audio chunks via a channel for low-latency playback:

```go
sr, err := sdk.StreamSpeech(ctx,
    sdk.WithSpeechModel(model),
    sdk.WithText("你好，这是流式语音合成测试。"),
    sdk.WithSpeechConfig(map[string]any{
        "voice": "zh-CN-XiaoxiaoNeural",
    }),
)
if err != nil {
    log.Fatal(err)
}

for chunk := range sr.Stream {
    // Write each chunk to an audio player, file, or HTTP response
    writer.Write(chunk)
}
```

Or use the convenience method to collect all audio:

```go
sr, _ := sdk.StreamSpeech(ctx, ...)
audio, err := sr.Bytes()
```

## Options

| Option | Description |
|--------|-------------|
| `sdk.WithSpeechModel(model)` | **Required.** The speech model to use |
| `sdk.WithText(text)` | **Required.** The text to synthesize |
| `sdk.WithSpeechConfig(cfg)` | Provider-specific configuration map |

## Using a Client Instance

The package-level `sdk.GenerateSpeech` and `sdk.StreamSpeech` use a default client. You can also create your own:

```go
client := sdk.NewClient()
result, err := client.GenerateSpeech(ctx,
    sdk.WithSpeechModel(model),
    sdk.WithText("Hello"),
)
```

## Edge TTS Provider

The `provider/edge/speech` package provides free speech synthesis via Microsoft Edge's built-in TTS service. No API key is required.

### Basic Usage

```go
import "github.com/memohai/twilight-ai/provider/edge/speech"

provider := speech.New()
model := provider.SpeechModel("edge-read-aloud")
```

### Configuration Keys

The Edge provider reads these keys from `WithSpeechConfig`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `voice` | `string` | `en-US-EmmaMultilingualNeural` | Voice ID from the Edge TTS voice catalog |
| `language` | `string` | Auto-detected from voice | BCP-47 language tag (e.g. `en-US`, `zh-CN`) |
| `format` | `string` | `audio-24khz-48kbitrate-mono-mp3` | Output audio format string |
| `speed` | `float64` | `0` (server default) | Speech rate; `1.0` = normal, `2.0` = double speed |
| `pitch` | `float64` | `0` | Pitch adjustment in Hz |

### Available Formats

| Format | Content Type |
|--------|-------------|
| `audio-24khz-48kbitrate-mono-mp3` | `audio/mpeg` |
| `audio-24khz-96kbitrate-mono-mp3` | `audio/mpeg` |
| `webm-24khz-16bit-mono-opus` | `audio/opus` |

### Voices

Edge TTS supports 400+ voices across 100+ languages. Use `speech.EdgeTTSVoices` to browse the full catalog:

```go
import "github.com/memohai/twilight-ai/provider/edge/speech"

// Map of language tag → voice IDs
for lang, voices := range speech.EdgeTTSVoices {
    fmt.Printf("%s: %v\n", lang, voices)
}

// Reverse lookup: voice ID → language
lang, ok := speech.LookupVoiceLang("zh-CN-XiaoxiaoNeural")
// lang = "zh-CN", ok = true
```

Popular voices:

| Language | Voice ID |
|----------|----------|
| English (US) | `en-US-EmmaMultilingualNeural`, `en-US-JennyNeural`, `en-US-AriaNeural` |
| English (GB) | `en-GB-SoniaNeural`, `en-GB-RyanNeural` |
| Chinese (CN) | `zh-CN-XiaoxiaoNeural`, `zh-CN-YunxiNeural` |
| Japanese | `ja-JP-NanamiNeural`, `ja-JP-KeitaNeural` |
| Korean | `ko-KR-SunHiNeural`, `ko-KR-InJoonNeural` |

### Provider Options

| Option | Description |
|--------|-------------|
| `speech.WithBaseURL(url)` | Override the WebSocket endpoint (for testing) |

## Implementing a Custom Speech Provider

To add support for a new speech backend, implement the `sdk.SpeechProvider` interface:

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

func (p *MyProvider) SpeechModel(id string) *sdk.SpeechModel {
    return &sdk.SpeechModel{ID: id, Provider: p}
}

func (p *MyProvider) DoSynthesize(ctx context.Context, params sdk.SpeechParams) (*sdk.SpeechResult, error) {
    // Read provider-specific config from params.Config
    voice, _ := params.Config["voice"].(string)
    format, _ := params.Config["response_format"].(string)

    // Call your TTS API...
    audio := synthesizeAudio(params.Text, voice, format)

    return &sdk.SpeechResult{
        Audio:       audio,
        ContentType: "audio/mpeg",
    }, nil
}

func (p *MyProvider) DoStream(ctx context.Context, params sdk.SpeechParams) (*sdk.SpeechStreamResult, error) {
    ch := make(chan []byte, 8)
    errCh := make(chan error, 1)

    go func() {
        defer close(ch)
        defer close(errCh)
        // Stream audio chunks into ch...
    }()

    return sdk.NewSpeechStreamResult(ch, "audio/mpeg", errCh), nil
}
```

Then use it like the built-in provider:

```go
provider := myprovider.New("my-api-key")
model := provider.SpeechModel("tts-1")

result, err := sdk.GenerateSpeech(ctx,
    sdk.WithSpeechModel(model),
    sdk.WithText("Hello"),
    sdk.WithSpeechConfig(map[string]any{
        "voice":           "alloy",
        "response_format": "mp3",
    }),
)
```

Each provider defines its own configuration keys in `Config`. This open-ended design allows providers with very different parameter sets (Edge TTS pitch/format vs. OpenAI voice/response_format vs. others) to coexist under the same interface.

## Common Patterns

### Save to File

```go
result, _ := sdk.GenerateSpeech(ctx,
    sdk.WithSpeechModel(model),
    sdk.WithText("Save this to a file."),
)
os.WriteFile("output.mp3", result.Audio, 0644)
```

### Stream to HTTP Response

```go
func handleTTS(w http.ResponseWriter, r *http.Request) {
    sr, err := sdk.StreamSpeech(r.Context(),
        sdk.WithSpeechModel(model),
        sdk.WithText(r.FormValue("text")),
    )
    if err != nil {
        http.Error(w, err.Error(), 500)
        return
    }
    w.Header().Set("Content-Type", sr.ContentType)
    for chunk := range sr.Stream {
        w.Write(chunk)
        if f, ok := w.(http.Flusher); ok {
            f.Flush()
        }
    }
}
```

## Next Steps

- [Providers](providers.md) — learn about chat and embedding providers
- [Streaming](streaming.md) — understand channel-based streaming for text
- [API Reference](api-reference.md) — complete type and function reference
