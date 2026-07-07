# Images

The Twilight AI SDK provides image generation and editing through the `sdk.GenerateImage` and `sdk.EditImage` APIs. These are provider-agnostic functions backed by the `ImageGenerationProvider` and `ImageEditProvider` interfaces.

## Quick Start

### Generate an Image

```go
import (
    "context"
    "fmt"
    "log"

    "github.com/memohai/twilight-ai/provider/openai/images"
    "github.com/memohai/twilight-ai/sdk"
)

func main() {
    provider := images.New(images.WithAPIKey("sk-..."))
    model := provider.GenerationModel("gpt-image-1")

    result, err := sdk.GenerateImage(context.Background(),
        sdk.WithImageGenerationModel(model),
        sdk.WithImagePrompt("A futuristic city skyline at dusk, cyberpunk style"),
        sdk.WithImageSize("1024x1024"),
    )
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Generated %d image(s)\n", len(result.Data))
    // result.Data[0].B64JSON contains the base64-encoded image
}
```

### Edit an Image

```go
provider := images.New(images.WithAPIKey("sk-..."))
model := provider.EditModel("gpt-image-1")

result, err := sdk.EditImage(context.Background(),
    sdk.WithImageEditModel(model),
    sdk.WithEditPrompt("Replace the sky with a starry night"),
    sdk.WithEditImages(sdk.ImageInput{
        Data:     originalPNG,
        Filename: "photo.png",
    }),
    sdk.WithEditMask(&sdk.ImageInput{
        Data:     maskPNG,
        Filename: "mask.png",
    }),
)
```

## Providers

### OpenAI Images

The `provider/openai/images` package supports the full OpenAI Images API.

```go
provider := images.New(
    images.WithAPIKey("sk-..."),
    images.WithBaseURL("https://api.openai.com/v1"), // default
)
```

Two factory methods create models:

- `provider.GenerationModel(id)` — returns an `*sdk.ImageGenerationModel`
- `provider.EditModel(id)` — returns an `*sdk.ImageEditModel`

### Supported Models

| Model | Generation | Editing | Key Features |
|-------|-----------|---------|-------------|
| `dall-e-2` | Yes | Yes | Legacy; small sizes (256/512/1024); up to 10 images |
| `dall-e-3` | Yes | No | `n=1` only; `style` (vivid/natural); `revised_prompt` |
| `gpt-image-1` | Yes | Yes | `background`, `output_format`, `moderation`; base64 output |
| `gpt-image-1-mini` | Yes | Yes | Smaller GPT Image variant |
| `gpt-image-1.5` | Yes | Yes | Latest GPT Image; `input_fidelity` for edits |

### OpenAI-Compatible Endpoints

Any service implementing the OpenAI Images API works:

```go
provider := images.New(
    images.WithAPIKey("your-key"),
    images.WithBaseURL("https://your-api.com/v1"),
)
```

### Alibaba Cloud DashScope Images

The `provider/alibabacloud/images` package supports Alibaba Cloud Model Studio (DashScope) image models — Qwen-Image, Wan, Z-Image, and third-party models such as FLUX and Stable Diffusion.

```go
import "github.com/memohai/twilight-ai/provider/alibabacloud/images"

provider := images.New(
    images.WithAPIKey("sk-..."), // DashScope / Model Studio API key
)

model := provider.GenerationModel("qwen-image-max")
result, err := sdk.GenerateImage(ctx,
    sdk.WithImageGenerationModel(model),
    sdk.WithImagePrompt("A sunset over mountains"),
    sdk.WithImageSize("1024x1024"),
)
// result.Data[0].URL contains the generated image URL (valid for 24 hours)
```

The provider routes each model to the correct DashScope endpoint automatically: `qwen-image` / `qwen-image-plus`, legacy Wan models (`wanx-*`, `wan2.0`–`wan2.5`), and third-party models (`flux-*`, `stable-diffusion-*`) use the async text2image task API with polling; newer Wan models use the async image-generation task API; the remaining Qwen-Image models (`qwen-image-max`, `qwen-image-2.0*`, dated snapshots) and `z-image*` use the synchronous multimodal API. Polling behavior is configurable via `images.WithPollInterval` and `images.WithPollTimeout`.

DashScope models honor the `Prompt`, `N`, and `Size` options; the other generation options are OpenAI-specific and ignored. Image editing (`sdk.EditImage`) is not supported by this provider.

## Generation Options

All generation options are of type `ImageGenerateOption`:

| Function | Description |
|----------|-------------|
| `WithImageGenerationModel(model)` | **Required.** The image generation model |
| `WithImagePrompt(prompt)` | **Required.** Text description of the desired image |
| `WithImageN(n)` | Number of images to generate (1-10) |
| `WithImageSize(size)` | Image dimensions (e.g. `"1024x1024"`, `"1536x1024"`) |
| `WithImageQuality(quality)` | Quality level: `"auto"`, `"low"`, `"medium"`, `"high"`, `"standard"`, `"hd"` |
| `WithImageStyle(style)` | dall-e-3 only: `"vivid"` or `"natural"` |
| `WithImageResponseFormat(format)` | dall-e-2/3: `"url"` or `"b64_json"` |
| `WithImageBackground(background)` | GPT Image: `"transparent"`, `"opaque"`, `"auto"` |
| `WithImageOutputFormat(format)` | GPT Image: `"png"`, `"jpeg"`, `"webp"` |
| `WithImageOutputCompression(n)` | GPT Image, jpeg/webp: compression 0-100 |
| `WithImageModeration(moderation)` | GPT Image: `"low"` or `"auto"` |
| `WithImageUser(user)` | End-user identifier for abuse monitoring |

## Edit Options

All edit options are of type `ImageEditOption`:

| Function | Description |
|----------|-------------|
| `WithImageEditModel(model)` | **Required.** The image edit model |
| `WithEditPrompt(prompt)` | **Required.** Description of the edit |
| `WithEditImages(images...)` | Source images (up to 16 for GPT Image) |
| `WithEditMask(mask)` | Mask image (transparent regions = edit area) |
| `WithEditN(n)` | Number of images to generate |
| `WithEditSize(size)` | Output size |
| `WithEditQuality(quality)` | Quality level |
| `WithEditBackground(background)` | Background transparency |
| `WithEditOutputFormat(format)` | Output format |
| `WithEditOutputCompression(n)` | Compression level |
| `WithEditInputFidelity(fidelity)` | GPT Image: `"high"` or `"low"` |
| `WithEditModeration(moderation)` | Moderation level |
| `WithEditResponseFormat(format)` | dall-e-2: `"url"` or `"b64_json"` |
| `WithEditUser(user)` | End-user identifier |

## Image Input

The `ImageInput` struct represents an image provided as input to edit requests. Exactly one of `Data`, `URL`, or `FileID` should be set:

```go
// File upload (sent as multipart/form-data)
sdk.ImageInput{
    Data:     pngBytes,     // raw file bytes
    MediaType: "image/png",
    Filename: "photo.png",
}

// URL reference (sent as JSON)
sdk.ImageInput{
    URL: "https://example.com/image.png",
}

// File ID reference (sent as JSON)
sdk.ImageInput{
    FileID: "file-abc123",
}
```

The provider automatically selects `multipart/form-data` when `Data` bytes are present, or JSON when using `URL`/`FileID` references.

## Result

Both `GenerateImage` and `EditImage` return an `*sdk.ImageResult`:

```go
type ImageResult struct {
    Created int64        // Unix timestamp
    Data    []ImageData  // generated images
    Usage   ImageUsage   // token usage (GPT Image models)
}

type ImageData struct {
    B64JSON       string // base64-encoded image data
    URL           string // temporary URL (dall-e-2/3 with response_format=url)
    RevisedPrompt string // dall-e-3 only: model's revised prompt
}

type ImageUsage struct {
    TotalTokens       int
    InputTokens       int
    OutputTokens      int
    InputTokenDetails *ImageInputTokenDetails
}
```

## Next Steps

- [Providers](providers.md) — all provider details and options
- [Embeddings](embeddings.md) — generate vector embeddings
- [Speech](speech.md) — speech synthesis
- [API Reference](api-reference.md) — complete type reference
