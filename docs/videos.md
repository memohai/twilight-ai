# Videos

The Twilight AI SDK provides asynchronous video generation through `sdk.CreateVideo`, `sdk.GetVideo`, `sdk.CancelVideo`, `sdk.DownloadVideo`, and the convenience helper `sdk.GenerateVideo`.

Video generation jobs can take minutes, so providers are modeled as create-and-poll backends instead of streaming or single-response calls.

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"
    "time"

    openroutervideos "github.com/memohai/twilight-ai/provider/openrouter/videos"
    "github.com/memohai/twilight-ai/sdk"
)

func main() {
    provider := openroutervideos.New(
        openroutervideos.WithAPIKey("sk-or-..."),
    )
    model := provider.VideoModel("google/veo-3.1")

    result, err := sdk.GenerateVideo(context.Background(),
        sdk.WithVideoModel(model),
        sdk.WithVideoPrompt("A cinematic tracking shot of waves at sunrise"),
        sdk.WithVideoDuration(8),
        sdk.WithVideoResolution("720p"),
        sdk.WithVideoAspectRatio("16:9"),
        sdk.WithVideoPollInterval(5*time.Second),
    )
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(result.Job.ID, result.Job.Status)
    if result.Output != nil {
        fmt.Println(result.Output.URL)
    }
}
```

For manual polling:

```go
job, err := sdk.CreateVideo(ctx,
    sdk.WithVideoModel(model),
    sdk.WithVideoPrompt("A short product reveal clip"),
    sdk.WithVideoWait(false),
)
if err != nil {
    return err
}

for job.Status != sdk.VideoJobSucceeded {
    job, err = sdk.GetVideo(ctx, model, job.ID)
    if err != nil {
        return err
    }
    time.Sleep(5 * time.Second)
}

data, contentType, err := sdk.DownloadVideo(ctx, model, job.Outputs[0])
```

## Unified API

`VideoParams` covers common video fields and keeps provider-specific options in `Config`:

| Field | Purpose |
|-------|---------|
| `Prompt` | Text description |
| `Size` | Exact output dimensions, e.g. `1280x720` |
| `Resolution` | Provider resolution label, e.g. `720p` |
| `AspectRatio` | Ratio such as `16:9`, `9:16`, `1:1` |
| `DurationSeconds` | Clip duration |
| `Seed` | Deterministic generation seed when supported |
| `GenerateAudio` | Request generated audio when supported |
| `CallbackURL` | Provider webhook callback URL |
| `InputImage` | First-frame or image-to-video input |
| `InputVideo` | Video-to-video or edit input when supported |
| `ReferenceImages` | Image references |
| `ReferenceVideos` | Video references |
| `ReferenceAudio` | Audio references |
| `Config` | Provider-specific passthrough |

Statuses are normalized to:

```go
sdk.VideoJobQueued
sdk.VideoJobRunning
sdk.VideoJobSucceeded
sdk.VideoJobFailed
sdk.VideoJobCanceled
```

`sdk.GenerateVideo` waits by default with a 10 minute timeout and 5 second poll interval. Use `WithVideoWait(false)`, `WithVideoPollTimeout(...)`, and `WithVideoPollInterval(...)` to override this behavior.

## OpenRouter Videos

Package:

```go
import openroutervideos "github.com/memohai/twilight-ai/provider/openrouter/videos"
```

Default base URL:

```text
https://openrouter.ai/api
```

The provider appends `/v1/...`, so custom base URLs should follow the same convention and include `/api` when targeting OpenRouter directly.

Official docs:

- [Video Generation](https://openrouter.ai/docs/guides/overview/multimodal/video-generation)
- [Submit a video generation request](https://openrouter.ai/docs/api/api-reference/video-generation/create-videos)
- [Poll video generation status](https://openrouter.ai/docs/api/api-reference/video-generation/get-videos)
- [Download generated video content](https://openrouter.ai/docs/api/api-reference/video-generation/list-videos-content)
- [List all video generation models](https://openrouter.ai/docs/api/api-reference/video-generation/list-videos-models)

Endpoints:

| SDK operation | OpenRouter API |
|---------------|----------------|
| `CreateVideo` | `POST /v1/videos` |
| `GetVideo` | `GET /v1/videos/{jobId}` |
| `DownloadVideo` | returned `unsigned_urls` |
| `ListModels` | `GET /v1/videos/models` |

Field mapping:

| SDK field | OpenRouter field |
|-----------|------------------|
| `Prompt` | `prompt` |
| `Model.ID` | `model` |
| `DurationSeconds` | `duration` |
| `Resolution` | `resolution` |
| `AspectRatio` | `aspect_ratio` |
| `Size` | `size` |
| `Seed` | `seed` |
| `GenerateAudio` | `generate_audio` |
| `CallbackURL` | `callback_url` |
| `InputImage` | `frame_images[0]` as `first_frame` |
| references | `input_references` |
| `Config["provider"]` | `provider` passthrough object |

## Ark / ModelArk Videos

Package:

```go
import arkvideos "github.com/memohai/twilight-ai/provider/ark/videos"
```

Base URL constants:

```go
arkvideos.BytePlusBaseURL   // https://ark.ap-southeast.bytepluses.com/api/v3
arkvideos.VolcengineBaseURL // https://ark.cn-beijing.volces.com/api/v3
```

Official docs:

- [BytePlus Seedance 2.0 API Reference](https://docs.byteplus.com/en/docs/ModelArk/1520757)
- [BytePlus Video generation API](https://docs.byteplus.com/en/docs/ModelArk/Video_Generation_API)
- [火山方舟查询视频生成任务 API](https://www.volcengine.com/docs/82379/1521309)

Example:

```go
provider := arkvideos.New(
    arkvideos.WithAPIKey("ark-..."),
    arkvideos.WithBaseURL(arkvideos.VolcengineBaseURL),
)
model := provider.VideoModel("doubao-seedance-2-0-260128")
```

Endpoints:

| SDK operation | Ark / ModelArk API |
|---------------|--------------------|
| `CreateVideo` | `POST /contents/generations/tasks` |
| `GetVideo` | `GET /contents/generations/tasks/{id}` |
| `CancelVideo` | `DELETE /contents/generations/tasks/{id}` |

Field mapping:

| SDK field | Ark / ModelArk field |
|-----------|----------------------|
| `Model.ID` | `model` |
| `Prompt` | `content[0] = {type:"text", text:...}` |
| media inputs | `content[]` URL items |
| `DurationSeconds` | `duration` |
| `Resolution` | `resolution` |
| `AspectRatio` | `ratio` |
| `GenerateAudio` | `generate_audio` |
| `Seed` | `seed` |
| `CallbackURL` | `callback_url` |
| `Config` | top-level passthrough fields |

The provider reads completed outputs from `content.video_url` and other video URL fields returned by the task API. `ListModels` intentionally returns an empty list in v1 because model discovery for Ark lives in control-plane APIs rather than the data-plane task API.

## OpenAI Sora Videos

Package:

```go
import openaivideos "github.com/memohai/twilight-ai/provider/openai/videos"
```

Default base URL:

```text
https://api.openai.com/v1
```

Official docs:

- [Video generation with Sora](https://developers.openai.com/api/docs/guides/video-generation)
- [Create video API reference](https://developers.openai.com/api/reference/resources/videos/methods/create/)

Deprecated: OpenAI documents the Sora 2 video generation models and Videos API as deprecated, with shutdown scheduled for 2026-09-24.

Endpoints:

| SDK operation | OpenAI API |
|---------------|------------|
| `CreateVideo` | `POST /videos` |
| `GetVideo` | `GET /videos/{video_id}` |
| `DownloadVideo` | `GET /videos/{video_id}/content` |

Field mapping:

| SDK field | OpenAI field |
|-----------|--------------|
| `Model.ID` | `model` |
| `Prompt` | `prompt` |
| `Size` | `size` |
| `DurationSeconds` | `seconds` |
| `InputImage` with bytes | multipart `input_reference` |
| `InputImage` with URL/FileID | JSON `input_reference.image_url` / `file_id` |
| `Config["variant"]` | download query `variant` (`video`, `thumbnail`, `spritesheet`) |

Example:

```go
provider := openaivideos.New(openaivideos.WithAPIKey("sk-..."))
model := provider.VideoModel("sora-2")

job, err := sdk.CreateVideo(ctx,
    sdk.WithVideoModel(model),
    sdk.WithVideoPrompt("A teal coupe driving through desert heat haze"),
    sdk.WithVideoSize("1280x720"),
    sdk.WithVideoDuration(8),
)
```
