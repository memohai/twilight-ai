package sdk

import "context"

// VideoProvider is the interface that asynchronous video generation backends
// must implement.
type VideoProvider interface {
	ListModels(ctx context.Context) ([]*VideoModel, error)
	DoCreate(ctx context.Context, params VideoParams) (*VideoJob, error)
	DoGet(ctx context.Context, model *VideoModel, id string) (*VideoJob, error)
	DoCancel(ctx context.Context, model *VideoModel, id string) error
	DoDownload(ctx context.Context, model *VideoModel, output VideoOutput) ([]byte, string, error)
}

// VideoModel represents a video generation model bound to a VideoProvider.
type VideoModel struct {
	ID               string
	Provider         VideoProvider
	ProviderMetadata map[string]any
}

// MediaInput represents an input media asset for video generation.
// Exactly one of Data, URL, or FileID should usually be set.
type MediaInput struct {
	Data        []byte
	URL         string
	FileID      string
	Filename    string
	ContentType string
}

// VideoParams holds provider-agnostic video generation parameters.
// Config is an escape hatch for provider-specific options.
type VideoParams struct {
	Model  *VideoModel
	Prompt string

	Size            string
	Resolution      string
	AspectRatio     string
	DurationSeconds *int
	Seed            *int64
	GenerateAudio   *bool
	CallbackURL     string

	InputImage      *MediaInput
	InputVideo      *MediaInput
	ReferenceImages []MediaInput
	ReferenceVideos []MediaInput
	ReferenceAudio  []MediaInput

	Config map[string]any
}

// VideoJobStatus is the unified status for asynchronous video generation jobs.
type VideoJobStatus string

const (
	VideoJobQueued    VideoJobStatus = "queued"
	VideoJobRunning   VideoJobStatus = "running"
	VideoJobSucceeded VideoJobStatus = "succeeded"
	VideoJobFailed    VideoJobStatus = "failed"
	VideoJobCanceled  VideoJobStatus = "canceled"
)

// Terminal reports whether the job status cannot make more progress.
func (s VideoJobStatus) Terminal() bool {
	switch s {
	case VideoJobSucceeded, VideoJobFailed, VideoJobCanceled:
		return true
	default:
		return false
	}
}

// VideoJob holds the unified state returned by a video generation provider.
type VideoJob struct {
	ID               string
	ModelID          string
	Status           VideoJobStatus
	Progress         *float64
	Outputs          []VideoOutput
	Error            *VideoError
	ProviderMetadata map[string]any
}

// VideoOutput describes a generated video or related downloadable asset.
type VideoOutput struct {
	URL              string
	ContentType      string
	Width            int
	Height           int
	DurationSeconds  float64
	HasAudio         bool
	ProviderMetadata map[string]any
}

// VideoError is a provider-normalized error payload for failed jobs.
type VideoError struct {
	Code    string
	Message string
}

// VideoResult is returned by GenerateVideo. Data is populated only when
// WithVideoDownload(true) is used.
type VideoResult struct {
	Job         *VideoJob
	Output      *VideoOutput
	Data        []byte
	ContentType string
}
