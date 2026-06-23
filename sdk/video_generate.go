package sdk

import (
	"context"
	"fmt"
	"time"
)

const (
	defaultVideoPollInterval = 5 * time.Second
	defaultVideoPollTimeout  = 10 * time.Minute
)

type videoConfig struct {
	Params       VideoParams
	Wait         bool
	PollInterval time.Duration
	PollTimeout  time.Duration
	Download     bool
}

// VideoOption configures a video generation request.
type VideoOption func(*videoConfig)

func WithVideoModel(model *VideoModel) VideoOption {
	return func(c *videoConfig) { c.Params.Model = model }
}

func WithVideoPrompt(prompt string) VideoOption {
	return func(c *videoConfig) { c.Params.Prompt = prompt }
}

func WithVideoSize(size string) VideoOption {
	return func(c *videoConfig) { c.Params.Size = size }
}

func WithVideoResolution(resolution string) VideoOption {
	return func(c *videoConfig) { c.Params.Resolution = resolution }
}

func WithVideoAspectRatio(aspectRatio string) VideoOption {
	return func(c *videoConfig) { c.Params.AspectRatio = aspectRatio }
}

func WithVideoDuration(seconds int) VideoOption {
	return func(c *videoConfig) { c.Params.DurationSeconds = &seconds }
}

func WithVideoSeed(seed int64) VideoOption {
	return func(c *videoConfig) { c.Params.Seed = &seed }
}

func WithVideoGenerateAudio(generate bool) VideoOption {
	return func(c *videoConfig) { c.Params.GenerateAudio = &generate }
}

func WithVideoCallbackURL(url string) VideoOption {
	return func(c *videoConfig) { c.Params.CallbackURL = url }
}

func WithVideoInputImage(input *MediaInput) VideoOption {
	return func(c *videoConfig) { c.Params.InputImage = input }
}

func WithVideoInputVideo(input *MediaInput) VideoOption {
	return func(c *videoConfig) { c.Params.InputVideo = input }
}

func WithVideoReferenceImages(inputs ...MediaInput) VideoOption {
	return func(c *videoConfig) { c.Params.ReferenceImages = inputs }
}

func WithVideoReferenceVideos(inputs ...MediaInput) VideoOption {
	return func(c *videoConfig) { c.Params.ReferenceVideos = inputs }
}

func WithVideoReferenceAudio(inputs ...MediaInput) VideoOption {
	return func(c *videoConfig) { c.Params.ReferenceAudio = inputs }
}

func WithVideoConfig(config map[string]any) VideoOption {
	return func(c *videoConfig) { c.Params.Config = config }
}

func WithVideoWait(wait bool) VideoOption {
	return func(c *videoConfig) { c.Wait = wait }
}

func WithVideoPollInterval(interval time.Duration) VideoOption {
	return func(c *videoConfig) { c.PollInterval = interval }
}

func WithVideoPollTimeout(timeout time.Duration) VideoOption {
	return func(c *videoConfig) { c.PollTimeout = timeout }
}

func WithVideoDownload(download bool) VideoOption {
	return func(c *videoConfig) { c.Download = download }
}

func buildVideoConfig(options []VideoOption) (*videoConfig, VideoProvider, error) {
	cfg := &videoConfig{
		Wait:         true,
		PollInterval: defaultVideoPollInterval,
		PollTimeout:  defaultVideoPollTimeout,
	}
	for _, opt := range options {
		opt(cfg)
	}
	if cfg.Params.Model == nil {
		return nil, nil, fmt.Errorf("twilightai: video model is required (use WithVideoModel)")
	}
	if cfg.Params.Model.Provider == nil {
		return nil, nil, fmt.Errorf("twilightai: video model %q has no provider", cfg.Params.Model.ID)
	}
	if cfg.Params.Prompt == "" {
		return nil, nil, fmt.Errorf("twilightai: prompt is required (use WithVideoPrompt)")
	}
	if cfg.PollInterval <= 0 {
		return nil, nil, fmt.Errorf("twilightai: video poll interval must be positive")
	}
	if cfg.PollTimeout <= 0 {
		return nil, nil, fmt.Errorf("twilightai: video poll timeout must be positive")
	}
	return cfg, cfg.Params.Model.Provider, nil
}

// CreateVideo starts an asynchronous video generation job.
func (c *Client) CreateVideo(ctx context.Context, options ...VideoOption) (*VideoJob, error) {
	cfg, prov, err := buildVideoConfig(options)
	if err != nil {
		return nil, err
	}
	return prov.DoCreate(ctx, cfg.Params)
}

// GetVideo retrieves a video generation job by ID.
func (c *Client) GetVideo(ctx context.Context, model *VideoModel, id string) (*VideoJob, error) {
	prov, err := videoProviderFromModel(model)
	if err != nil {
		return nil, err
	}
	if id == "" {
		return nil, fmt.Errorf("twilightai: video id is required")
	}
	return prov.DoGet(ctx, model, id)
}

// CancelVideo requests cancellation of a video generation job.
func (c *Client) CancelVideo(ctx context.Context, model *VideoModel, id string) error {
	prov, err := videoProviderFromModel(model)
	if err != nil {
		return err
	}
	if id == "" {
		return fmt.Errorf("twilightai: video id is required")
	}
	return prov.DoCancel(ctx, model, id)
}

// DownloadVideo downloads a provider output and returns bytes plus content type.
func (c *Client) DownloadVideo(ctx context.Context, model *VideoModel, output VideoOutput) (data []byte, contentType string, err error) {
	prov, err := videoProviderFromModel(model)
	if err != nil {
		return nil, "", err
	}
	return prov.DoDownload(ctx, model, output)
}

// GenerateVideo starts a job, waits for it by default, and optionally downloads
// the first output when WithVideoDownload(true) is set.
func (c *Client) GenerateVideo(ctx context.Context, options ...VideoOption) (*VideoResult, error) {
	cfg, prov, err := buildVideoConfig(options)
	if err != nil {
		return nil, err
	}

	job, err := prov.DoCreate(ctx, cfg.Params)
	if err != nil {
		return nil, err
	}
	result := &VideoResult{Job: job}
	if !cfg.Wait {
		return result, nil
	}

	waitCtx, cancel := context.WithTimeout(ctx, cfg.PollTimeout)
	defer cancel()

	ticker := time.NewTicker(cfg.PollInterval)
	defer ticker.Stop()

	for job == nil || !job.Status.Terminal() {
		select {
		case <-waitCtx.Done():
			return nil, fmt.Errorf("twilightai: video generation timed out after %s", cfg.PollTimeout)
		case <-ticker.C:
			if job == nil || job.ID == "" {
				return nil, fmt.Errorf("twilightai: video provider returned empty job id")
			}
			job, err = prov.DoGet(waitCtx, cfg.Params.Model, job.ID)
			if err != nil {
				return nil, err
			}
			result.Job = job
		}
	}

	if job.Status != VideoJobSucceeded {
		if job.Error != nil && job.Error.Message != "" {
			return result, fmt.Errorf("twilightai: video generation failed: %s", job.Error.Message)
		}
		return result, fmt.Errorf("twilightai: video generation finished with status %s", job.Status)
	}
	if len(job.Outputs) > 0 {
		result.Output = &job.Outputs[0]
	}
	if cfg.Download && result.Output != nil {
		mergeVideoOutputMetadata(result.Output, cfg.Params.Config)
		data, contentType, err := prov.DoDownload(ctx, cfg.Params.Model, *result.Output)
		if err != nil {
			return result, err
		}
		result.Data = data
		result.ContentType = contentType
	}
	return result, nil
}

func videoProviderFromModel(model *VideoModel) (VideoProvider, error) {
	if model == nil {
		return nil, fmt.Errorf("twilightai: video model is required")
	}
	if model.Provider == nil {
		return nil, fmt.Errorf("twilightai: video model %q has no provider", model.ID)
	}
	return model.Provider, nil
}

func mergeVideoOutputMetadata(output *VideoOutput, config map[string]any) {
	if output == nil || len(config) == 0 {
		return
	}
	if output.ProviderMetadata == nil {
		output.ProviderMetadata = map[string]any{}
	}
	for k, v := range config {
		if _, exists := output.ProviderMetadata[k]; !exists {
			output.ProviderMetadata[k] = v
		}
	}
}
