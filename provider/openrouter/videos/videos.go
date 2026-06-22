// Package videos provides OpenRouter video generation support.
package videos

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/memohai/twilight-ai/internal/utils"
	"github.com/memohai/twilight-ai/sdk"
)

const defaultBaseURL = "https://openrouter.ai/api"

type Provider struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

type Option func(*Provider)

func WithAPIKey(apiKey string) Option {
	return func(p *Provider) { p.apiKey = apiKey }
}

func WithBaseURL(baseURL string) Option {
	return func(p *Provider) { p.baseURL = strings.TrimRight(baseURL, "/") }
}

func WithHTTPClient(client *http.Client) Option {
	return func(p *Provider) { p.httpClient = client }
}

func New(options ...Option) *Provider {
	p := &Provider{
		baseURL:    defaultBaseURL,
		httpClient: &http.Client{},
	}
	for _, opt := range options {
		opt(p)
	}
	p.baseURL = strings.TrimRight(p.baseURL, "/")
	return p
}

func (p *Provider) VideoModel(id string) *sdk.VideoModel {
	return &sdk.VideoModel{ID: id, Provider: p}
}

func (p *Provider) ListModels(ctx context.Context) ([]*sdk.VideoModel, error) {
	resp, err := utils.FetchJSON[listModelsResponse](ctx, p.httpClient, &utils.RequestOptions{
		Method:  http.MethodGet,
		BaseURL: p.baseURL,
		Path:    "/v1/videos/models",
		Headers: utils.AuthHeader(p.apiKey),
	})
	if err != nil {
		return nil, fmt.Errorf("openrouter videos: list models request failed: %w", err)
	}
	models := make([]*sdk.VideoModel, 0, len(resp.Data))
	for _, item := range resp.Data {
		models = append(models, &sdk.VideoModel{
			ID:       item.ID,
			Provider: p,
			ProviderMetadata: map[string]any{
				"canonical_slug":                 item.CanonicalSlug,
				"name":                           item.Name,
				"description":                    item.Description,
				"allowed_passthrough_parameters": item.AllowedPassthroughParameters,
				"generate_audio":                 item.GenerateAudio,
				"seed":                           item.Seed,
				"supported_aspect_ratios":        item.SupportedAspectRatios,
				"supported_durations":            item.SupportedDurations,
				"supported_frame_images":         item.SupportedFrameImages,
				"supported_resolutions":          item.SupportedResolutions,
				"supported_sizes":                item.SupportedSizes,
				"pricing_skus":                   item.PricingSKUs,
			},
		})
	}
	return models, nil
}

func (p *Provider) DoCreate(ctx context.Context, params sdk.VideoParams) (*sdk.VideoJob, error) {
	if params.Model == nil {
		return nil, fmt.Errorf("openrouter videos: model is required")
	}
	req := createRequest{
		Model:         params.Model.ID,
		Prompt:        params.Prompt,
		AspectRatio:   params.AspectRatio,
		CallbackURL:   params.CallbackURL,
		Duration:      params.DurationSeconds,
		GenerateAudio: params.GenerateAudio,
		Resolution:    params.Resolution,
		Seed:          params.Seed,
		Size:          params.Size,
	}
	if params.InputImage != nil {
		if url := mediaURL(*params.InputImage); url != "" {
			req.FrameImages = append(req.FrameImages, frameImage{
				Type:      "image_url",
				FrameType: "first_frame",
				ImageURL:  mediaURLObject{URL: url},
			})
		}
	}
	for _, input := range params.ReferenceImages {
		if url := mediaURL(input); url != "" {
			req.InputReferences = append(req.InputReferences, inputReference{Type: "image_url", ImageURL: &mediaURLObject{URL: url}})
		}
	}
	for _, input := range params.ReferenceAudio {
		if url := mediaURL(input); url != "" {
			req.InputReferences = append(req.InputReferences, inputReference{Type: "audio_url", AudioURL: &mediaURLObject{URL: url}})
		}
	}
	for _, input := range params.ReferenceVideos {
		if url := mediaURL(input); url != "" {
			req.InputReferences = append(req.InputReferences, inputReference{Type: "video_url", VideoURL: &mediaURLObject{URL: url}})
		}
	}
	if providerOptions, ok := params.Config["provider"]; ok {
		req.Provider = providerOptions
	}

	resp, err := utils.FetchJSON[videoResponse](ctx, p.httpClient, &utils.RequestOptions{
		Method:  http.MethodPost,
		BaseURL: p.baseURL,
		Path:    "/v1/videos",
		Headers: utils.AuthHeader(p.apiKey),
		Body:    req,
	})
	if err != nil {
		return nil, fmt.Errorf("openrouter videos: create request failed: %w", err)
	}
	return toVideoJob(resp, params.Model.ID), nil
}

func (p *Provider) DoGet(ctx context.Context, model *sdk.VideoModel, id string) (*sdk.VideoJob, error) {
	resp, err := utils.FetchJSON[videoResponse](ctx, p.httpClient, &utils.RequestOptions{
		Method:  http.MethodGet,
		BaseURL: p.baseURL,
		Path:    "/v1/videos/" + id,
		Headers: utils.AuthHeader(p.apiKey),
	})
	if err != nil {
		return nil, fmt.Errorf("openrouter videos: get request failed: %w", err)
	}
	modelID := ""
	if model != nil {
		modelID = model.ID
	}
	return toVideoJob(resp, modelID), nil
}

func (p *Provider) DoCancel(_ context.Context, _ *sdk.VideoModel, _ string) error {
	return fmt.Errorf("openrouter videos: cancel is not supported")
}

func (p *Provider) DoDownload(ctx context.Context, _ *sdk.VideoModel, output sdk.VideoOutput) ([]byte, string, error) {
	if output.URL == "" {
		return nil, "", fmt.Errorf("openrouter videos: output URL is required")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, output.URL, http.NoBody)
	if err != nil {
		return nil, "", fmt.Errorf("openrouter videos: build download request: %w", err)
	}
	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, "", fmt.Errorf("openrouter videos: download request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		return nil, "", fmt.Errorf("openrouter videos: download failed with status %d: %s", resp.StatusCode, string(body))
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, "", fmt.Errorf("openrouter videos: read download response: %w", err)
	}
	contentType := resp.Header.Get("Content-Type")
	if contentType == "" {
		contentType = output.ContentType
	}
	if contentType == "" {
		contentType = "video/mp4"
	}
	return data, contentType, nil
}

func toVideoJob(resp *videoResponse, modelID string) *sdk.VideoJob {
	job := &sdk.VideoJob{
		ID:      resp.ID,
		ModelID: modelID,
		Status:  mapStatus(resp.Status),
		ProviderMetadata: map[string]any{
			"polling_url":   resp.PollingURL,
			"generation_id": resp.GenerationID,
			"usage":         resp.Usage,
		},
	}
	if resp.Error != "" {
		job.Error = &sdk.VideoError{Message: resp.Error}
	}
	for _, url := range resp.UnsignedURLs {
		if strings.TrimSpace(url) == "" {
			continue
		}
		job.Outputs = append(job.Outputs, sdk.VideoOutput{
			URL:         url,
			ContentType: "video/mp4",
			ProviderMetadata: map[string]any{
				"generation_id": resp.GenerationID,
			},
		})
	}
	return job
}

func mapStatus(status string) sdk.VideoJobStatus {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "pending", "queued":
		return sdk.VideoJobQueued
	case "running", "in_progress", "processing":
		return sdk.VideoJobRunning
	case "completed", "succeeded", "success":
		return sdk.VideoJobSucceeded
	case "failed", "error":
		return sdk.VideoJobFailed
	case "canceled", "cancelled":
		return sdk.VideoJobCanceled
	default:
		return sdk.VideoJobRunning
	}
}

func mediaURL(input sdk.MediaInput) string {
	if input.URL != "" {
		return input.URL
	}
	if len(input.Data) == 0 {
		return input.FileID
	}
	contentType := input.ContentType
	if contentType == "" {
		contentType = "application/octet-stream"
	}
	return "data:" + contentType + ";base64," + base64.StdEncoding.EncodeToString(input.Data)
}
