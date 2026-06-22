// Package videos provides Ark / ModelArk video generation support for
// BytePlus ModelArk and Volcengine Ark data-plane APIs.
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

const (
	BytePlusBaseURL   = "https://ark.ap-southeast.bytepluses.com/api/v3"
	VolcengineBaseURL = "https://ark.cn-beijing.volces.com/api/v3"
)

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
	p := &Provider{httpClient: &http.Client{}}
	for _, opt := range options {
		opt(p)
	}
	p.baseURL = strings.TrimRight(p.baseURL, "/")
	return p
}

func (p *Provider) VideoModel(id string) *sdk.VideoModel {
	return &sdk.VideoModel{ID: id, Provider: p}
}

func (p *Provider) ListModels(context.Context) ([]*sdk.VideoModel, error) {
	return []*sdk.VideoModel{}, nil
}

func (p *Provider) DoCreate(ctx context.Context, params sdk.VideoParams) (*sdk.VideoJob, error) {
	if params.Model == nil {
		return nil, fmt.Errorf("ark videos: model is required")
	}
	body := p.buildCreateBody(params)
	resp, err := utils.FetchJSON[map[string]any](ctx, p.httpClient, &utils.RequestOptions{
		Method:  http.MethodPost,
		BaseURL: p.baseURL,
		Path:    "/contents/generations/tasks",
		Headers: utils.AuthHeader(p.apiKey),
		Body:    body,
	})
	if err != nil {
		return nil, fmt.Errorf("ark videos: create request failed: %w", err)
	}
	return toVideoJob(*resp, params.Model.ID), nil
}

func (p *Provider) DoGet(ctx context.Context, model *sdk.VideoModel, id string) (*sdk.VideoJob, error) {
	resp, err := utils.FetchJSON[map[string]any](ctx, p.httpClient, &utils.RequestOptions{
		Method:  http.MethodGet,
		BaseURL: p.baseURL,
		Path:    "/contents/generations/tasks/" + id,
		Headers: utils.AuthHeader(p.apiKey),
	})
	if err != nil {
		return nil, fmt.Errorf("ark videos: get request failed: %w", err)
	}
	modelID := ""
	if model != nil {
		modelID = model.ID
	}
	return toVideoJob(*resp, modelID), nil
}

func (p *Provider) DoCancel(ctx context.Context, _ *sdk.VideoModel, id string) error {
	resp, err := utils.FetchRaw(ctx, p.httpClient, &utils.RequestOptions{
		Method:  http.MethodDelete,
		BaseURL: p.baseURL,
		Path:    "/contents/generations/tasks/" + id,
		Headers: utils.AuthHeader(p.apiKey),
	})
	if err != nil {
		return fmt.Errorf("ark videos: cancel/delete request failed: %w", err)
	}
	_ = resp.Body.Close()
	return nil
}

func (p *Provider) DoDownload(ctx context.Context, _ *sdk.VideoModel, output sdk.VideoOutput) ([]byte, string, error) {
	if output.URL == "" {
		return nil, "", fmt.Errorf("ark videos: output URL is required")
	}
	url := output.URL
	if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
		built, err := utils.BuildURL(p.baseURL, url)
		if err != nil {
			return nil, "", fmt.Errorf("ark videos: build download URL: %w", err)
		}
		url = built
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, http.NoBody)
	if err != nil {
		return nil, "", fmt.Errorf("ark videos: build download request: %w", err)
	}
	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, "", fmt.Errorf("ark videos: download request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		return nil, "", fmt.Errorf("ark videos: download failed with status %d: %s", resp.StatusCode, string(body))
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, "", fmt.Errorf("ark videos: read download response: %w", err)
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

func (p *Provider) buildCreateBody(params sdk.VideoParams) map[string]any {
	body := map[string]any{
		"model":   params.Model.ID,
		"content": buildContent(params),
	}
	if params.DurationSeconds != nil {
		body["duration"] = *params.DurationSeconds
	}
	if params.Resolution != "" {
		body["resolution"] = params.Resolution
	}
	if params.AspectRatio != "" {
		body["ratio"] = params.AspectRatio
	}
	if params.GenerateAudio != nil {
		body["generate_audio"] = *params.GenerateAudio
	}
	if params.Seed != nil {
		body["seed"] = *params.Seed
	}
	if params.CallbackURL != "" {
		body["callback_url"] = params.CallbackURL
	}
	for k, v := range params.Config {
		body[k] = v
	}
	return body
}

func buildContent(params sdk.VideoParams) []map[string]any {
	content := []map[string]any{{"type": "text", "text": params.Prompt}}
	if params.InputImage != nil {
		if url := mediaURL(*params.InputImage); url != "" {
			content = append(content, mediaItem("image_url", "image_url", url, "first_frame"))
		}
	}
	if params.InputVideo != nil {
		if url := mediaURL(*params.InputVideo); url != "" {
			content = append(content, mediaItem("video_url", "video_url", url, "input_video"))
		}
	}
	for _, input := range params.ReferenceImages {
		if url := mediaURL(input); url != "" {
			content = append(content, mediaItem("image_url", "image_url", url, "reference_image"))
		}
	}
	for _, input := range params.ReferenceVideos {
		if url := mediaURL(input); url != "" {
			content = append(content, mediaItem("video_url", "video_url", url, "reference_video"))
		}
	}
	for _, input := range params.ReferenceAudio {
		if url := mediaURL(input); url != "" {
			content = append(content, mediaItem("audio_url", "audio_url", url, "reference_audio"))
		}
	}
	return content
}

func mediaItem(itemType, field, url, role string) map[string]any {
	item := map[string]any{
		"type": field,
		field:  map[string]any{"url": url},
	}
	if itemType != field {
		item["type"] = itemType
	}
	if role != "" {
		item["role"] = role
	}
	return item
}

func toVideoJob(raw map[string]any, fallbackModelID string) *sdk.VideoJob {
	inner := unwrap(raw)
	id := firstString(inner, "id", "task_id", "taskId")
	if id == "" {
		id = firstString(raw, "id", "task_id", "taskId")
	}
	modelID := firstString(inner, "model", "model_id")
	if modelID == "" {
		modelID = fallbackModelID
	}
	status := firstString(inner, "status", "task_status")
	progress := firstFloat(inner, "progress", "percent")

	job := &sdk.VideoJob{
		ID:               id,
		ModelID:          modelID,
		Status:           mapStatus(status),
		Progress:         progress,
		ProviderMetadata: raw,
	}
	if errMsg := extractError(inner); errMsg != "" {
		job.Error = &sdk.VideoError{Message: errMsg}
	}
	for _, url := range extractVideoURLs(inner) {
		job.Outputs = append(job.Outputs, sdk.VideoOutput{
			URL:         url,
			ContentType: "video/mp4",
			ProviderMetadata: map[string]any{
				"task_id": id,
			},
		})
	}
	return job
}

func unwrap(raw map[string]any) map[string]any {
	for _, key := range []string{"data", "task"} {
		if nested, ok := raw[key].(map[string]any); ok {
			return nested
		}
	}
	return raw
}

func mapStatus(status string) sdk.VideoJobStatus {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "queued", "pending":
		return sdk.VideoJobQueued
	case "running", "in_progress", "processing":
		return sdk.VideoJobRunning
	case "succeeded", "success", "completed":
		return sdk.VideoJobSucceeded
	case "failed", "error":
		return sdk.VideoJobFailed
	case "canceled", "cancelled", "deleted":
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

func firstString(obj map[string]any, keys ...string) string {
	for _, key := range keys {
		if s, ok := obj[key].(string); ok {
			return s
		}
	}
	return ""
}

func firstFloat(obj map[string]any, keys ...string) *float64 {
	for _, key := range keys {
		if f, ok := utils.ToFloat64(obj[key]); ok {
			return &f
		}
	}
	return nil
}

func extractError(obj map[string]any) string {
	if s, ok := obj["error"].(string); ok {
		return s
	}
	if m, ok := obj["error"].(map[string]any); ok {
		if s, ok := m["message"].(string); ok {
			return s
		}
	}
	if s, ok := obj["message"].(string); ok && mapStatus(firstString(obj, "status")) == sdk.VideoJobFailed {
		return s
	}
	return ""
}

func extractVideoURLs(v any) []string {
	var out []string
	var walk func(any)
	walk = func(value any) {
		switch typed := value.(type) {
		case map[string]any:
			for _, key := range []string{"video_url", "videoUrl", "output_url", "outputUrl"} {
				if s, ok := typed[key].(string); ok && s != "" {
					out = append(out, s)
				}
			}
			if s, ok := typed["url"].(string); ok && hasVideoExtension(s) {
				out = append(out, s)
			}
			for _, nested := range typed {
				walk(nested)
			}
		case []any:
			for _, item := range typed {
				walk(item)
			}
		}
	}
	walk(v)
	return dedupe(out)
}

func hasVideoExtension(s string) bool {
	if s == "" {
		return false
	}
	lower := strings.ToLower(s)
	return strings.Contains(lower, ".mp4") || strings.Contains(lower, ".mov")
}

func dedupe(values []string) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, len(values))
	for _, value := range values {
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}
