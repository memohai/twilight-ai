package videos

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"strconv"
	"strings"

	"github.com/memohai/twilight-ai/internal/utils"
	"github.com/memohai/twilight-ai/sdk"
)

const defaultBaseURL = "https://api.openai.com/v1"

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

func (p *Provider) ListModels(context.Context) ([]*sdk.VideoModel, error) {
	return []*sdk.VideoModel{
		{
			ID:       "sora-2",
			Provider: p,
			ProviderMetadata: map[string]any{
				"deprecated": true,
				"shutdown":   "2026-09-24",
			},
		},
		{
			ID:       "sora-2-pro",
			Provider: p,
			ProviderMetadata: map[string]any{
				"deprecated": true,
				"shutdown":   "2026-09-24",
			},
		},
	}, nil
}

func (p *Provider) DoCreate(ctx context.Context, params sdk.VideoParams) (*sdk.VideoJob, error) {
	if params.Model == nil {
		return nil, fmt.Errorf("openai videos: model is required")
	}
	if params.InputImage != nil && len(params.InputImage.Data) > 0 {
		return p.doCreateMultipart(ctx, params)
	}
	return p.doCreateJSON(ctx, params)
}

func (p *Provider) doCreateJSON(ctx context.Context, params sdk.VideoParams) (*sdk.VideoJob, error) {
	body := p.buildCreateBody(params)
	resp, err := utils.FetchJSON[videoResponse](ctx, p.httpClient, &utils.RequestOptions{
		Method:  http.MethodPost,
		BaseURL: p.baseURL,
		Path:    "/videos",
		Headers: utils.AuthHeader(p.apiKey),
		Body:    body,
	})
	if err != nil {
		return nil, fmt.Errorf("openai videos: create request failed: %w", err)
	}
	return toVideoJob(resp, params.Model.ID), nil
}

func (p *Provider) doCreateMultipart(ctx context.Context, params sdk.VideoParams) (*sdk.VideoJob, error) {
	body, contentType, err := buildMultipartBody(params)
	if err != nil {
		return nil, fmt.Errorf("openai videos: build multipart body: %w", err)
	}
	fullURL, err := utils.BuildURL(p.baseURL, "/videos")
	if err != nil {
		return nil, fmt.Errorf("openai videos: build URL: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fullURL, body)
	if err != nil {
		return nil, fmt.Errorf("openai videos: create request: %w", err)
	}
	req.Header.Set("Authorization", utils.BearerToken(p.apiKey))
	req.Header.Set("Content-Type", contentType)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai videos: multipart create request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("openai videos: create request failed with status %d: %s", resp.StatusCode, string(respBody))
	}
	var result videoResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("openai videos: decode response: %w", err)
	}
	return toVideoJob(&result, params.Model.ID), nil
}

func (p *Provider) DoGet(ctx context.Context, model *sdk.VideoModel, id string) (*sdk.VideoJob, error) {
	resp, err := utils.FetchJSON[videoResponse](ctx, p.httpClient, &utils.RequestOptions{
		Method:  http.MethodGet,
		BaseURL: p.baseURL,
		Path:    "/videos/" + id,
		Headers: utils.AuthHeader(p.apiKey),
	})
	if err != nil {
		return nil, fmt.Errorf("openai videos: get request failed: %w", err)
	}
	modelID := ""
	if model != nil {
		modelID = model.ID
	}
	return toVideoJob(resp, modelID), nil
}

func (p *Provider) DoCancel(_ context.Context, _ *sdk.VideoModel, _ string) error {
	return fmt.Errorf("openai videos: cancel is not supported")
}

func (p *Provider) DoDownload(ctx context.Context, _ *sdk.VideoModel, output sdk.VideoOutput) ([]byte, string, error) {
	if strings.HasPrefix(output.URL, "http://") || strings.HasPrefix(output.URL, "https://") {
		return p.downloadURL(ctx, output.URL, output.ContentType)
	}
	videoID := ""
	if output.ProviderMetadata != nil {
		if v, ok := output.ProviderMetadata["video_id"].(string); ok {
			videoID = v
		}
	}
	if videoID == "" && strings.HasPrefix(output.URL, "openai://") {
		videoID = strings.TrimPrefix(output.URL, "openai://")
	}
	if videoID == "" {
		return nil, "", fmt.Errorf("openai videos: output video_id is required")
	}
	query := map[string]string{}
	if output.ProviderMetadata != nil {
		if variant, ok := output.ProviderMetadata["variant"].(string); ok && variant != "" {
			query["variant"] = variant
		}
	}
	resp, err := utils.FetchRaw(ctx, p.httpClient, &utils.RequestOptions{
		Method:  http.MethodGet,
		BaseURL: p.baseURL,
		Path:    "/videos/" + videoID + "/content",
		Query:   query,
		Headers: utils.AuthHeader(p.apiKey),
	})
	if err != nil {
		return nil, "", fmt.Errorf("openai videos: download request failed: %w", err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, "", fmt.Errorf("openai videos: read download response: %w", err)
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

func (p *Provider) downloadURL(ctx context.Context, url, fallbackContentType string) ([]byte, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, http.NoBody)
	if err != nil {
		return nil, "", fmt.Errorf("openai videos: build download request: %w", err)
	}
	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, "", fmt.Errorf("openai videos: download request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		return nil, "", fmt.Errorf("openai videos: download failed with status %d: %s", resp.StatusCode, string(body))
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, "", fmt.Errorf("openai videos: read download response: %w", err)
	}
	contentType := resp.Header.Get("Content-Type")
	if contentType == "" {
		contentType = fallbackContentType
	}
	if contentType == "" {
		contentType = "video/mp4"
	}
	return data, contentType, nil
}

func (p *Provider) buildCreateBody(params sdk.VideoParams) map[string]any {
	body := map[string]any{
		"model":  params.Model.ID,
		"prompt": params.Prompt,
	}
	if params.Size != "" {
		body["size"] = params.Size
	}
	if params.DurationSeconds != nil {
		body["seconds"] = *params.DurationSeconds
	}
	if params.InputImage != nil {
		ref := map[string]any{}
		switch {
		case params.InputImage.FileID != "":
			ref["file_id"] = params.InputImage.FileID
		case params.InputImage.URL != "":
			ref["image_url"] = params.InputImage.URL
		case len(params.InputImage.Data) > 0:
			ref["image_url"] = dataURL(*params.InputImage)
		}
		if len(ref) > 0 {
			body["input_reference"] = ref
		}
	}
	for k, v := range params.Config {
		if k == "variant" {
			continue
		}
		body[k] = v
	}
	return body
}

func buildMultipartBody(params sdk.VideoParams) (*bytes.Buffer, string, error) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	if err := w.WriteField("model", params.Model.ID); err != nil {
		return nil, "", err
	}
	if err := w.WriteField("prompt", params.Prompt); err != nil {
		return nil, "", err
	}
	if params.Size != "" {
		if err := w.WriteField("size", params.Size); err != nil {
			return nil, "", err
		}
	}
	if params.DurationSeconds != nil {
		if err := w.WriteField("seconds", strconv.Itoa(*params.DurationSeconds)); err != nil {
			return nil, "", err
		}
	}
	for k, v := range params.Config {
		if k == "variant" {
			continue
		}
		if err := w.WriteField(k, fmt.Sprint(v)); err != nil {
			return nil, "", err
		}
	}
	if params.InputImage != nil && len(params.InputImage.Data) > 0 {
		filename := params.InputImage.Filename
		if filename == "" {
			filename = "input_reference"
		}
		contentType := params.InputImage.ContentType
		if contentType == "" {
			contentType = "application/octet-stream"
		}
		header := textproto.MIMEHeader{}
		header.Set("Content-Disposition", fmt.Sprintf(`form-data; name="input_reference"; filename="%s"`, escapeQuotes(filename)))
		header.Set("Content-Type", contentType)
		part, err := w.CreatePart(header)
		if err != nil {
			return nil, "", err
		}
		if _, err := part.Write(params.InputImage.Data); err != nil {
			return nil, "", err
		}
	}
	if err := w.Close(); err != nil {
		return nil, "", err
	}
	return &buf, w.FormDataContentType(), nil
}

func toVideoJob(resp *videoResponse, fallbackModelID string) *sdk.VideoJob {
	modelID := resp.Model
	if modelID == "" {
		modelID = fallbackModelID
	}
	job := &sdk.VideoJob{
		ID:       resp.ID,
		ModelID:  modelID,
		Status:   mapStatus(resp.Status),
		Progress: resp.Progress,
		ProviderMetadata: map[string]any{
			"object":       resp.Object,
			"created_at":   resp.CreatedAt,
			"completed_at": resp.CompletedAt,
			"seconds":      resp.Seconds,
			"size":         resp.Size,
		},
	}
	if resp.Error != nil {
		job.Error = &sdk.VideoError{Code: resp.Error.Code, Message: resp.Error.Message}
	}
	if job.Status == sdk.VideoJobSucceeded && resp.ID != "" {
		job.Outputs = append(job.Outputs, sdk.VideoOutput{
			URL:         "openai://" + resp.ID,
			ContentType: "video/mp4",
			ProviderMetadata: map[string]any{
				"video_id": resp.ID,
			},
		})
	}
	return job
}

func mapStatus(status string) sdk.VideoJobStatus {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "queued":
		return sdk.VideoJobQueued
	case "in_progress", "running":
		return sdk.VideoJobRunning
	case "completed", "succeeded":
		return sdk.VideoJobSucceeded
	case "failed", "error":
		return sdk.VideoJobFailed
	case "canceled", "cancelled":
		return sdk.VideoJobCanceled
	default:
		return sdk.VideoJobRunning
	}
}

func dataURL(input sdk.MediaInput) string {
	contentType := input.ContentType
	if contentType == "" {
		contentType = "application/octet-stream"
	}
	return "data:" + contentType + ";base64," + base64.StdEncoding.EncodeToString(input.Data)
}

func escapeQuotes(value string) string {
	return strings.ReplaceAll(value, `"`, `\"`)
}
