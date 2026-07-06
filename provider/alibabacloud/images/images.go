// Package images provides an Alibaba Cloud Model Studio DashScope image provider.
package images

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/memohai/twilight-ai/internal/utils"
	"github.com/memohai/twilight-ai/sdk"
)

const (
	defaultBaseURL      = "https://dashscope.aliyuncs.com/api/v1"
	defaultPollInterval = 3 * time.Second
	defaultPollTimeout  = 5 * time.Minute
	wanGenerationPath   = "/services/aigc/image-generation/generation"
	text2ImagePath      = "/services/aigc/text2image/image-synthesis"
	qwenMultimodalPath  = "/services/aigc/multimodal-generation/generation"
)

// Provider implements sdk.ImageGenerationProvider for Alibaba Cloud Model Studio
// DashScope image generation models such as Qwen-Image and Wan image models.
type Provider struct {
	apiKey       string
	baseURL      string
	httpClient   *http.Client
	pollInterval time.Duration
	pollTimeout  time.Duration
}

// Option configures the Provider.
type Option func(*Provider)

// WithAPIKey sets the DashScope or Model Studio API key.
func WithAPIKey(apiKey string) Option {
	return func(p *Provider) { p.apiKey = apiKey }
}

// WithBaseURL overrides the DashScope HTTP API base URL.
//
// The provider accepts both native DashScope bases such as
// https://dashscope.aliyuncs.com/api/v1 and Model Studio compatible-mode bases
// such as https://{workspace}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1.
func WithBaseURL(baseURL string) Option {
	return func(p *Provider) { p.baseURL = normalizeBaseURL(baseURL) }
}

// WithHTTPClient sets the HTTP client used for requests.
func WithHTTPClient(client *http.Client) Option {
	return func(p *Provider) {
		if client != nil {
			p.httpClient = client
		}
	}
}

// WithPollInterval sets the async task polling interval.
func WithPollInterval(interval time.Duration) Option {
	return func(p *Provider) {
		if interval > 0 {
			p.pollInterval = interval
		}
	}
}

// WithPollTimeout sets the maximum time to wait for an async image task.
func WithPollTimeout(timeout time.Duration) Option {
	return func(p *Provider) {
		if timeout > 0 {
			p.pollTimeout = timeout
		}
	}
}

// New creates a new DashScope image provider.
func New(options ...Option) *Provider {
	p := &Provider{
		baseURL:      defaultBaseURL,
		httpClient:   &http.Client{},
		pollInterval: defaultPollInterval,
		pollTimeout:  defaultPollTimeout,
	}
	for _, opt := range options {
		opt(p)
	}
	return p
}

// GenerationModel creates an ImageGenerationModel bound to this provider.
func (p *Provider) GenerationModel(id string) *sdk.ImageGenerationModel {
	return &sdk.ImageGenerationModel{
		ID:       id,
		Provider: p,
	}
}

// DoGenerate implements sdk.ImageGenerationProvider.
func (p *Provider) DoGenerate(ctx context.Context, params *sdk.ImageGenerationParams) (*sdk.ImageResult, error) {
	if params.Model == nil {
		return nil, fmt.Errorf("alibabacloud images: generation model is required")
	}
	if isQwenMultimodalModel(params.Model.ID) {
		return p.generateQwenMultimodal(ctx, params)
	}

	resp, err := p.createTask(ctx, params)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(resp.Code) != "" {
		return nil, fmt.Errorf("alibabacloud images: generation request failed: %s: %s", resp.Code, resp.Message)
	}
	if hasImages(resp.Output.Choices) || hasResultImages(resp.Output.Results) {
		return toImageResult(resp), nil
	}
	if strings.EqualFold(resp.Output.TaskStatus, taskStatusSucceeded) {
		return nil, fmt.Errorf("alibabacloud images: generation request succeeded without image output")
	}
	taskID := strings.TrimSpace(resp.Output.TaskID)
	if taskID == "" {
		return nil, fmt.Errorf("alibabacloud images: generation response did not include a task_id")
	}
	return p.waitTask(ctx, taskID)
}

func (p *Provider) createTask(ctx context.Context, params *sdk.ImageGenerationParams) (*dashScopeResponse, error) {
	path := wanGenerationPath
	var body any = dashScopeImageRequest{
		Model:      params.Model.ID,
		Input:      promptMessagesInput(params.Prompt),
		Parameters: imageParameters(params),
	}
	if usesPromptInputText2Image(params.Model.ID) {
		path = text2ImagePath
		body = dashScopePromptInputRequest{
			Model: params.Model.ID,
			Input: dashScopePromptInput{
				Prompt: params.Prompt,
			},
			Parameters: imageParameters(params),
		}
	}

	resp, err := utils.FetchJSON[dashScopeResponse](ctx, p.httpClient, &utils.RequestOptions{
		Method:  http.MethodPost,
		BaseURL: p.baseURL,
		Path:    path,
		Headers: map[string]string{
			"Authorization":     utils.BearerToken(p.apiKey),
			"X-DashScope-Async": "enable",
		},
		Body: body,
	})
	if err != nil {
		return nil, fmt.Errorf("alibabacloud images: create task request failed: %w", err)
	}
	return resp, nil
}

func (p *Provider) generateQwenMultimodal(ctx context.Context, params *sdk.ImageGenerationParams) (*sdk.ImageResult, error) {
	req := dashScopeImageRequest{
		Model:      params.Model.ID,
		Input:      promptMessagesInput(params.Prompt),
		Parameters: imageParameters(params),
	}
	resp, err := utils.FetchJSON[dashScopeResponse](ctx, p.httpClient, &utils.RequestOptions{
		Method:  http.MethodPost,
		BaseURL: p.baseURL,
		Path:    qwenMultimodalPath,
		Headers: utils.AuthHeader(p.apiKey),
		Body:    req,
	})
	if err != nil {
		return nil, fmt.Errorf("alibabacloud images: qwen multimodal generation request failed: %w", err)
	}
	if strings.TrimSpace(resp.Code) != "" {
		return nil, fmt.Errorf("alibabacloud images: qwen multimodal generation failed: %s: %s", resp.Code, resp.Message)
	}
	if !hasImages(resp.Output.Choices) && !hasResultImages(resp.Output.Results) {
		return nil, fmt.Errorf("alibabacloud images: qwen multimodal generation response did not include image output")
	}
	return toImageResult(resp), nil
}

func (p *Provider) waitTask(ctx context.Context, taskID string) (*sdk.ImageResult, error) {
	waitCtx := ctx
	cancel := func() {}
	if p.pollTimeout > 0 {
		waitCtx, cancel = context.WithTimeout(ctx, p.pollTimeout)
	}
	defer cancel()

	for {
		resp, err := p.getTask(waitCtx, taskID)
		if err != nil {
			return nil, err
		}
		if code := firstNonEmpty(resp.Code, resp.Output.Code); code != "" {
			return nil, fmt.Errorf("alibabacloud images: task %s failed: %s: %s", taskID, code, firstNonEmpty(resp.Message, resp.Output.Message))
		}

		switch strings.ToUpper(strings.TrimSpace(resp.Output.TaskStatus)) {
		case taskStatusSucceeded:
			if !hasImages(resp.Output.Choices) && !hasResultImages(resp.Output.Results) {
				return nil, fmt.Errorf("alibabacloud images: task %s succeeded without image output", taskID)
			}
			return toImageResult(resp), nil
		case taskStatusFailed, taskStatusUnknown, taskStatusCanceled:
			if message := firstNonEmpty(resp.Message, resp.Output.Message); message != "" {
				return nil, fmt.Errorf("alibabacloud images: task %s finished with status %s: %s", taskID, resp.Output.TaskStatus, message)
			}
			return nil, fmt.Errorf("alibabacloud images: task %s finished with status %s", taskID, resp.Output.TaskStatus)
		}

		timer := time.NewTimer(p.pollInterval)
		select {
		case <-waitCtx.Done():
			timer.Stop()
			return nil, fmt.Errorf("alibabacloud images: task %s did not finish before timeout: %w", taskID, waitCtx.Err())
		case <-timer.C:
		}
	}
}

func (p *Provider) getTask(ctx context.Context, taskID string) (*dashScopeResponse, error) {
	resp, err := utils.FetchJSON[dashScopeResponse](ctx, p.httpClient, &utils.RequestOptions{
		Method:  http.MethodGet,
		BaseURL: p.baseURL,
		Path:    "/tasks/" + taskID,
		Headers: utils.AuthHeader(p.apiKey),
	})
	if err != nil {
		return nil, fmt.Errorf("alibabacloud images: get task request failed: %w", err)
	}
	return resp, nil
}

func toImageResult(resp *dashScopeResponse) *sdk.ImageResult {
	result := &sdk.ImageResult{}
	if resp == nil {
		return result
	}
	if resp.Usage != nil {
		result.Usage = sdk.ImageUsage{
			TotalTokens:  resp.Usage.TotalTokens,
			InputTokens:  resp.Usage.InputTokens,
			OutputTokens: resp.Usage.OutputTokens,
		}
	}
	for _, choice := range resp.Output.Choices {
		for _, part := range choice.Message.Content {
			if imageURL := strings.TrimSpace(part.Image); imageURL != "" {
				result.Data = append(result.Data, sdk.ImageData{URL: imageURL})
			}
		}
	}
	for _, item := range resp.Output.Results {
		if imageURL := strings.TrimSpace(item.URL); imageURL != "" {
			result.Data = append(result.Data, sdk.ImageData{URL: imageURL})
		}
	}
	return result
}

func hasImages(choices []dashScopeImageChoice) bool {
	for _, choice := range choices {
		for _, part := range choice.Message.Content {
			if strings.TrimSpace(part.Image) != "" {
				return true
			}
		}
	}
	return false
}

func hasResultImages(results []dashScopeImageResult) bool {
	for _, item := range results {
		if strings.TrimSpace(item.URL) != "" {
			return true
		}
	}
	return false
}

func promptMessagesInput(prompt string) dashScopeImageInput {
	return dashScopeImageInput{
		Messages: []dashScopeImageMessage{{
			Role:    "user",
			Content: []dashScopeImageContent{{Text: prompt}},
		}},
	}
}

func imageParameters(params *sdk.ImageGenerationParams) dashScopeImageParameters {
	return dashScopeImageParameters{
		N:    params.N,
		Size: dashScopeImageSize(params.Size),
	}
}

func dashScopeImageSize(size string) string {
	size = strings.TrimSpace(size)
	if size == "" {
		return ""
	}
	return strings.ReplaceAll(size, "x", "*")
}

func isQwenLegacyImageModel(modelID string) bool {
	modelID = strings.ToLower(strings.TrimSpace(modelID))
	return modelID == "qwen-image" || modelID == "qwen-image-plus"
}

func isQwenMultimodalModel(modelID string) bool {
	modelID = strings.ToLower(strings.TrimSpace(modelID))
	return strings.HasPrefix(modelID, "qwen-image") && !isQwenLegacyImageModel(modelID)
}

func usesPromptInputText2Image(modelID string) bool {
	return isQwenLegacyImageModel(modelID) || isLegacyWanText2ImageModel(modelID)
}

func isLegacyWanText2ImageModel(modelID string) bool {
	modelID = strings.ToLower(strings.TrimSpace(modelID))
	if strings.HasPrefix(modelID, "wanx") {
		return true
	}
	if !strings.HasPrefix(modelID, "wan2.") {
		return false
	}
	remainder := strings.TrimPrefix(modelID, "wan2.")
	end := 0
	for end < len(remainder) && remainder[end] >= '0' && remainder[end] <= '9' {
		end++
	}
	if end == 0 {
		return false
	}
	minor, err := strconv.Atoi(remainder[:end])
	return err == nil && minor <= 5
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			return trimmed
		}
	}
	return ""
}

func normalizeBaseURL(baseURL string) string {
	trimmed := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if trimmed == "" {
		return defaultBaseURL
	}
	parsed, err := url.Parse(trimmed)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return trimmed
	}
	switch {
	case strings.Contains(parsed.Path, "/compatible-mode/v1"):
		idx := strings.Index(parsed.Path, "/compatible-mode/v1")
		parsed.Path = parsed.Path[:idx] + "/api/v1"
	case strings.Contains(parsed.Path, "/api/v1"):
		parsed.Path = parsed.Path[:strings.Index(parsed.Path, "/api/v1")+len("/api/v1")]
	case parsed.Path == "" || parsed.Path == "/":
		parsed.Path = "/api/v1"
	}
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return strings.TrimRight(parsed.String(), "/")
}
