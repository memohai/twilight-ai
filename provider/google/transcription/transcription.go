package transcription

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/memohai/twilight-ai/internal/utils"
	sdk "github.com/memohai/twilight-ai/sdk"
)

const (
	defaultModelID = "gemini-2.5-flash"
	defaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"
)

type Option func(*Provider)

func WithAPIKey(key string) Option { return func(p *Provider) { p.apiKey = key } }
func WithBaseURL(url string) Option {
	return func(p *Provider) { p.baseURL = strings.TrimRight(url, "/") }
}
func WithHTTPClient(hc *http.Client) Option { return func(p *Provider) { p.httpClient = hc } }

type Provider struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

func New(opts ...Option) *Provider {
	p := &Provider{baseURL: defaultBaseURL, httpClient: &http.Client{}}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

func (p *Provider) TranscriptionModel(id string) *sdk.TranscriptionModel {
	if id == "" {
		id = defaultModelID
	}
	return &sdk.TranscriptionModel{ID: id, Provider: p}
}

func (p *Provider) ListModels(ctx context.Context) ([]*sdk.TranscriptionModel, error) {
	var resp struct {
		Models []struct {
			Name                       string   `json:"name"`
			DisplayName                string   `json:"displayName"`
			SupportedGenerationMethods []string `json:"supportedGenerationMethods"`
		} `json:"models"`
	}
	reqOpts := &utils.RequestOptions{
		Method:  http.MethodGet,
		BaseURL: p.baseURL,
		Path:    "/models",
		Headers: map[string]string{"x-goog-api-key": p.apiKey},
	}
	out, err := utils.FetchJSON[struct {
		Models []struct {
			Name                       string   `json:"name"`
			DisplayName                string   `json:"displayName"`
			SupportedGenerationMethods []string `json:"supportedGenerationMethods"`
		} `json:"models"`
	}](ctx, p.httpClient, reqOpts)
	if err != nil {
		return nil, fmt.Errorf("google transcription: list models request failed: %w", err)
	}
	resp.Models = out.Models

	models := make([]*sdk.TranscriptionModel, 0, len(resp.Models))
	for _, m := range resp.Models {
		if supportsGenerateContent(m.SupportedGenerationMethods) {
			id := strings.TrimPrefix(m.Name, "models/")
			models = append(models, p.TranscriptionModel(id))
		}
	}
	if len(models) == 0 {
		return nil, fmt.Errorf("google transcription: no transcription-capable models returned by provider")
	}
	return models, nil
}

func supportsGenerateContent(methods []string) bool {
	for _, m := range methods {
		if m == "generateContent" {
			return true
		}
	}
	return false
}

type audioConfig struct {
	Prompt string
}

func parseConfig(cfg map[string]any) audioConfig {
	ac := audioConfig{}
	if cfg == nil {
		return ac
	}
	if v, ok := cfg["prompt"].(string); ok && v != "" {
		ac.Prompt = v
	}
	return ac
}

func (p *Provider) DoTranscribe(ctx context.Context, params sdk.TranscriptionParams) (*sdk.TranscriptionResult, error) {
	cfg := parseConfig(params.Config)
	modelID := defaultModelID
	if params.Model != nil && params.Model.ID != "" {
		modelID = params.Model.ID
	}
	prompt := cfg.Prompt
	if prompt == "" {
		prompt = "Transcribe this audio exactly. Return only the transcript text."
	}

	reqBody := map[string]any{
		"contents": []map[string]any{
			{
				"role": "user",
				"parts": []map[string]any{
					{"text": prompt},
					{
						"inlineData": map[string]any{
							"mimeType": normalizedContentType(params.ContentType),
							"data":     base64.StdEncoding.EncodeToString(params.Audio),
						},
					},
				},
			},
		},
		"generationConfig": map[string]any{
			"temperature":     0,
			"maxOutputTokens": 8192,
		},
	}
	data, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("google transcription: marshal request: %w", err)
	}

	path := "/" + getModelPath(modelID) + ":generateContent"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+path, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("google transcription: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", p.apiKey)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("google transcription: request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		var body bytes.Buffer
		_, _ = body.ReadFrom(resp.Body)
		return nil, fmt.Errorf("google transcription: unexpected status %d: %s", resp.StatusCode, body.String())
	}

	var payload struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("google transcription: decode response: %w", err)
	}
	if len(payload.Candidates) == 0 || len(payload.Candidates[0].Content.Parts) == 0 {
		return nil, fmt.Errorf("google transcription: empty response")
	}
	return &sdk.TranscriptionResult{Text: payload.Candidates[0].Content.Parts[0].Text}, nil
}

func getModelPath(modelID string) string {
	if strings.HasPrefix(modelID, "models/") {
		return modelID
	}
	return "models/" + modelID
}

func normalizedContentType(contentType string) string {
	if contentType == "" {
		return "audio/wav"
	}
	return contentType
}
