package transcription

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"strings"

	sdk "github.com/memohai/twilight-ai/sdk"
)

const (
	defaultModelID  = "mimo-v2.5-asr"
	defaultBaseURL  = "https://api.xiaomimimo.com/v1"
	defaultLanguage = "auto"
)

type Option func(*Provider)

func WithAPIKey(key string) Option {
	return func(p *Provider) { p.apiKey = key }
}

func WithBaseURL(url string) Option {
	return func(p *Provider) { p.baseURL = strings.TrimRight(url, "/") }
}

func WithHTTPClient(hc *http.Client) Option {
	return func(p *Provider) {
		if hc != nil {
			p.httpClient = hc
		}
	}
}

type Provider struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

func New(opts ...Option) *Provider {
	p := &Provider{
		baseURL:    defaultBaseURL,
		httpClient: &http.Client{},
	}
	for _, opt := range opts {
		opt(p)
	}
	p.baseURL = strings.TrimRight(p.baseURL, "/")
	return p
}

func (p *Provider) TranscriptionModel(id string) *sdk.TranscriptionModel {
	if id == "" {
		id = defaultModelID
	}
	return &sdk.TranscriptionModel{ID: id, Provider: p}
}

func (p *Provider) ListModels(context.Context) ([]*sdk.TranscriptionModel, error) {
	return []*sdk.TranscriptionModel{p.TranscriptionModel(defaultModelID)}, nil
}

func (p *Provider) DoTranscribe(ctx context.Context, params sdk.TranscriptionParams) (*sdk.TranscriptionResult, error) {
	modelID := defaultModelID
	if params.Model != nil && strings.TrimSpace(params.Model.ID) != "" {
		modelID = strings.TrimSpace(params.Model.ID)
	}
	cfg := parseConfig(params.Config)

	reqBody := map[string]any{
		"model": modelID,
		"messages": []map[string]any{
			{
				"role": "user",
				"content": []map[string]any{
					{
						"type": "input_audio",
						"input_audio": map[string]any{
							"data": buildAudioDataURL(params.Audio, params.ContentType, params.Filename),
						},
					},
				},
			},
		},
		"asr_options": map[string]any{
			"language": cfg.Language,
		},
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("mimo transcription: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("mimo transcription: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("api-key", p.apiKey)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("mimo transcription: request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("mimo transcription: unexpected status %d: %s", resp.StatusCode, string(body))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("mimo transcription: read response: %w", err)
	}
	return decodeResponse(body)
}

type transcriptionConfig struct {
	Language string
}

func parseConfig(cfg map[string]any) transcriptionConfig {
	out := transcriptionConfig{Language: defaultLanguage}
	if cfg == nil {
		return out
	}
	if v, ok := cfg["language"].(string); ok && strings.TrimSpace(v) != "" {
		out.Language = strings.TrimSpace(v)
	}
	return out
}

func buildAudioDataURL(audio []byte, contentType, filename string) string {
	mime := strings.TrimSpace(contentType)
	if mime == "" {
		switch strings.ToLower(filepath.Ext(filename)) {
		case ".mp3":
			mime = "audio/mpeg"
		case ".wav":
			mime = "audio/wav"
		default:
			mime = "audio/wav"
		}
	}
	return "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(audio)
}

type response struct {
	Choices []struct {
		Message struct {
			Content any `json:"content"`
			Audio   *struct {
				Transcript string `json:"transcript"`
			} `json:"audio"`
		} `json:"message"`
	} `json:"choices"`
	Text string `json:"text"`
}

func decodeResponse(body []byte) (*sdk.TranscriptionResult, error) {
	var payload response
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, fmt.Errorf("mimo transcription: decode response: %w", err)
	}

	text := strings.TrimSpace(payload.Text)
	if text == "" && len(payload.Choices) > 0 {
		text = extractContentText(payload.Choices[0].Message.Content)
		if text == "" && payload.Choices[0].Message.Audio != nil {
			text = strings.TrimSpace(payload.Choices[0].Message.Audio.Transcript)
		}
	}
	if text == "" {
		return nil, fmt.Errorf("mimo transcription: response missing transcript text")
	}

	return &sdk.TranscriptionResult{Text: text}, nil
}

func extractContentText(content any) string {
	switch v := content.(type) {
	case string:
		return strings.TrimSpace(v)
	case []any:
		parts := make([]string, 0, len(v))
		for _, item := range v {
			part := strings.TrimSpace(extractContentText(item))
			if part != "" {
				parts = append(parts, part)
			}
		}
		return strings.Join(parts, "\n")
	case map[string]any:
		if text, ok := v["text"].(string); ok {
			return strings.TrimSpace(text)
		}
		if textObj, ok := v["text"].(map[string]any); ok {
			if value, ok := textObj["value"].(string); ok {
				return strings.TrimSpace(value)
			}
		}
		if contentValue, ok := v["content"]; ok {
			return extractContentText(contentValue)
		}
	}
	return ""
}
