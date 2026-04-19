package transcription

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	sdk "github.com/memohai/twilight-ai/sdk"
)

const (
	defaultModelID = "nova-3"
	defaultBaseURL = "https://api.deepgram.com"
)

type Option func(*Provider)

func WithAPIKey(key string) Option { return func(p *Provider) { p.apiKey = key } }
func WithBaseURL(baseURL string) Option {
	return func(p *Provider) { p.baseURL = strings.TrimRight(baseURL, "/") }
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

func (p *Provider) ListModels(context.Context) ([]*sdk.TranscriptionModel, error) {
	return nil, fmt.Errorf("deepgram transcription: provider does not expose a remote models discovery API in this SDK")
}

type audioConfig struct {
	Language    string
	SmartFormat bool
	DetectLang  bool
	Diarize     bool
	Punctuate   bool
}

func parseConfig(cfg map[string]any) audioConfig {
	ac := audioConfig{SmartFormat: true, Punctuate: true}
	if cfg == nil {
		return ac
	}
	if v, ok := cfg["language"].(string); ok && v != "" {
		ac.Language = v
	}
	if v, ok := cfg["smart_format"].(bool); ok {
		ac.SmartFormat = v
	}
	if v, ok := cfg["detect_language"].(bool); ok {
		ac.DetectLang = v
	}
	if v, ok := cfg["diarize"].(bool); ok {
		ac.Diarize = v
	}
	if v, ok := cfg["punctuate"].(bool); ok {
		ac.Punctuate = v
	}
	return ac
}

func (p *Provider) DoTranscribe(ctx context.Context, params sdk.TranscriptionParams) (*sdk.TranscriptionResult, error) {
	cfg := parseConfig(params.Config)
	modelID := defaultModelID
	if params.Model != nil && params.Model.ID != "" {
		modelID = params.Model.ID
	}

	u, err := url.Parse(p.baseURL + "/v1/listen")
	if err != nil {
		return nil, fmt.Errorf("deepgram transcription: parse URL: %w", err)
	}
	q := u.Query()
	q.Set("model", modelID)
	if cfg.Language != "" {
		q.Set("language", cfg.Language)
	}
	if cfg.SmartFormat {
		q.Set("smart_format", "true")
	}
	if cfg.DetectLang {
		q.Set("detect_language", "true")
	}
	if cfg.Diarize {
		q.Set("diarize", "true")
	}
	if cfg.Punctuate {
		q.Set("punctuate", "true")
	}
	u.RawQuery = q.Encode()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u.String(), bytes.NewReader(params.Audio))
	if err != nil {
		return nil, fmt.Errorf("deepgram transcription: build request: %w", err)
	}
	if params.ContentType != "" {
		req.Header.Set("Content-Type", params.ContentType)
	} else {
		req.Header.Set("Content-Type", "audio/wav")
	}
	req.Header.Set("Authorization", "Token "+p.apiKey)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("deepgram transcription: request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("deepgram transcription: unexpected status %d: %s", resp.StatusCode, string(body))
	}

	var payload struct {
		Results struct {
			Channels []struct {
				DetectedLanguage string `json:"detected_language"`
				Alternatives     []struct {
					Transcript string `json:"transcript"`
					Words      []struct {
						Word    string  `json:"word"`
						Start   float64 `json:"start"`
						End     float64 `json:"end"`
						Speaker int     `json:"speaker"`
					} `json:"words"`
				} `json:"alternatives"`
			} `json:"channels"`
		} `json:"results"`
		Metadata struct {
			Duration float64 `json:"duration"`
		} `json:"metadata"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("deepgram transcription: decode response: %w", err)
	}
	if len(payload.Results.Channels) == 0 || len(payload.Results.Channels[0].Alternatives) == 0 {
		return nil, fmt.Errorf("deepgram transcription: empty transcript in response")
	}
	alt := payload.Results.Channels[0].Alternatives[0]
	out := &sdk.TranscriptionResult{
		Text:            alt.Transcript,
		Language:        payload.Results.Channels[0].DetectedLanguage,
		DurationSeconds: payload.Metadata.Duration,
	}
	if len(alt.Words) > 0 {
		out.Words = make([]sdk.TranscriptionWord, 0, len(alt.Words))
		for _, w := range alt.Words {
			out.Words = append(out.Words, sdk.TranscriptionWord{
				Text:      w.Word,
				Start:     w.Start,
				End:       w.End,
				SpeakerID: fmt.Sprintf("speaker_%d", w.Speaker),
			})
		}
	}
	return out, nil
}
