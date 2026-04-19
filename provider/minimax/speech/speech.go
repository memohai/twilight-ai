// Package speech provides a MiniMax TTS provider.
// It targets the POST /v1/t2a_v2 endpoint.
// The response is a JSON object whose audio data is hex-encoded; this provider
// decodes the hex into raw bytes before returning.
// Streaming is emulated by returning the fully decoded audio as a single chunk.
package speech

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	sdk "github.com/memohai/twilight-ai/sdk"
)

const (
	defaultModelID   = "minimax-tts"
	defaultBaseURL   = "https://api.minimax.io"
	defaultVoiceID   = "English_expressive_narrator"
	defaultModel     = "speech-2.8-hd"
	defaultFormat    = "mp3"
	contentTypeAudio = "audio/mpeg"
)

// Option configures the MiniMax TTS provider.
type Option func(*Provider)

// WithAPIKey sets the MiniMax API key used as a Bearer credential.
func WithAPIKey(key string) Option {
	return func(p *Provider) { p.apiKey = key }
}

// WithBaseURL overrides the API base URL (useful for testing).
func WithBaseURL(u string) Option {
	return func(p *Provider) { p.baseURL = u }
}

// WithHTTPClient replaces the default HTTP client.
func WithHTTPClient(hc *http.Client) Option {
	return func(p *Provider) { p.httpClient = hc }
}

// Provider implements sdk.SpeechProvider for MiniMax TTS.
type Provider struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

// New creates a new MiniMax TTS provider.
func New(opts ...Option) *Provider {
	p := &Provider{
		baseURL:    defaultBaseURL,
		httpClient: &http.Client{},
	}
	for _, o := range opts {
		o(p)
	}
	p.baseURL = strings.TrimRight(p.baseURL, "/")
	return p
}

// SpeechModel creates a SpeechModel bound to this provider.
func (p *Provider) SpeechModel(id string) *sdk.SpeechModel {
	if id == "" {
		id = defaultModel
	}
	return &sdk.SpeechModel{ID: id, Provider: p}
}

// ListModels returns the speech models exposed by this provider.
func (p *Provider) ListModels(context.Context) ([]*sdk.SpeechModel, error) {
	return nil, fmt.Errorf("minimax speech: provider does not expose a remote models discovery API in this SDK")
}

// t2aResponse is the JSON structure returned by the MiniMax /v1/t2a_v2 endpoint.
type t2aResponse struct {
	Data struct {
		Audio string `json:"audio"` // hex-encoded audio bytes
	} `json:"data"`
	BaseResp struct {
		StatusCode int    `json:"status_code"`
		StatusMsg  string `json:"status_msg"`
	} `json:"base_resp"`
}

// DoSynthesize synthesizes speech and returns the complete audio bytes.
func (p *Provider) DoSynthesize(ctx context.Context, params sdk.SpeechParams) (*sdk.SpeechResult, error) {
	cfg := parseConfig(params.Config)
	if params.Model != nil && params.Model.ID != "" {
		cfg.Model = params.Model.ID
	}

	audio, err := p.synthesize(ctx, params.Text, &cfg)
	if err != nil {
		return nil, err
	}
	return &sdk.SpeechResult{
		Audio:       audio,
		ContentType: contentTypeForFormat(cfg.OutputFormat),
	}, nil
}

// DoStream returns a streaming result containing the fully synthesized audio as a single chunk.
// MiniMax does not expose a public streaming TTS endpoint, so DoStream behaves like DoSynthesize.
func (p *Provider) DoStream(ctx context.Context, params sdk.SpeechParams) (*sdk.SpeechStreamResult, error) {
	cfg := parseConfig(params.Config)
	if params.Model != nil && params.Model.ID != "" {
		cfg.Model = params.Model.ID
	}

	audio, err := p.synthesize(ctx, params.Text, &cfg)
	if err != nil {
		return nil, err
	}

	ch := make(chan []byte, 1)
	errCh := make(chan error, 1)

	ch <- audio
	close(ch)
	close(errCh)

	return sdk.NewSpeechStreamResult(ch, contentTypeForFormat(cfg.OutputFormat), errCh), nil
}

// synthesize calls the MiniMax API and returns decoded audio bytes.
func (p *Provider) synthesize(ctx context.Context, text string, cfg *audioConfig) ([]byte, error) {
	// speed, vol, pitch are always sent with their defaults (1.0, 1.0, 0) so the
	// server receives explicit values rather than relying on its own defaults.
	voiceSetting := map[string]any{
		"voice_id": cfg.VoiceID,
		"speed":    cfg.Speed,
		"vol":      cfg.Vol,
		"pitch":    cfg.Pitch,
	}

	reqBody := map[string]any{
		"model":         cfg.Model,
		"text":          text,
		"voice_setting": voiceSetting,
		"audio_setting": map[string]any{
			"format":      cfg.OutputFormat,
			"sample_rate": cfg.SampleRate,
		},
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("minimax speech: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/v1/t2a_v2", bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("minimax speech: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("minimax speech: request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("minimax speech: unexpected status %d: %s", resp.StatusCode, string(body))
	}

	var result t2aResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("minimax speech: decode response: %w", err)
	}
	if result.BaseResp.StatusCode != 0 {
		return nil, fmt.Errorf("minimax speech: api error %d: %s",
			result.BaseResp.StatusCode, result.BaseResp.StatusMsg)
	}
	if result.Data.Audio == "" {
		return nil, fmt.Errorf("minimax speech: empty audio in response")
	}

	audio, err := hex.DecodeString(result.Data.Audio)
	if err != nil {
		return nil, fmt.Errorf("minimax speech: hex decode: %w", err)
	}
	return audio, nil
}
