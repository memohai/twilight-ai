// Package speech provides a Deepgram TTS provider.
// It targets the POST /v1/speak endpoint and uses Token-based authentication.
package speech

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/memohai/twilight-ai/internal/utils"
	sdk "github.com/memohai/twilight-ai/sdk"
)

const (
	// defaultModelID is the SDK-level model identifier exposed via SpeechModel.
	defaultModelID = "deepgram-tts"
	defaultBaseURL = "https://api.deepgram.com"
	// defaultVoiceModel is the Deepgram voice model sent in the API query parameter.
	defaultVoiceModel = "aura-2-asteria-en"
	contentTypeAudio  = "audio/mpeg"
)

// Option configures the Deepgram TTS provider.
type Option func(*Provider)

// WithAPIKey sets the Deepgram API key used as a Token credential.
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

// Provider implements sdk.SpeechProvider for Deepgram TTS.
type Provider struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

// New creates a new Deepgram TTS provider.
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
		id = defaultVoiceModel
	}
	return &sdk.SpeechModel{ID: id, Provider: p}
}

// ListModels returns the speech models exposed by this provider.
func (p *Provider) ListModels(ctx context.Context) ([]*sdk.SpeechModel, error) {
	resp, err := p.listModels(ctx)
	if err != nil {
		return nil, err
	}

	models := make([]*sdk.SpeechModel, 0, len(resp.TTS))
	for _, m := range resp.TTS {
		if m.CanonicalName != "" {
			models = append(models, p.SpeechModel(m.CanonicalName))
		}
	}
	if len(models) == 0 {
		return nil, errors.New("deepgram speech: no speech models returned by provider")
	}
	return models, nil
}

type deepgramModelsResponse struct {
	STT []deepgramModel `json:"stt"`
	TTS []deepgramModel `json:"tts"`
}

type deepgramModel struct {
	CanonicalName string `json:"canonical_name"`
}

func (p *Provider) listModels(ctx context.Context) (*deepgramModelsResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.baseURL+"/v1/models", http.NoBody)
	if err != nil {
		return nil, fmt.Errorf("deepgram speech: build list models request: %w", err)
	}
	req.Header.Set("Authorization", "Token "+p.apiKey)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("deepgram speech: list models request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("deepgram speech: unexpected status %d: %s", resp.StatusCode, string(body))
	}

	var payload deepgramModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("deepgram speech: decode list models response: %w", err)
	}
	return &payload, nil
}

// DoSynthesize synthesizes speech and returns the complete audio bytes.
func (p *Provider) DoSynthesize(ctx context.Context, params sdk.SpeechParams) (*sdk.SpeechResult, error) {
	cfg := parseConfig(params.Config)
	if params.Model != nil && params.Model.ID != "" {
		cfg.Model = params.Model.ID
	}

	body, err := p.doRequest(ctx, params.Text, cfg)
	if err != nil {
		return nil, err
	}
	defer body.Close()

	audio, err := io.ReadAll(body)
	if err != nil {
		return nil, fmt.Errorf("deepgram speech: read response: %w", err)
	}
	return &sdk.SpeechResult{
		Audio:       audio,
		ContentType: contentTypeForEncoding(cfg.Encoding, cfg.Container),
	}, nil
}

// DoStream synthesizes speech and returns a streaming result backed by chunked HTTP body.
func (p *Provider) DoStream(ctx context.Context, params sdk.SpeechParams) (*sdk.SpeechStreamResult, error) {
	cfg := parseConfig(params.Config)
	if params.Model != nil && params.Model.ID != "" {
		cfg.Model = params.Model.ID
	}

	body, err := p.doRequest(ctx, params.Text, cfg)
	if err != nil {
		return nil, err
	}

	ch, errCh := utils.StreamHTTPBody(ctx, body, "deepgram speech")
	return sdk.NewSpeechStreamResult(ch, contentTypeForEncoding(cfg.Encoding, cfg.Container), errCh), nil
}

// doRequest sends a POST /v1/speak request and returns the raw response body.
func (p *Provider) doRequest(ctx context.Context, text string, cfg audioConfig) (io.ReadCloser, error) {
	u, err := url.Parse(p.baseURL + "/v1/speak")
	if err != nil {
		return nil, fmt.Errorf("deepgram speech: parse url: %w", err)
	}
	q := u.Query()
	q.Set("model", cfg.Model)
	if cfg.Encoding != "" {
		q.Set("encoding", cfg.Encoding)
	}
	if cfg.SampleRate > 0 {
		q.Set("sample_rate", fmt.Sprintf("%d", cfg.SampleRate))
	}
	if cfg.Container != "" {
		q.Set("container", cfg.Container)
	}
	u.RawQuery = q.Encode()

	reqBody, err := json.Marshal(map[string]string{"text": text})
	if err != nil {
		return nil, fmt.Errorf("deepgram speech: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u.String(), bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("deepgram speech: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Token "+p.apiKey)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("deepgram speech: request: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		return nil, fmt.Errorf("deepgram speech: unexpected status %d: %s", resp.StatusCode, string(respBody))
	}
	return resp.Body, nil
}
