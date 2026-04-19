// Package speech provides an ElevenLabs TTS provider.
// It targets the /v1/text-to-speech/{voice_id} (full) and
// /v1/text-to-speech/{voice_id}/stream (streaming) endpoints.
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
	defaultModelID   = "elevenlabs-tts"
	defaultBaseURL   = "https://api.elevenlabs.io"
	defaultModelLLM  = "eleven_multilingual_v2"
	defaultFormat    = "mp3_44100_128"
	contentTypeAudio = "audio/mpeg"
)

// Option configures the ElevenLabs TTS provider.
type Option func(*Provider)

// WithAPIKey sets the ElevenLabs API key (xi-api-key header).
func WithAPIKey(key string) Option {
	return func(p *Provider) { p.apiKey = key }
}

// WithBaseURL overrides the API base URL (useful for testing).
func WithBaseURL(rawURL string) Option {
	return func(p *Provider) { p.baseURL = rawURL }
}

// WithHTTPClient replaces the default HTTP client.
func WithHTTPClient(hc *http.Client) Option {
	return func(p *Provider) { p.httpClient = hc }
}

// Provider implements sdk.SpeechProvider for ElevenLabs TTS.
type Provider struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

// New creates a new ElevenLabs TTS provider.
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
		id = defaultModelLLM
	}
	return &sdk.SpeechModel{ID: id, Provider: p}
}

// ListModels returns the speech models exposed by this provider.
func (p *Provider) ListModels(ctx context.Context) ([]*sdk.SpeechModel, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.baseURL+"/v1/models", http.NoBody)
	if err != nil {
		return nil, fmt.Errorf("elevenlabs speech: build list models request: %w", err)
	}
	req.Header.Set("xi-api-key", p.apiKey)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("elevenlabs speech: list models request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("elevenlabs speech: unexpected status %d: %s", resp.StatusCode, string(body))
	}

	rawModels, err := decodeModelsResponse(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("elevenlabs speech: decode response: %w", err)
	}

	models := make([]*sdk.SpeechModel, 0, len(rawModels))
	for _, m := range rawModels {
		if m.CanDoTTS && m.ModelID != "" {
			models = append(models, p.SpeechModel(m.ModelID))
		}
	}
	if len(models) == 0 {
		return nil, errors.New("elevenlabs speech: no speech models returned by provider")
	}
	return models, nil
}

type elevenlabsModel struct {
	ModelID  string `json:"model_id"`
	CanDoTTS bool   `json:"can_do_text_to_speech"`
}

func decodeModelsResponse(r io.Reader) ([]elevenlabsModel, error) {
	body, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}

	var wrapped struct {
		Models []elevenlabsModel `json:"models"`
	}
	if err := json.Unmarshal(body, &wrapped); err == nil && len(wrapped.Models) > 0 {
		return wrapped.Models, nil
	}

	var direct []elevenlabsModel
	if err := json.Unmarshal(body, &direct); err != nil {
		return nil, err
	}
	return direct, nil
}

// DoSynthesize synthesizes speech and returns the complete audio bytes.
// Uses the non-streaming /v1/text-to-speech/{voice_id} endpoint.
func (p *Provider) DoSynthesize(ctx context.Context, params sdk.SpeechParams) (*sdk.SpeechResult, error) {
	cfg := parseConfig(params.Config)
	if params.Model != nil && params.Model.ID != "" {
		cfg.ModelID = params.Model.ID
	}
	if cfg.VoiceID == "" {
		return nil, fmt.Errorf("elevenlabs speech: voice_id is required")
	}

	endpoint := p.baseURL + "/v1/text-to-speech/" + cfg.VoiceID
	body, err := p.doRequest(ctx, endpoint, params.Text, &cfg)
	if err != nil {
		return nil, err
	}
	defer body.Close()

	audio, err := io.ReadAll(body)
	if err != nil {
		return nil, fmt.Errorf("elevenlabs speech: read response: %w", err)
	}
	return &sdk.SpeechResult{
		Audio:       audio,
		ContentType: contentTypeForFormat(cfg.OutputFormat),
	}, nil
}

// DoStream synthesizes speech and returns a streaming result.
// Uses the /v1/text-to-speech/{voice_id}/stream endpoint which returns chunked audio.
func (p *Provider) DoStream(ctx context.Context, params sdk.SpeechParams) (*sdk.SpeechStreamResult, error) {
	cfg := parseConfig(params.Config)
	if params.Model != nil && params.Model.ID != "" {
		cfg.ModelID = params.Model.ID
	}
	if cfg.VoiceID == "" {
		return nil, fmt.Errorf("elevenlabs speech: voice_id is required")
	}

	endpoint := p.baseURL + "/v1/text-to-speech/" + cfg.VoiceID + "/stream"
	body, err := p.doRequest(ctx, endpoint, params.Text, &cfg)
	if err != nil {
		return nil, err
	}

	ch, errCh := utils.StreamHTTPBody(ctx, body, "elevenlabs speech")
	return sdk.NewSpeechStreamResult(ch, contentTypeForFormat(cfg.OutputFormat), errCh), nil
}

// doRequest sends a POST TTS request and returns the response body.
func (p *Provider) doRequest(ctx context.Context, endpoint, text string, cfg *audioConfig) (io.ReadCloser, error) {
	// voice_settings must always include speed (default 1.0), per API contract.
	voiceSettings := map[string]any{
		"stability":         cfg.Stability,
		"similarity_boost":  cfg.SimilarityBoost,
		"style":             cfg.Style,
		"use_speaker_boost": cfg.UseSpeakerBoost,
		"speed":             cfg.Speed,
	}
	reqBody := map[string]any{
		"text":           text,
		"model_id":       cfg.ModelID,
		"voice_settings": voiceSettings,
	}
	if cfg.Seed != nil {
		reqBody["seed"] = *cfg.Seed
	}
	if cfg.ApplyTextNormalization != "" {
		reqBody["apply_text_normalization"] = cfg.ApplyTextNormalization
	}
	if cfg.LanguageCode != "" {
		reqBody["language_code"] = cfg.LanguageCode
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("elevenlabs speech: marshal request: %w", err)
	}

	reqURL := endpoint
	if cfg.OutputFormat != "" {
		q := url.Values{"output_format": {cfg.OutputFormat}}
		reqURL += "?" + q.Encode()
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, reqURL, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("elevenlabs speech: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "audio/mpeg")
	req.Header.Set("xi-api-key", p.apiKey)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("elevenlabs speech: request: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		requestID := resp.Header.Get("x-request-id")
		if requestID == "" {
			requestID = resp.Header.Get("request-id")
		}
		msg := fmt.Sprintf("elevenlabs speech: unexpected status %d: %s", resp.StatusCode, string(respBody))
		if requestID != "" {
			msg += " [request_id=" + requestID + "]"
		}
		return nil, fmt.Errorf("%s", msg)
	}
	return resp.Body, nil
}
