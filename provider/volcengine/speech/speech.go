// Package speech provides a Volcengine SAMI TTS provider.
//
// Authentication uses a two-step process:
//  1. Call GetToken via open.volcengineapi.com with Volcengine V4 HMAC-SHA256 signing
//     (access_key + secret_key + app_key).
//  2. Use the returned token to invoke POST https://sami.bytedance.com/api/v1/invoke.
//
// The TTS response is a JSON object where the audio is base64-encoded in the "data" field.
// Streaming is emulated by returning the fully synthesized audio as a single chunk.
package speech

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	sdk "github.com/memohai/twilight-ai/sdk"
)

const (
	defaultModelID    = "sami-tts"
	defaultBaseURL    = "https://sami.bytedance.com"
	defaultEncoding   = "mp3"
	defaultSampleRate = 24000
	contentTypeAudio  = "audio/mpeg"

	samiAPIVersion = "v4"
	samiNamespace  = "TTS"
)

// Option configures the Volcengine SAMI TTS provider.
type Option func(*Provider)

// WithAccessKey sets the Volcengine AccessKeyID.
func WithAccessKey(key string) Option {
	return func(p *Provider) { p.accessKey = key }
}

// WithSecretKey sets the Volcengine SecretAccessKey.
func WithSecretKey(key string) Option {
	return func(p *Provider) { p.secretKey = key }
}

// WithAppKey sets the SAMI application AppKey.
func WithAppKey(key string) Option {
	return func(p *Provider) { p.appKey = key }
}

// WithBaseURL overrides the SAMI API base URL (useful for testing).
func WithBaseURL(u string) Option {
	return func(p *Provider) { p.baseURL = u }
}

// WithHTTPClient replaces the default HTTP client.
func WithHTTPClient(hc *http.Client) Option {
	return func(p *Provider) { p.httpClient = hc }
}

// WithToken sets a static pre-obtained SAMI token, bypassing the GetToken call.
// Useful for testing where you already have a valid token.
func WithToken(token string) Option {
	return func(p *Provider) { p.staticToken = token }
}

// Provider implements sdk.SpeechProvider for Volcengine SAMI TTS.
type Provider struct {
	accessKey   string
	secretKey   string
	appKey      string
	baseURL     string
	httpClient  *http.Client
	staticToken string // for testing
	tokenCache  tokenCache
}

// New creates a new Volcengine SAMI TTS provider.
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
		id = defaultModelID
	}
	return &sdk.SpeechModel{ID: id, Provider: p}
}

// ListModels returns the speech models exposed by this provider.
func (p *Provider) ListModels(context.Context) ([]*sdk.SpeechModel, error) {
	return nil, fmt.Errorf("volcengine speech: provider does not expose a remote models discovery API in this SDK")
}

// invokeResponse is the JSON structure returned by SAMI /api/v1/invoke.
type invokeResponse struct {
	StatusCode int32   `json:"status_code"`
	StatusText string  `json:"status_text"`
	TaskID     string  `json:"task_id"`
	Namespace  string  `json:"namespace"`
	Data       []byte  `json:"data"` // raw bytes; JSON unmarshaler decodes base64 automatically
	Payload    *string `json:"payload,omitempty"`
}

// DoSynthesize synthesizes speech and returns the complete audio bytes.
func (p *Provider) DoSynthesize(ctx context.Context, params sdk.SpeechParams) (*sdk.SpeechResult, error) {
	cfg := parseConfig(params.Config)

	audio, err := p.synthesize(ctx, params.Text, cfg)
	if err != nil {
		return nil, err
	}
	return &sdk.SpeechResult{
		Audio:       audio,
		ContentType: contentTypeForEncoding(cfg.Encoding),
	}, nil
}

// DoStream returns a streaming result containing the fully synthesized audio as a single chunk.
// SAMI does not expose a public non-SDK streaming endpoint, so DoStream wraps DoSynthesize.
func (p *Provider) DoStream(ctx context.Context, params sdk.SpeechParams) (*sdk.SpeechStreamResult, error) {
	cfg := parseConfig(params.Config)

	audio, err := p.synthesize(ctx, params.Text, cfg)
	if err != nil {
		return nil, err
	}

	ch := make(chan []byte, 1)
	errCh := make(chan error, 1)

	ch <- audio
	close(ch)
	close(errCh)

	return sdk.NewSpeechStreamResult(ch, contentTypeForEncoding(cfg.Encoding), errCh), nil
}

// synthesize calls the SAMI TTS API and returns decoded audio bytes.
func (p *Provider) synthesize(ctx context.Context, text string, cfg audioConfig) ([]byte, error) {
	token, err := p.resolveToken(ctx)
	if err != nil {
		return nil, err
	}

	innerPayload := map[string]any{
		"text":    text,
		"speaker": cfg.Speaker,
		"audio_config": map[string]any{
			"format":      cfg.Encoding,
			"sample_rate": cfg.SampleRate,
			"speech_rate": cfg.SpeechRate,
			"pitch_rate":  cfg.PitchRate,
		},
	}
	innerPayloadJSON, err := json.Marshal(innerPayload)
	if err != nil {
		return nil, fmt.Errorf("volcengine speech: marshal inner payload: %w", err)
	}

	reqBody := map[string]any{
		"payload": string(innerPayloadJSON),
	}
	reqBodyJSON, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("volcengine speech: marshal request body: %w", err)
	}

	invokeURL := fmt.Sprintf("%s/api/v1/invoke?version=%s&token=%s&appkey=%s&namespace=%s",
		p.baseURL, samiAPIVersion,
		token, p.appKey, samiNamespace)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, invokeURL, bytes.NewReader(reqBodyJSON))
	if err != nil {
		return nil, fmt.Errorf("volcengine speech: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("volcengine speech: request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("volcengine speech: unexpected status %d: %s", resp.StatusCode, string(b))
	}

	// The Go JSON unmarshaler automatically base64-decodes []byte fields.
	var result invokeResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("volcengine speech: decode response: %w", err)
	}
	if result.StatusCode != 0 && result.StatusCode != 20000000 {
		return nil, fmt.Errorf("volcengine speech: api error %d: %s", result.StatusCode, result.StatusText)
	}
	if len(result.Data) == 0 {
		return nil, fmt.Errorf("volcengine speech: empty audio in response")
	}
	return result.Data, nil
}

// resolveToken returns a valid SAMI token, obtaining a new one if needed.
func (p *Provider) resolveToken(ctx context.Context) (string, error) {
	if p.staticToken != "" {
		return p.staticToken, nil
	}
	if token, ok := p.tokenCache.get(); ok {
		return token, nil
	}
	token, expiresAt, err := getToken(ctx, p.accessKey, p.secretKey, p.appKey, p.httpClient)
	if err != nil {
		return "", err
	}
	p.tokenCache.set(token, expiresAt)
	return token, nil
}

func contentTypeForEncoding(encoding string) string {
	switch strings.ToLower(encoding) {
	case "mp3":
		return "audio/mpeg"
	case "wav":
		return "audio/wav"
	case "aac":
		return "audio/aac"
	default:
		return contentTypeAudio
	}
}
