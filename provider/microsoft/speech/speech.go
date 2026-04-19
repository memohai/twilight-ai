// Package speech provides a Microsoft Azure Cognitive Services Text-to-Speech
// provider.  It targets the REST endpoint
// https://{region}.tts.speech.microsoft.com/cognitiveservices/v1
// and authenticates with an Ocp-Apim-Subscription-Key header.
//
// The request body is a minimal SSML document; style, rate, and pitch
// adjustments are embedded via <mstts:express-as> and <prosody> elements.
package speech

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/memohai/twilight-ai/internal/utils"
	sdk "github.com/memohai/twilight-ai/sdk"
)

const (
	defaultModelID      = "microsoft-tts"
	defaultVoice        = "en-US-JennyNeural"
	defaultOutputFormat = "audio-16khz-128kbitrate-mono-mp3"
	// ttsPath is appended to the region-specific base URL.
	ttsPath = "/cognitiveservices/v1"
)

// Option configures the Microsoft Azure TTS provider.
type Option func(*Provider)

// WithAPIKey sets the Azure subscription key (Ocp-Apim-Subscription-Key).
func WithAPIKey(key string) Option {
	return func(p *Provider) { p.apiKey = key }
}

// WithBaseURL overrides the full TTS base URL (useful for testing).
// When set, the region from Config is ignored for URL construction.
func WithBaseURL(u string) Option {
	return func(p *Provider) { p.baseURL = strings.TrimRight(u, "/") }
}

// WithHTTPClient replaces the default HTTP client.
func WithHTTPClient(hc *http.Client) Option {
	return func(p *Provider) { p.httpClient = hc }
}

// Provider implements sdk.SpeechProvider for Microsoft Azure TTS.
type Provider struct {
	apiKey     string
	baseURL    string // overrides region-derived URL when non-empty
	httpClient *http.Client
}

// New creates a new Microsoft Azure TTS provider.
func New(opts ...Option) *Provider {
	p := &Provider{httpClient: &http.Client{}}
	for _, o := range opts {
		o(p)
	}
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
	return nil, fmt.Errorf("microsoft speech: provider does not expose a remote models discovery API")
}

// DoSynthesize synthesizes speech and returns the complete audio bytes.
func (p *Provider) DoSynthesize(ctx context.Context, params sdk.SpeechParams) (*sdk.SpeechResult, error) {
	cfg := parseConfig(params.Config)

	body, err := p.doRequest(ctx, params.Text, &cfg)
	if err != nil {
		return nil, err
	}
	defer body.Close()

	audio, err := io.ReadAll(body)
	if err != nil {
		return nil, fmt.Errorf("microsoft speech: read response: %w", err)
	}
	return &sdk.SpeechResult{
		Audio:       audio,
		ContentType: contentTypeForFormat(cfg.OutputFormat),
	}, nil
}

// DoStream synthesizes speech and returns a streaming result.
// Azure TTS returns chunked audio so chunks are forwarded as they arrive.
func (p *Provider) DoStream(ctx context.Context, params sdk.SpeechParams) (*sdk.SpeechStreamResult, error) {
	cfg := parseConfig(params.Config)

	body, err := p.doRequest(ctx, params.Text, &cfg)
	if err != nil {
		return nil, err
	}

	ch, errCh := utils.StreamHTTPBody(ctx, body, "microsoft speech")
	return sdk.NewSpeechStreamResult(ch, contentTypeForFormat(cfg.OutputFormat), errCh), nil
}

// endpointURL returns the TTS REST endpoint.
// Priority: explicit WithBaseURL override → region from Config.
func (p *Provider) endpointURL(cfg *audioConfig) (string, error) {
	if p.baseURL != "" {
		return p.baseURL + ttsPath, nil
	}
	if cfg.Region == "" {
		return "", fmt.Errorf("microsoft speech: region is required (set via Config[\"region\"] or WithBaseURL)")
	}
	return fmt.Sprintf("https://%s.tts.speech.microsoft.com%s", cfg.Region, ttsPath), nil
}

// doRequest builds and sends the SSML POST request.
func (p *Provider) doRequest(ctx context.Context, text string, cfg *audioConfig) (io.ReadCloser, error) {
	endpoint, err := p.endpointURL(cfg)
	if err != nil {
		return nil, err
	}

	ssml := buildSSML(text, cfg)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewBufferString(ssml))
	if err != nil {
		return nil, fmt.Errorf("microsoft speech: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/ssml+xml")
	req.Header.Set("X-Microsoft-OutputFormat", cfg.OutputFormat)
	req.Header.Set("Ocp-Apim-Subscription-Key", p.apiKey)
	req.Header.Set("User-Agent", "twilight-ai")

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("microsoft speech: request: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		return nil, fmt.Errorf("microsoft speech: unexpected status %d: %s", resp.StatusCode, string(respBody))
	}
	return resp.Body, nil
}
