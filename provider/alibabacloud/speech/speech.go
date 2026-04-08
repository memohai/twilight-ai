// Package speech provides an Alibaba Cloud DashScope CosyVoice TTS provider.
// It communicates over WebSocket using the run-task / continue-task / finish-task
// protocol described in the DashScope CosyVoice WebSocket API documentation.
package speech

import (
	"context"
	"strings"

	sdk "github.com/memohai/twilight-ai/sdk"
)

const (
	defaultModelID    = "cosyvoice-tts"
	defaultBaseURL    = "wss://dashscope.aliyuncs.com/api-ws/v1/inference/"
	defaultModel      = "cosyvoice-v1"
	defaultFormat     = "mp3"
	defaultSampleRate = 22050
	contentTypeAudio  = "audio/mpeg"
)

// Option configures the DashScope CosyVoice TTS provider.
type Option func(*Provider)

// WithAPIKey sets the DashScope API key used as a Bearer credential during the WebSocket handshake.
func WithAPIKey(key string) Option {
	return func(p *Provider) { p.client.apiKey = key }
}

// WithBaseURL overrides the WebSocket endpoint URL (useful for testing).
func WithBaseURL(url string) Option {
	return func(p *Provider) { p.client.baseURL = url }
}

// Provider implements sdk.SpeechProvider for DashScope CosyVoice TTS.
type Provider struct {
	client *wsClient
}

// New creates a new DashScope CosyVoice TTS provider.
func New(opts ...Option) *Provider {
	p := &Provider{
		client: newWSClient("", defaultBaseURL),
	}
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

// DoSynthesize synthesizes speech and returns the complete audio bytes.
func (p *Provider) DoSynthesize(ctx context.Context, params sdk.SpeechParams) (*sdk.SpeechResult, error) {
	cfg := parseConfig(params.Config)

	audio, err := p.client.synthesize(ctx, params.Text, &cfg)
	if err != nil {
		return nil, err
	}
	return &sdk.SpeechResult{
		Audio:       audio,
		ContentType: contentTypeForFormat(cfg.Format),
	}, nil
}

// DoStream synthesizes speech and returns a streaming result.
func (p *Provider) DoStream(ctx context.Context, params sdk.SpeechParams) (*sdk.SpeechStreamResult, error) {
	cfg := parseConfig(params.Config)

	ch, errCh := p.client.stream(ctx, params.Text, &cfg)
	return sdk.NewSpeechStreamResult(ch, contentTypeForFormat(cfg.Format), errCh), nil
}

func contentTypeForFormat(format string) string {
	switch strings.ToLower(format) {
	case "mp3":
		return "audio/mpeg"
	case "wav":
		return "audio/wav"
	case "pcm":
		return "audio/pcm"
	case "opus":
		return "audio/opus"
	default:
		return contentTypeAudio
	}
}
