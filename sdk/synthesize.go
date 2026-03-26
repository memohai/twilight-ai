package sdk

import (
	"context"
	"fmt"
)

type speechConfig struct {
	Params SpeechParams
}

// SpeechOption configures a speech synthesis request.
type SpeechOption func(*speechConfig)

// WithSpeechModel sets the speech model for the request.
func WithSpeechModel(model *SpeechModel) SpeechOption {
	return func(c *speechConfig) { c.Params.Model = model }
}

// WithText sets the text to synthesize.
func WithText(text string) SpeechOption {
	return func(c *speechConfig) { c.Params.Text = text }
}

// WithSpeechConfig sets provider-specific configuration (e.g. voice, format, speed).
func WithSpeechConfig(cfg map[string]any) SpeechOption {
	return func(c *speechConfig) { c.Params.Config = cfg }
}

func buildSpeechConfig(options []SpeechOption) (*speechConfig, SpeechProvider, error) {
	cfg := &speechConfig{}
	for _, opt := range options {
		opt(cfg)
	}
	if cfg.Params.Model == nil {
		return nil, nil, fmt.Errorf("twilightai: speech model is required (use WithSpeechModel)")
	}
	if cfg.Params.Model.Provider == nil {
		return nil, nil, fmt.Errorf("twilightai: speech model %q has no provider", cfg.Params.Model.ID)
	}
	if cfg.Params.Text == "" {
		return nil, nil, fmt.Errorf("twilightai: text is required (use WithText)")
	}
	return cfg, cfg.Params.Model.Provider, nil
}

// GenerateSpeech synthesizes speech and returns the complete audio.
func (c *Client) GenerateSpeech(ctx context.Context, options ...SpeechOption) (*SpeechResult, error) {
	cfg, prov, err := buildSpeechConfig(options)
	if err != nil {
		return nil, err
	}
	return prov.DoSynthesize(ctx, cfg.Params)
}

// StreamSpeech synthesizes speech and returns a streaming result.
func (c *Client) StreamSpeech(ctx context.Context, options ...SpeechOption) (*SpeechStreamResult, error) {
	cfg, prov, err := buildSpeechConfig(options)
	if err != nil {
		return nil, err
	}
	return prov.DoStream(ctx, cfg.Params)
}
