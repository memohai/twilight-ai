package sdk

import (
	"context"
	"fmt"
)

type transcriptionConfig struct {
	Params TranscriptionParams
}

// TranscriptionOption configures a transcription request.
type TranscriptionOption func(*transcriptionConfig)

// WithTranscriptionModel sets the transcription model for the request.
func WithTranscriptionModel(model *TranscriptionModel) TranscriptionOption {
	return func(c *transcriptionConfig) { c.Params.Model = model }
}

// WithAudio sets the audio payload for the request.
func WithAudio(data []byte, filename, contentType string) TranscriptionOption {
	return func(c *transcriptionConfig) {
		c.Params.Audio = data
		c.Params.Filename = filename
		c.Params.ContentType = contentType
	}
}

// WithTranscriptionConfig sets provider-specific configuration.
func WithTranscriptionConfig(cfg map[string]any) TranscriptionOption {
	return func(c *transcriptionConfig) { c.Params.Config = cfg }
}

func buildTranscriptionConfig(options []TranscriptionOption) (*transcriptionConfig, TranscriptionProvider, error) {
	cfg := &transcriptionConfig{}
	for _, opt := range options {
		opt(cfg)
	}
	if cfg.Params.Model == nil {
		return nil, nil, fmt.Errorf("twilightai: transcription model is required (use WithTranscriptionModel)")
	}
	if cfg.Params.Model.Provider == nil {
		return nil, nil, fmt.Errorf("twilightai: transcription model %q has no provider", cfg.Params.Model.ID)
	}
	if len(cfg.Params.Audio) == 0 {
		return nil, nil, fmt.Errorf("twilightai: audio is required (use WithAudio)")
	}
	if cfg.Params.Filename == "" {
		cfg.Params.Filename = "audio.wav"
	}
	return cfg, cfg.Params.Model.Provider, nil
}

// Transcribe converts audio to text.
func (c *Client) Transcribe(ctx context.Context, options ...TranscriptionOption) (*TranscriptionResult, error) {
	cfg, prov, err := buildTranscriptionConfig(options)
	if err != nil {
		return nil, err
	}
	return prov.DoTranscribe(ctx, cfg.Params)
}
