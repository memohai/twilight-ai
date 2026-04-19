package sdk

import "context"

// TranscriptionProvider is the interface that speech-to-text backends must implement.
type TranscriptionProvider interface {
	ListModels(ctx context.Context) ([]*TranscriptionModel, error)
	DoTranscribe(ctx context.Context, params TranscriptionParams) (*TranscriptionResult, error)
}

// TranscriptionModel represents a transcription model bound to a TranscriptionProvider.
type TranscriptionModel struct {
	ID       string
	Provider TranscriptionProvider
}

// TranscriptionParams holds the parameters for a transcription request.
// Config is provider-specific (e.g. prompt, language, diarization settings).
type TranscriptionParams struct {
	Model       *TranscriptionModel
	Audio       []byte
	Filename    string
	ContentType string
	Config      map[string]any
}

// TranscriptionWord represents a word-level alignment item when a provider returns it.
type TranscriptionWord struct {
	Text      string
	Start     float64
	End       float64
	SpeakerID string
}

// TranscriptionResult holds the unified result of a transcription request.
type TranscriptionResult struct {
	Text             string
	Language         string
	DurationSeconds  float64
	Words            []TranscriptionWord
	ProviderMetadata map[string]any
}
