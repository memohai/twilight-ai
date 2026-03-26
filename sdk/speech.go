package sdk

import "context"

// SpeechProvider is the interface that speech synthesis backends must implement.
type SpeechProvider interface {
	DoSynthesize(ctx context.Context, params SpeechParams) (*SpeechResult, error)
	DoStream(ctx context.Context, params SpeechParams) (*SpeechStreamResult, error)
}

// SpeechModel represents a speech model bound to a SpeechProvider.
type SpeechModel struct {
	ID       string
	Provider SpeechProvider
}

// SpeechParams holds the parameters for a speech synthesis request.
// Config is open-ended and provider-specific (e.g. voice, format, speed).
type SpeechParams struct {
	Model  *SpeechModel
	Text   string
	Config map[string]any
}

// SpeechResult holds the result of a non-streaming synthesis.
type SpeechResult struct {
	Audio       []byte
	ContentType string // MIME type, e.g. "audio/mpeg"
}

// SpeechStreamResult holds a channel that yields raw audio chunks.
// The channel is closed when the stream ends.
type SpeechStreamResult struct {
	Stream      <-chan []byte
	ContentType string // MIME type, e.g. "audio/mpeg"
	errCh       <-chan error
}

// Bytes consumes the entire stream and returns concatenated audio data.
func (r *SpeechStreamResult) Bytes() ([]byte, error) {
	var out []byte
	for chunk := range r.Stream {
		out = append(out, chunk...)
	}
	if r.errCh != nil {
		if err, ok := <-r.errCh; ok && err != nil {
			return out, err
		}
	}
	return out, nil
}

// NewSpeechStreamResult creates a SpeechStreamResult from data and error channels.
func NewSpeechStreamResult(stream <-chan []byte, contentType string, errCh <-chan error) *SpeechStreamResult {
	return &SpeechStreamResult{
		Stream:      stream,
		ContentType: contentType,
		errCh:       errCh,
	}
}
