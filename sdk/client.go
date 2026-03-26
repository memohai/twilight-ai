package sdk

import "context"

// Client provides text generation methods.
// The provider is resolved from the Model passed via WithModel.
type Client struct{}

func NewClient() *Client {
	return &Client{}
}

// --- Package-level convenience functions ---

func GenerateText(ctx context.Context, options ...GenerateOption) (string, error) {
	return defaultClient.GenerateText(ctx, options...)
}

func GenerateTextResult(ctx context.Context, options ...GenerateOption) (*GenerateResult, error) {
	return defaultClient.GenerateTextResult(ctx, options...)
}

func StreamText(ctx context.Context, options ...GenerateOption) (*StreamResult, error) {
	return defaultClient.StreamText(ctx, options...)
}

// --- Embedding convenience functions ---

func Embed(ctx context.Context, value string, options ...EmbedOption) ([]float64, error) {
	return defaultClient.Embed(ctx, value, options...)
}

func EmbedMany(ctx context.Context, values []string, options ...EmbedOption) (*EmbedResult, error) {
	return defaultClient.EmbedMany(ctx, values, options...)
}

// --- Speech convenience functions ---

func GenerateSpeech(ctx context.Context, options ...SpeechOption) (*SpeechResult, error) {
	return defaultClient.GenerateSpeech(ctx, options...)
}

func StreamSpeech(ctx context.Context, options ...SpeechOption) (*SpeechStreamResult, error) {
	return defaultClient.StreamSpeech(ctx, options...)
}

var defaultClient = &Client{}
