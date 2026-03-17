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

var defaultClient = &Client{}
