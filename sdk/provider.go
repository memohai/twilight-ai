package sdk

import "context"

// Provider is the interface that AI backends must implement.
type Provider interface {
	Name() string
	GetModels() ([]Model, error)
	DoGenerate(ctx context.Context, params GenerateParams) (*GenerateResult, error)
	DoStream(ctx context.Context, params GenerateParams) (*StreamResult, error)
}
