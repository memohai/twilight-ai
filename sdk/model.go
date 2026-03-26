package sdk

import "context"

type ModelType string

const (
	ModelTypeChat ModelType = "chat"
)

type Model struct {
	ID          string
	DisplayName string
	Provider    Provider
	Type        ModelType
}

// Test checks whether this model is supported by its provider.
func (m *Model) Test(ctx context.Context) (*ModelTestResult, error) {
	return m.Provider.TestModel(ctx, m.ID)
}
