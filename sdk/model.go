package sdk

type ModelType string

const (
	ModelTypeChat ModelType = "chat"
)

type Model struct {
	ID          string
	DisplayName string
	Provider    Provider
	Type        ModelType
	MaxTokens   int
}
