package copilot

// AutoModel tells the provider to let GitHub choose the backing Copilot model.
// GitHub's Copilot agent endpoint does not expose a public models discovery API,
// so this sentinel keeps the SDK API explicit without guessing an upstream model ID.
const AutoModel = "copilot-auto"

// ModelDescriptor describes a GitHub Copilot chat model entry exposed by this provider.
type ModelDescriptor struct {
	ID          string
	DisplayName string
}

// Catalog returns the static model directory supported by this provider.
func Catalog() []ModelDescriptor {
	return []ModelDescriptor{
		{
			ID:          AutoModel,
			DisplayName: "GitHub-managed Copilot model",
		},
	}
}
