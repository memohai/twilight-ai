package sdk

type Source struct {
	SourceType       string         `json:"sourceType"`
	ID               string         `json:"id"`
	URL              string         `json:"url"`
	Title            string         `json:"title,omitempty"`
	ProviderMetadata map[string]any `json:"providerMetadata,omitempty"`
}
