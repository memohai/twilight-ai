package sdk

type InputTokenDetail struct {
	NoCacheTokens    int `json:"noCacheTokens"`
	CacheReadTokens  int `json:"cacheReadTokens"`
	CacheWriteTokens int `json:"cacheWriteTokens"`
	// CacheWrite5mTokens is the number of tokens written to the 5-minute cache
	// (Anthropic-specific, populated when using cache_control with default TTL).
	CacheWrite5mTokens int `json:"cacheWrite5mTokens,omitempty"`
	// CacheWrite1hTokens is the number of tokens written to the 1-hour cache
	// (Anthropic-specific, populated when using cache_control with ttl="1h").
	CacheWrite1hTokens int `json:"cacheWrite1hTokens,omitempty"`
}

type OutputTokenDetail struct {
	TextTokens      int `json:"textTokens"`
	ReasoningTokens int `json:"reasoningTokens"`
}

type Usage struct {
	InputTokens        int               `json:"inputTokens"`
	OutputTokens       int               `json:"outputTokens"`
	TotalTokens        int               `json:"totalTokens"`
	ReasoningTokens    int               `json:"reasoningTokens,omitempty"`
	CachedInputTokens  int               `json:"cachedInputTokens,omitempty"`
	InputTokenDetails  InputTokenDetail  `json:"inputTokenDetails,omitempty"`
	OutputTokenDetails OutputTokenDetail `json:"outputTokenDetails,omitempty"`
}
