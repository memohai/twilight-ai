package sdk

import "time"

type ResponseMetadata struct {
	ID        string            `json:"id,omitempty"`
	ModelID   string            `json:"modelId,omitempty"`
	Timestamp time.Time         `json:"timestamp,omitempty"`
	Headers   map[string]string `json:"headers,omitempty"`
}
