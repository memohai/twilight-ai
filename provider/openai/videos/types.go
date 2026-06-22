package videos

type videoResponse struct {
	ID          string      `json:"id"`
	Object      string      `json:"object,omitempty"`
	CreatedAt   int64       `json:"created_at,omitempty"`
	CompletedAt int64       `json:"completed_at,omitempty"`
	Status      string      `json:"status"`
	Model       string      `json:"model,omitempty"`
	Progress    *float64    `json:"progress,omitempty"`
	Seconds     any         `json:"seconds,omitempty"`
	Size        string      `json:"size,omitempty"`
	Error       *videoError `json:"error,omitempty"`
}

type videoError struct {
	Code    string `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
}
