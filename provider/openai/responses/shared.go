package responses

import (
	"crypto/rand"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/memohai/twilight-ai/internal/utils"
	"github.com/memohai/twilight-ai/sdk"
)

// streamingToolCall accumulates one function call's argument deltas. args
// uses strings.Builder because it grows by one small delta per SSE event;
// instances are always held by pointer (pendingToolCalls map).
type streamingToolCall struct {
	id       string
	name     string
	args     strings.Builder
	finished bool
}

func generateID() string {
	b := make([]byte, 12)
	if _, err := rand.Read(b); err != nil {
		panic("openai-responses: generateID entropy failure: " + err.Error())
	}
	return fmt.Sprintf("call_%x", b)
}

func classifyError(err error) *sdk.ProviderTestResult {
	var apiErr *utils.APIError
	if errors.As(err, &apiErr) {
		if apiErr.StatusCode == http.StatusUnauthorized || apiErr.StatusCode == http.StatusForbidden {
			return &sdk.ProviderTestResult{
				Status:  sdk.ProviderStatusUnhealthy,
				Message: fmt.Sprintf("authentication failed: %s", apiErr.Message),
				Error:   err,
			}
		}
		return &sdk.ProviderTestResult{
			Status:  sdk.ProviderStatusUnhealthy,
			Message: fmt.Sprintf("service error (%d): %s", apiErr.StatusCode, apiErr.Message),
			Error:   err,
		}
	}
	return &sdk.ProviderTestResult{
		Status:  sdk.ProviderStatusUnreachable,
		Message: fmt.Sprintf("connection failed: %s", err.Error()),
		Error:   err,
	}
}
