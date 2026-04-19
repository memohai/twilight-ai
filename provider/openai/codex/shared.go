package codex

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/memohai/twilight-ai/internal/utils"
	"github.com/memohai/twilight-ai/sdk"
)

const (
	defaultBaseURL         = "https://chatgpt.com/backend-api"
	defaultOriginator      = "codex_cli_rs"
	openAIBetaHeader       = "OpenAI-Beta"
	openAIBetaValue        = "responses=experimental"
	openAIAccountHeader    = "chatgpt-account-id"
	openAIOriginatorHeader = "originator"
	openAIAuthClaimPath    = "https://api.openai.com/auth"
)

type streamingToolCall struct {
	id       string
	name     string
	args     string
	finished bool
}

func generateID() string {
	b := make([]byte, 12)
	if _, err := rand.Read(b); err != nil {
		panic("openai-codex: generateID entropy failure: " + err.Error())
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

func accountIDFromToken(token string) (string, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return "", fmt.Errorf("invalid codex access token")
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return "", fmt.Errorf("decode codex token payload: %w", err)
	}
	var claims struct {
		OpenAIAuth struct {
			ChatGPTAccountID string `json:"chatgpt_account_id"`
		} `json:"https://api.openai.com/auth"`
	}
	if err := json.Unmarshal(payload, &claims); err != nil {
		return "", fmt.Errorf("parse codex token payload: %w", err)
	}
	accountID := strings.TrimSpace(claims.OpenAIAuth.ChatGPTAccountID)
	if accountID == "" {
		return "", fmt.Errorf("codex access token missing %s.chatgpt_account_id", openAIAuthClaimPath)
	}
	return accountID, nil
}
