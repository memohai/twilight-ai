package completions

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/memohai/twilight-ai/internal/utils"
	"github.com/memohai/twilight-ai/sdk"
)

const defaultBaseURL = "https://api.openai.com/v1"

type Provider struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

type Option func(*Provider)

func WithAPIKey(apiKey string) Option {
	return func(p *Provider) {
		p.apiKey = apiKey
	}
}

func WithBaseURL(baseURL string) Option {
	return func(p *Provider) {
		p.baseURL = baseURL
	}
}

func WithHTTPClient(client *http.Client) Option {
	return func(p *Provider) {
		p.httpClient = client
	}
}

func New(options ...Option) *Provider {
	provider := &Provider{
		baseURL:    defaultBaseURL,
		httpClient: &http.Client{},
	}
	for _, option := range options {
		option(provider)
	}
	return provider
}

func (p *Provider) Name() string {
	return "openai-completions"
}

func (p *Provider) ListModels(ctx context.Context) ([]sdk.Model, error) {
	resp, err := utils.FetchJSON[modelsListResponse](ctx, p.httpClient, &utils.RequestOptions{
		Method:  http.MethodGet,
		BaseURL: p.baseURL,
		Path:    "/models",
		Headers: utils.AuthHeader(p.apiKey),
	})
	if err != nil {
		return nil, fmt.Errorf("openai: list models request failed: %w", err)
	}

	models := make([]sdk.Model, 0, len(resp.Data))
	for _, m := range resp.Data {
		models = append(models, sdk.Model{
			ID:       m.ID,
			Provider: p,
			Type:     sdk.ModelTypeChat,
		})
	}
	return models, nil
}

func (p *Provider) Test(ctx context.Context) *sdk.ProviderTestResult {
	_, err := utils.FetchJSON[modelsListResponse](ctx, p.httpClient, &utils.RequestOptions{
		Method:  http.MethodGet,
		BaseURL: p.baseURL,
		Path:    "/models",
		Query:   map[string]string{"limit": "1"},
		Headers: utils.AuthHeader(p.apiKey),
	})
	if err != nil {
		return classifyError(err)
	}
	return &sdk.ProviderTestResult{Status: sdk.ProviderStatusOK, Message: "ok"}
}

func (p *Provider) TestModel(ctx context.Context, modelID string) (*sdk.ModelTestResult, error) {
	_, err := utils.FetchJSON[modelObject](ctx, p.httpClient, &utils.RequestOptions{
		Method:  http.MethodGet,
		BaseURL: p.baseURL,
		Path:    "/models/" + modelID,
		Headers: utils.AuthHeader(p.apiKey),
	})
	if err != nil {
		var apiErr *utils.APIError
		if errors.As(err, &apiErr) && apiErr.StatusCode == http.StatusNotFound {
			return &sdk.ModelTestResult{Supported: false, Message: "model not found"}, nil
		}
		return nil, fmt.Errorf("openai: test model request failed: %w", err)
	}
	return &sdk.ModelTestResult{Supported: true, Message: "supported"}, nil
}

// ChatModel creates a Model bound to this provider.
func (p *Provider) ChatModel(id string) *sdk.Model {
	return &sdk.Model{
		ID:       id,
		Provider: p,
		Type:     sdk.ModelTypeChat,
	}
}

// ---------- DoGenerate ----------

func (p *Provider) DoGenerate(ctx context.Context, params sdk.GenerateParams) (*sdk.GenerateResult, error) { //nolint:gocritic // interface method
	if params.Model == nil {
		return nil, fmt.Errorf("openai: model is required")
	}

	req := p.buildRequest(&params)

	resp, err := utils.FetchJSON[chatResponse](ctx, p.httpClient, &utils.RequestOptions{
		Method:  http.MethodPost,
		BaseURL: p.baseURL,
		Path:    "/chat/completions",
		Headers: utils.AuthHeader(p.apiKey),
		Body:    req,
	})
	if err != nil {
		var apiErr *utils.APIError
		if errors.As(err, &apiErr) {
			return nil, fmt.Errorf("openai: chat completions request failed: %s", apiErr.Detail())
		}
		return nil, fmt.Errorf("openai: chat completions request failed: %w", err)
	}

	return p.parseResponse(resp)
}

// ---------- buildRequest ----------

func (p *Provider) buildRequest(params *sdk.GenerateParams) *chatRequest {
	req := &chatRequest{
		Model:               params.Model.ID,
		Messages:            convertMessages(params),
		Temperature:         params.Temperature,
		TopP:                params.TopP,
		MaxCompletionTokens: params.MaxTokens,
		FrequencyPenalty:    params.FrequencyPenalty,
		PresencePenalty:     params.PresencePenalty,
		Seed:                params.Seed,
		ReasoningEffort:     params.ReasoningEffort,
	}
	if len(params.StopSequences) > 0 {
		req.Stop = params.StopSequences
	}
	if len(params.Tools) > 0 {
		req.Tools = convertTools(params.Tools)
		req.ToolChoice = params.ToolChoice
	}
	if params.ResponseFormat != nil {
		req.ResponseFormat = &chatRespFormat{
			Type:       string(params.ResponseFormat.Type),
			JSONSchema: params.ResponseFormat.JSONSchema,
		}
	}
	return req
}

func convertTools(tools []sdk.Tool) []chatTool {
	out := make([]chatTool, 0, len(tools))
	for _, t := range tools {
		out = append(out, chatTool{
			Type: "function",
			Function: chatFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
			},
		})
	}
	return out
}

// ---------- message conversion ----------

func convertMessages(params *sdk.GenerateParams) []chatMessage {
	var out []chatMessage

	if params.System != "" {
		out = append(out, chatMessage{
			Role:    "system",
			Content: params.System,
		})
	}

	for _, msg := range params.Messages {
		out = append(out, convertMessage(msg)...)
	}
	return out
}

func convertMessage(msg sdk.Message) []chatMessage {
	switch msg.Role {
	case sdk.MessageRoleTool:
		return convertToolResultMessages(msg)
	case sdk.MessageRoleAssistant:
		return []chatMessage{convertAssistantMessage(msg)}
	default:
		return []chatMessage{{
			Role:    string(msg.Role),
			Content: convertContent(msg.Content),
		}}
	}
}

func convertAssistantMessage(msg sdk.Message) chatMessage {
	cm := chatMessage{Role: "assistant"}

	var contentParts []sdk.MessagePart
	var toolCalls []chatToolCall
	var reasoning string

	for _, part := range msg.Content {
		switch p := part.(type) {
		case sdk.ToolCallPart:
			args, err := json.Marshal(p.Input)
			if err != nil {
				continue
			}
			id := p.ToolCallID
			if id == "" {
				id = generateID()
			}
			toolCalls = append(toolCalls, chatToolCall{
				ID:   id,
				Type: "function",
				Function: chatFunctionCall{
					Name:      p.ToolName,
					Arguments: string(args),
				},
			})
		case sdk.ReasoningPart:
			reasoning += p.Text
		default:
			contentParts = append(contentParts, part)
		}
	}

	if len(contentParts) > 0 {
		cm.Content = convertContent(contentParts)
	}
	if reasoning != "" {
		cm.ReasoningContent = reasoning
	}
	if len(toolCalls) > 0 {
		cm.ToolCalls = toolCalls
	}

	return cm
}

func convertToolResultMessages(msg sdk.Message) []chatMessage {
	var out []chatMessage
	for _, part := range msg.Content {
		if trp, ok := part.(sdk.ToolResultPart); ok {
			content, _ := json.Marshal(trp.Result)
			out = append(out, chatMessage{
				Role:       "tool",
				ToolCallID: trp.ToolCallID,
				Content:    string(content),
			})
		}
	}
	return out
}

func convertContent(parts []sdk.MessagePart) any {
	if len(parts) == 1 {
		if tp, ok := parts[0].(sdk.TextPart); ok {
			return tp.Text
		}
	}

	out := make([]any, 0, len(parts))
	for _, part := range parts {
		switch p := part.(type) {
		case sdk.TextPart:
			out = append(out, chatContentPartText{Type: "text", Text: p.Text})
		case sdk.ImagePart:
			out = append(out, chatContentPartImage{
				Type:     "image_url",
				ImageURL: chatImageURL{URL: p.Image},
			})
		case sdk.FilePart:
			out = append(out, chatContentPartText{Type: "text", Text: p.Data})
		}
	}
	return out
}

// ---------- parseResponse ----------

func (p *Provider) parseResponse(resp *chatResponse) (*sdk.GenerateResult, error) {
	result := &sdk.GenerateResult{
		Usage: convertUsage(&resp.Usage),
		Response: sdk.ResponseMetadata{
			ID:        resp.ID,
			ModelID:   resp.Model,
			Timestamp: time.Unix(resp.Created, 0),
		},
	}

	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]
		result.Text = choice.Message.Content
		result.Reasoning = reasoningFromMessage(&choice.Message)
		result.FinishReason = mapFinishReason(choice.FinishReason)
		result.RawFinishReason = choice.FinishReason

		for _, tc := range choice.Message.ToolCalls {
			var input any
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &input); err != nil {
				return result, fmt.Errorf("openai: unmarshal tool call arguments for %q: %w", tc.Function.Name, err)
			}
			id := tc.ID
			if id == "" {
				id = generateID()
			}
			result.ToolCalls = append(result.ToolCalls, sdk.ToolCall{
				ToolCallID: id,
				ToolName:   tc.Function.Name,
				Input:      input,
			})
		}
	}

	return result, nil
}

// ---------- DoStream ----------

func (p *Provider) DoStream(ctx context.Context, params sdk.GenerateParams) (*sdk.StreamResult, error) { //nolint:gocritic // interface method
	if params.Model == nil {
		return nil, fmt.Errorf("openai: model is required")
	}

	req := p.buildRequest(&params)
	req.Stream = true
	req.StreamOptions = &chatStreamOptions{IncludeUsage: true}

	ch := make(chan sdk.StreamPart, 64)

	go func() {
		defer close(ch)

		var (
			textStartSent      bool
			reasoningStartSent bool
			rawFinishReason    string
			finishReason       sdk.FinishReason
			usage              sdk.Usage
			chunkID            string
			chunkModel         string
			chunkCreated       int64
			flushed            bool
			pendingToolCalls   = map[int]*streamingToolCall{}
		)

		send := func(part sdk.StreamPart) bool {
			select {
			case ch <- part:
				return true
			case <-ctx.Done():
				return false
			}
		}

		flush := func() {
			if flushed {
				return
			}
			flushed = true
			if reasoningStartSent {
				send(&sdk.ReasoningEndPart{ID: chunkID})
				reasoningStartSent = false
			}
			if textStartSent {
				send(&sdk.TextEndPart{ID: chunkID})
				textStartSent = false
			}
			for _, stc := range pendingToolCalls {
				if stc.finished {
					continue
				}
				send(&sdk.ToolInputEndPart{ID: stc.id})
				var input any
				if err := json.Unmarshal([]byte(stc.args), &input); err != nil {
					send(&sdk.ErrorPart{Error: fmt.Errorf("openai: unmarshal tool call arguments for %q: %w", stc.name, err)})
				}
				send(&sdk.StreamToolCallPart{
					ToolCallID: stc.id,
					ToolName:   stc.name,
					Input:      input,
				})
				stc.finished = true
			}
		}

		if !send(&sdk.StartPart{}) {
			return
		}
		if !send(&sdk.StartStepPart{}) {
			return
		}

		err := utils.FetchSSE(ctx, p.httpClient, &utils.RequestOptions{
			Method:  http.MethodPost,
			BaseURL: p.baseURL,
			Path:    "/chat/completions",
			Headers: utils.AuthHeader(p.apiKey),
			Body:    req,
		}, func(ev *utils.SSEEvent) error {
			if ev.Data == "[DONE]" {
				return utils.ErrStreamDone
			}

			var chunk chatChunkResponse
			if err := json.Unmarshal([]byte(ev.Data), &chunk); err != nil {
				send(&sdk.ErrorPart{Error: fmt.Errorf("openai: unmarshal chunk: %w", err)})
				return err
			}

			if chunkID == "" {
				chunkID = chunk.ID
				chunkModel = chunk.Model
				chunkCreated = chunk.Created
			}

			if chunk.Usage != nil {
				usage = convertUsage(chunk.Usage)
			}

			if len(chunk.Choices) == 0 {
				return nil
			}
			choice := chunk.Choices[0]

			reasoningContent := reasoningFromDelta(&choice.Delta)
			if reasoningContent != "" {
				if !reasoningStartSent {
					send(&sdk.ReasoningStartPart{ID: chunk.ID})
					reasoningStartSent = true
				}
				send(&sdk.ReasoningDeltaPart{ID: chunk.ID, Text: reasoningContent})
			}

			if choice.Delta.Content != "" {
				if reasoningStartSent {
					send(&sdk.ReasoningEndPart{ID: chunk.ID})
					reasoningStartSent = false
				}
				if !textStartSent {
					send(&sdk.TextStartPart{ID: chunk.ID})
					textStartSent = true
				}
				send(&sdk.TextDeltaPart{ID: chunk.ID, Text: choice.Delta.Content})
			}

			if len(choice.Delta.ToolCalls) > 0 {
				if reasoningStartSent {
					send(&sdk.ReasoningEndPart{ID: chunk.ID})
					reasoningStartSent = false
				}
				if textStartSent {
					send(&sdk.TextEndPart{ID: chunk.ID})
					textStartSent = false
				}
			}

			for _, tc := range choice.Delta.ToolCalls {
				idx := tc.Index
				stc, exists := pendingToolCalls[idx]
				if !exists {
					id := tc.ID
					if id == "" {
						id = generateID()
					}
					stc = &streamingToolCall{id: id, name: tc.Function.Name}
					pendingToolCalls[idx] = stc
					send(&sdk.ToolInputStartPart{
						ID:       stc.id,
						ToolName: stc.name,
					})
				}
				if tc.Function.Arguments != "" {
					stc.args += tc.Function.Arguments
					send(&sdk.ToolInputDeltaPart{
						ID:    stc.id,
						Delta: tc.Function.Arguments,
					})

					if !stc.finished && json.Valid([]byte(stc.args)) {
						send(&sdk.ToolInputEndPart{ID: stc.id})
						var input any
						if err := json.Unmarshal([]byte(stc.args), &input); err != nil {
							send(&sdk.ErrorPart{Error: fmt.Errorf("openai: unmarshal tool call arguments for %q: %w", stc.name, err)})
						}
						send(&sdk.StreamToolCallPart{
							ToolCallID: stc.id,
							ToolName:   stc.name,
							Input:      input,
						})
						stc.finished = true
					}
				}
			}

			if choice.FinishReason != nil && *choice.FinishReason != "" {
				rawFinishReason = *choice.FinishReason
				finishReason = mapFinishReason(rawFinishReason)

				flush()

				send(&sdk.FinishStepPart{
					FinishReason:    finishReason,
					RawFinishReason: rawFinishReason,
					Usage:           usage,
					Response: sdk.ResponseMetadata{
						ID:        chunkID,
						ModelID:   chunkModel,
						Timestamp: time.Unix(chunkCreated, 0),
					},
				})
			}

			return nil
		})

		if err != nil {
			var apiErr *utils.APIError
			if errors.As(err, &apiErr) {
				send(&sdk.ErrorPart{Error: fmt.Errorf("openai: stream failed: %s", apiErr.Detail())})
			} else {
				send(&sdk.ErrorPart{Error: fmt.Errorf("openai: stream failed: %w", err)})
			}
		}

		flush()

		send(&sdk.FinishPart{
			FinishReason:    finishReason,
			RawFinishReason: rawFinishReason,
			TotalUsage:      usage,
		})
	}()

	return &sdk.StreamResult{Stream: ch}, nil
}

type streamingToolCall struct {
	id       string
	name     string
	args     string
	finished bool
}

// ---------- helpers ----------

func reasoningFromMessage(m *chatRespMessage) string {
	if m.ReasoningContent != "" {
		return m.ReasoningContent
	}
	return m.Reasoning
}

func reasoningFromDelta(d *chatChunkDelta) string {
	if d.ReasoningContent != "" {
		return d.ReasoningContent
	}
	return d.Reasoning
}

func convertUsage(u *chatUsage) sdk.Usage {
	usage := sdk.Usage{
		InputTokens:  u.PromptTokens,
		OutputTokens: u.CompletionTokens,
		TotalTokens:  u.TotalTokens,
	}
	if u.PromptTokensDetails != nil {
		usage.CachedInputTokens = u.PromptTokensDetails.CachedTokens
		usage.InputTokenDetails.CacheReadTokens = u.PromptTokensDetails.CachedTokens
	}
	if u.CompletionTokensDetails != nil {
		usage.ReasoningTokens = u.CompletionTokensDetails.ReasoningTokens
		usage.OutputTokenDetails.ReasoningTokens = u.CompletionTokensDetails.ReasoningTokens
		usage.OutputTokenDetails.TextTokens = u.CompletionTokensDetails.TextTokens
	}
	return usage
}

func mapFinishReason(reason string) sdk.FinishReason {
	switch reason {
	case "stop":
		return sdk.FinishReasonStop
	case "length":
		return sdk.FinishReasonLength
	case "content_filter":
		return sdk.FinishReasonContentFilter
	case "tool_calls":
		return sdk.FinishReasonToolCalls
	default:
		return sdk.FinishReasonUnknown
	}
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
