package generativeai

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/memohai/twilight-ai/internal/utils"
	"github.com/memohai/twilight-ai/sdk"
)

const defaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"

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
	return "google-generative-ai"
}

func (p *Provider) ListModels(ctx context.Context) ([]sdk.Model, error) {
	resp, err := utils.FetchJSON[googleModelsListResponse](ctx, p.httpClient, &utils.RequestOptions{
		Method:  http.MethodGet,
		BaseURL: p.baseURL,
		Path:    "/models",
		Headers: p.authHeaders(),
	})
	if err != nil {
		return nil, fmt.Errorf("google: list models request failed: %w", err)
	}

	models := make([]sdk.Model, 0, len(resp.Models))
	for _, m := range resp.Models {
		id := strings.TrimPrefix(m.Name, "models/")
		models = append(models, sdk.Model{
			ID:          id,
			DisplayName: m.DisplayName,
			Provider:    p,
			Type:        sdk.ModelTypeChat,
		})
	}
	return models, nil
}

func (p *Provider) Test(ctx context.Context) *sdk.ProviderTestResult {
	_, err := utils.FetchJSON[googleModelsListResponse](ctx, p.httpClient, &utils.RequestOptions{
		Method:  http.MethodGet,
		BaseURL: p.baseURL,
		Path:    "/models",
		Query:   map[string]string{"pageSize": "1"},
		Headers: p.authHeaders(),
	})
	if err != nil {
		return classifyError(err)
	}
	return &sdk.ProviderTestResult{Status: sdk.ProviderStatusOK, Message: "ok"}
}

func (p *Provider) TestModel(ctx context.Context, modelID string) (*sdk.ModelTestResult, error) {
	modelPath := getModelPath(modelID)
	_, err := utils.FetchJSON[googleModelObject](ctx, p.httpClient, &utils.RequestOptions{
		Method:  http.MethodGet,
		BaseURL: p.baseURL,
		Path:    "/" + modelPath,
		Headers: p.authHeaders(),
	})
	if err != nil {
		var apiErr *utils.APIError
		if errors.As(err, &apiErr) && apiErr.StatusCode == http.StatusNotFound {
			return &sdk.ModelTestResult{Supported: false, Message: "model not found"}, nil
		}
		return nil, fmt.Errorf("google: test model request failed: %w", err)
	}
	return &sdk.ModelTestResult{Supported: true, Message: "supported"}, nil
}

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
		return nil, fmt.Errorf("google: model is required")
	}

	req := p.buildRequest(&params)
	modelPath := getModelPath(params.Model.ID)

	resp, err := utils.FetchJSON[generateResponse](ctx, p.httpClient, &utils.RequestOptions{
		Method:  http.MethodPost,
		BaseURL: p.baseURL,
		Path:    "/" + modelPath + ":generateContent",
		Headers: p.authHeaders(),
		Body:    req,
	})
	if err != nil {
		var apiErr *utils.APIError
		if errors.As(err, &apiErr) {
			return nil, fmt.Errorf("google: generateContent request failed: %s", apiErr.Detail())
		}
		return nil, fmt.Errorf("google: generateContent request failed: %w", err)
	}

	return p.parseResponse(resp)
}

// ---------- buildRequest ----------

func (p *Provider) buildRequest(params *sdk.GenerateParams) *generateRequest {
	contents, sysInstruction := convertMessages(params)

	req := &generateRequest{
		Contents:          contents,
		SystemInstruction: sysInstruction,
	}

	genCfg := &generationConfig{
		Temperature:      params.Temperature,
		TopP:             params.TopP,
		MaxOutputTokens:  params.MaxTokens,
		FrequencyPenalty: params.FrequencyPenalty,
		PresencePenalty:  params.PresencePenalty,
		Seed:             params.Seed,
	}
	if len(params.StopSequences) > 0 {
		genCfg.StopSequences = params.StopSequences
	}
	if params.ResponseFormat != nil {
		switch params.ResponseFormat.Type {
		case sdk.ResponseFormatJSONObject, sdk.ResponseFormatJSONSchema:
			genCfg.ResponseMimeType = "application/json"
			if params.ResponseFormat.JSONSchema != nil {
				genCfg.ResponseSchema = params.ResponseFormat.JSONSchema
			}
		}
	}
	req.GenerationConfig = genCfg

	if len(params.Tools) > 0 {
		tools, toolCfg := convertTools(params.Tools, params.ToolChoice)
		req.Tools = tools
		req.ToolConfig = toolCfg
	}

	return req
}

// ---------- message conversion ----------

func convertMessages(params *sdk.GenerateParams) ([]content, *content) { //nolint:gocritic // unnamed results are clear in context
	contents := make([]content, 0, len(params.Messages))
	var sysInstruction *content

	if params.System != "" {
		sysInstruction = &content{
			Parts: []contentPart{{Text: params.System}},
		}
	}

	for _, msg := range params.Messages {
		contents = append(contents, convertMessage(msg)...)
	}
	return contents, sysInstruction
}

func convertMessage(msg sdk.Message) []content {
	switch msg.Role {
	case sdk.MessageRoleSystem:
		return nil
	case sdk.MessageRoleAssistant:
		return []content{convertAssistantMessage(msg)}
	case sdk.MessageRoleTool:
		return []content{convertToolResultMessage(msg)}
	default:
		return []content{convertUserMessage(msg)}
	}
}

func convertUserMessage(msg sdk.Message) content {
	var parts []contentPart
	for _, part := range msg.Content {
		switch p := part.(type) {
		case sdk.TextPart:
			parts = append(parts, contentPart{Text: p.Text})
		case sdk.ImagePart:
			if strings.HasPrefix(p.Image, "data:") || strings.Contains(p.Image, ";base64,") {
				mediaType, data := parseDataURI(p.Image)
				if mediaType == "" {
					mediaType = p.MediaType
				}
				if mediaType == "" {
					mediaType = "image/jpeg"
				}
				parts = append(parts, contentPart{
					InlineData: &inlineData{MimeType: mediaType, Data: data},
				})
			} else {
				mediaType := p.MediaType
				if mediaType == "" {
					mediaType = "image/jpeg"
				}
				parts = append(parts, contentPart{
					FileData: &fileData{MimeType: mediaType, FileURI: p.Image},
				})
			}
		case sdk.FilePart:
			mediaType := p.MediaType
			if mediaType == "" {
				mediaType = "application/octet-stream"
			}
			parts = append(parts, contentPart{
				InlineData: &inlineData{MimeType: mediaType, Data: p.Data},
			})
		}
	}
	return content{Role: "user", Parts: parts}
}

func convertAssistantMessage(msg sdk.Message) content {
	var parts []contentPart
	for _, part := range msg.Content {
		switch p := part.(type) {
		case sdk.TextPart:
			if p.Text != "" {
				cp := contentPart{Text: p.Text}
				if sig := extractGoogleThoughtSignature(p.ProviderMetadata); sig != "" {
					cp.ThoughtSignature = sig
				}
				parts = append(parts, cp)
			}
		case sdk.ReasoningPart:
			if p.Text != "" {
				thought := true
				cp := contentPart{
					Text:    p.Text,
					Thought: &thought,
				}
				if sig := extractGoogleThoughtSignature(p.ProviderMetadata); sig != "" {
					cp.ThoughtSignature = sig
				}
				parts = append(parts, cp)
			}
		case sdk.ToolCallPart:
			cp := contentPart{
				FunctionCall: &functionCall{
					Name: p.ToolName,
					Args: p.Input,
				},
			}
			if sig := extractGoogleThoughtSignature(p.ProviderMetadata); sig != "" {
				cp.ThoughtSignature = sig
			}
			parts = append(parts, cp)
		}
	}
	return content{Role: "model", Parts: parts}
}

func convertToolResultMessage(msg sdk.Message) content {
	var parts []contentPart
	for _, part := range msg.Content {
		if trp, ok := part.(sdk.ToolResultPart); ok {
			parts = append(parts, contentPart{
				FunctionResponse: &functionResponse{
					Name: trp.ToolName,
					Response: functionResponseVal{
						Name:    trp.ToolName,
						Content: trp.Result,
					},
				},
			})
		}
	}
	return content{Role: "user", Parts: parts}
}

// ---------- tool conversion ----------

func convertTools(tools []sdk.Tool, toolChoice any) ([]toolGroup, *toolConfig) {
	decls := make([]functionDeclaration, 0, len(tools))
	for _, t := range tools {
		decls = append(decls, functionDeclaration{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.Parameters,
		})
	}

	var tc *toolConfig
	if toolChoice != nil {
		if choice, ok := toolChoice.(string); ok {
			switch choice {
			case "auto":
				tc = &toolConfig{FunctionCallingConfig: &functionCallingConfig{Mode: "AUTO"}}
			case "none":
				tc = &toolConfig{FunctionCallingConfig: &functionCallingConfig{Mode: "NONE"}}
			case "required":
				tc = &toolConfig{FunctionCallingConfig: &functionCallingConfig{Mode: "ANY"}}
			}
		}
	}

	return []toolGroup{{FunctionDeclarations: decls}}, tc
}

// ---------- parseResponse ----------

func (p *Provider) parseResponse(resp *generateResponse) (*sdk.GenerateResult, error) {
	result := &sdk.GenerateResult{
		Response: sdk.ResponseMetadata{
			ModelID: "",
		},
	}

	if resp.UsageMetadata != nil {
		result.Usage = convertUsage(resp.UsageMetadata)
	}

	if len(resp.Candidates) == 0 {
		result.FinishReason = sdk.FinishReasonOther
		result.RawFinishReason = ""
		return result, nil
	}

	candidate := resp.Candidates[0]
	hasToolCalls := false

	if candidate.Content != nil {
		for _, part := range candidate.Content.Parts {
			switch {
			case part.FunctionCall != nil:
				hasToolCalls = true
				id := generateID()
				argsJSON, err := json.Marshal(part.FunctionCall.Args)
				if err != nil {
					return result, fmt.Errorf("google: marshal function call args for %q: %w", part.FunctionCall.Name, err)
				}
				var input any
				if err := json.Unmarshal(argsJSON, &input); err != nil {
					return result, fmt.Errorf("google: unmarshal function call args for %q: %w", part.FunctionCall.Name, err)
				}
				result.ToolCalls = append(result.ToolCalls, sdk.ToolCall{
					ToolCallID: id,
					ToolName:   part.FunctionCall.Name,
					Input:      input,
				})
			case part.Text != "":
				isThought := part.Thought != nil && *part.Thought
				if isThought {
					result.Reasoning += part.Text
					if part.ThoughtSignature != "" {
						result.ReasoningProviderMetadata = map[string]any{
							"google": map[string]any{"thoughtSignature": part.ThoughtSignature},
						}
					}
				} else {
					result.Text += part.Text
				}
			case part.InlineData != nil:
				result.Files = append(result.Files, sdk.GeneratedFile{
					Data:      part.InlineData.Data,
					MediaType: part.InlineData.MimeType,
				})
			}
		}
	}

	result.FinishReason = mapFinishReason(candidate.FinishReason, hasToolCalls)
	result.RawFinishReason = candidate.FinishReason

	return result, nil
}

// ---------- DoStream ----------

func (p *Provider) DoStream(ctx context.Context, params sdk.GenerateParams) (*sdk.StreamResult, error) { //nolint:gocritic // interface method
	if params.Model == nil {
		return nil, fmt.Errorf("google: model is required")
	}

	req := p.buildRequest(&params)
	modelPath := getModelPath(params.Model.ID)

	ch := make(chan sdk.StreamPart, 64)

	go func() {
		defer close(ch)

		var (
			textStartSent      bool
			reasoningStartSent bool
			rawFinishReason    string
			finishReason       sdk.FinishReason
			usage              sdk.Usage
			flushed            bool
			hasToolCalls       bool
			blockCounter       int
			currentTextID      string
			currentReasoningID string
			lastThoughtSig     string
		)

		send := func(part sdk.StreamPart) bool {
			select {
			case ch <- part:
				return true
			case <-ctx.Done():
				return false
			}
		}

		reasoningEndMeta := func() map[string]any {
			if lastThoughtSig == "" {
				return nil
			}
			return map[string]any{
				"google": map[string]any{"thoughtSignature": lastThoughtSig},
			}
		}

		flush := func() {
			if flushed {
				return
			}
			flushed = true
			if reasoningStartSent {
				send(&sdk.ReasoningEndPart{ID: currentReasoningID, ProviderMetadata: reasoningEndMeta()})
				reasoningStartSent = false
			}
			if textStartSent {
				send(&sdk.TextEndPart{ID: currentTextID})
				textStartSent = false
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
			Path:    "/" + modelPath + ":streamGenerateContent",
			Query:   map[string]string{"alt": "sse"},
			Headers: p.authHeaders(),
			Body:    req,
		}, func(ev *utils.SSEEvent) error {
			var chunk generateResponse
			if err := json.Unmarshal([]byte(ev.Data), &chunk); err != nil {
				send(&sdk.ErrorPart{Error: fmt.Errorf("google: unmarshal chunk: %w", err)})
				return err
			}

			if chunk.UsageMetadata != nil {
				usage = convertUsage(chunk.UsageMetadata)
			}

			if len(chunk.Candidates) == 0 {
				return nil
			}
			candidate := chunk.Candidates[0]

			if candidate.Content != nil {
				for _, part := range candidate.Content.Parts {
					switch {
					case part.FunctionCall != nil:
						if reasoningStartSent {
							send(&sdk.ReasoningEndPart{ID: currentReasoningID, ProviderMetadata: reasoningEndMeta()})
							reasoningStartSent = false
						}
						if textStartSent {
							send(&sdk.TextEndPart{ID: currentTextID})
							textStartSent = false
						}

						hasToolCalls = true
						toolCallID := generateID()
						argsJSON, _ := json.Marshal(part.FunctionCall.Args)
						argsStr := string(argsJSON)

						send(&sdk.ToolInputStartPart{
							ID:       toolCallID,
							ToolName: part.FunctionCall.Name,
						})
						send(&sdk.ToolInputDeltaPart{
							ID:    toolCallID,
							Delta: argsStr,
						})
						send(&sdk.ToolInputEndPart{ID: toolCallID})

						var input any
						if err := json.Unmarshal(argsJSON, &input); err != nil {
							_ = err // unmarshal failed, input remains nil
						}

						send(&sdk.StreamToolCallPart{
							ToolCallID: toolCallID,
							ToolName:   part.FunctionCall.Name,
							Input:      input,
						})
					case part.Text != "":
						isThought := part.Thought != nil && *part.Thought
						if isThought {
							if part.ThoughtSignature != "" {
								lastThoughtSig = part.ThoughtSignature
							}
							if textStartSent {
								send(&sdk.TextEndPart{ID: currentTextID})
								textStartSent = false
							}
							if !reasoningStartSent {
								currentReasoningID = fmt.Sprintf("%d", blockCounter)
								blockCounter++
								send(&sdk.ReasoningStartPart{ID: currentReasoningID})
								reasoningStartSent = true
							}
							send(&sdk.ReasoningDeltaPart{ID: currentReasoningID, Text: part.Text})
						} else {
							if reasoningStartSent {
								send(&sdk.ReasoningEndPart{ID: currentReasoningID, ProviderMetadata: reasoningEndMeta()})
								reasoningStartSent = false
							}
							if !textStartSent {
								currentTextID = fmt.Sprintf("%d", blockCounter)
								blockCounter++
								send(&sdk.TextStartPart{ID: currentTextID})
								textStartSent = true
							}
							send(&sdk.TextDeltaPart{ID: currentTextID, Text: part.Text})
						}
					case part.InlineData != nil:
						if textStartSent {
							send(&sdk.TextEndPart{ID: currentTextID})
							textStartSent = false
						}
						if reasoningStartSent {
							send(&sdk.ReasoningEndPart{ID: currentReasoningID, ProviderMetadata: reasoningEndMeta()})
							reasoningStartSent = false
						}
						send(&sdk.StreamFilePart{
							File: sdk.GeneratedFile{
								Data:      part.InlineData.Data,
								MediaType: part.InlineData.MimeType,
							},
						})
					}
				}
			}

			if candidate.FinishReason != "" {
				rawFinishReason = candidate.FinishReason
				finishReason = mapFinishReason(rawFinishReason, hasToolCalls)

				flush()

				send(&sdk.FinishStepPart{
					FinishReason:    finishReason,
					RawFinishReason: rawFinishReason,
					Usage:           usage,
					Response:        sdk.ResponseMetadata{},
				})
			}

			return nil
		})

		if err != nil {
			var apiErr *utils.APIError
			if errors.As(err, &apiErr) {
				send(&sdk.ErrorPart{Error: fmt.Errorf("google: stream failed: %s", apiErr.Detail())})
			} else {
				send(&sdk.ErrorPart{Error: fmt.Errorf("google: stream failed: %w", err)})
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

// ---------- helpers ----------

func getModelPath(modelID string) string {
	if strings.Contains(modelID, "/") {
		return modelID
	}
	return "models/" + modelID
}

func (p *Provider) authHeaders() map[string]string {
	return map[string]string{
		"x-goog-api-key": p.apiKey,
	}
}

func generateID() string {
	b := make([]byte, 12)
	rand.Read(b)
	return fmt.Sprintf("call_%x", b)
}

func convertUsage(u *usageMetadata) sdk.Usage {
	candidateTokens := u.CandidatesTokenCount
	thoughtTokens := u.ThoughtsTokenCount
	cachedTokens := u.CachedContentTokenCount

	return sdk.Usage{
		InputTokens:       u.PromptTokenCount,
		OutputTokens:      candidateTokens + thoughtTokens,
		TotalTokens:       u.TotalTokenCount,
		ReasoningTokens:   thoughtTokens,
		CachedInputTokens: cachedTokens,
		InputTokenDetails: sdk.InputTokenDetail{
			NoCacheTokens:   u.PromptTokenCount - cachedTokens,
			CacheReadTokens: cachedTokens,
		},
		OutputTokenDetails: sdk.OutputTokenDetail{
			TextTokens:      candidateTokens,
			ReasoningTokens: thoughtTokens,
		},
	}
}

func mapFinishReason(reason string, hasToolCalls bool) sdk.FinishReason {
	switch reason {
	case "STOP":
		if hasToolCalls {
			return sdk.FinishReasonToolCalls
		}
		return sdk.FinishReasonStop
	case "MAX_TOKENS":
		return sdk.FinishReasonLength
	case "SAFETY", "RECITATION", "IMAGE_SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII":
		return sdk.FinishReasonContentFilter
	case "MALFORMED_FUNCTION_CALL":
		return sdk.FinishReasonError
	default:
		return sdk.FinishReasonOther
	}
}

func extractGoogleThoughtSignature(meta map[string]any) string {
	if meta == nil {
		return ""
	}
	gm, ok := meta["google"].(map[string]any)
	if !ok {
		return ""
	}
	sig, _ := gm["thoughtSignature"].(string)
	return sig
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

func parseDataURI(uri string) (mediaType, data string) {
	idx := strings.Index(uri, ",")
	if idx < 0 {
		return "", uri
	}
	header := uri[:idx]
	data = uri[idx+1:]
	header = strings.TrimPrefix(header, "data:")
	header = strings.TrimSuffix(header, ";base64")
	return header, data
}
