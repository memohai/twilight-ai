package responses

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

const (
	defaultBaseURL = "https://api.openai.com/v1"

	// Output item types for OpenAI Responses API
	outputTypeMessage      = "message"
	outputTypeReasoning    = "reasoning"
	outputTypeFunctionCall = "function_call"
)

type Provider struct {
	apiKey         string
	baseURL        string
	httpClient     *http.Client
	prepareRequest func(*http.Request) error
}

type Option func(*Provider)

func WithAPIKey(apiKey string) Option {
	return func(p *Provider) { p.apiKey = apiKey }
}

func WithBaseURL(baseURL string) Option {
	return func(p *Provider) { p.baseURL = baseURL }
}

func WithHTTPClient(client *http.Client) Option {
	return func(p *Provider) { p.httpClient = client }
}

// WithBedrockRegion enables AWS SigV4 authentication for Amazon Bedrock's
// OpenAI-compatible Responses endpoint using the default AWS credential chain.
func WithBedrockRegion(region string) Option {
	return func(p *Provider) {
		p.prepareRequest = utils.NewBedrockDefaultCredentialsPreparer(region)
	}
}

// WithBedrockCredentials enables AWS SigV4 authentication for Amazon Bedrock's
// OpenAI-compatible Responses endpoint using static credentials.
func WithBedrockCredentials(region, accessKeyID, secretAccessKey, sessionToken string) Option {
	return func(p *Provider) {
		p.prepareRequest = utils.NewBedrockStaticCredentialsPreparer(region, accessKeyID, secretAccessKey, sessionToken)
	}
}

func New(options ...Option) *Provider {
	p := &Provider{
		baseURL:    defaultBaseURL,
		httpClient: &http.Client{},
	}
	for _, o := range options {
		o(p)
	}
	return p
}

func (p *Provider) Name() string { return "openai-responses" }

func (p *Provider) ListModels(ctx context.Context) ([]sdk.Model, error) {
	resp, err := utils.FetchJSON[modelsListResponse](ctx, p.httpClient, &utils.RequestOptions{
		Method:   http.MethodGet,
		BaseURL:  p.baseURL,
		Path:     "/models",
		Headers:  p.authHeaders(),
		Prepare:  p.prepareRequest,
	})
	if err != nil {
		return nil, fmt.Errorf("openai-responses: list models request failed: %w", err)
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
		Method:   http.MethodGet,
		BaseURL:  p.baseURL,
		Path:     "/models",
		Query:    map[string]string{"limit": "1"},
		Headers:  p.authHeaders(),
		Prepare:  p.prepareRequest,
	})
	if err != nil {
		return classifyError(err)
	}
	return &sdk.ProviderTestResult{Status: sdk.ProviderStatusOK, Message: "ok"}
}

func (p *Provider) TestModel(ctx context.Context, modelID string) (*sdk.ModelTestResult, error) {
	_, err := utils.FetchJSON[modelObject](ctx, p.httpClient, &utils.RequestOptions{
		Method:   http.MethodGet,
		BaseURL:  p.baseURL,
		Path:     "/models/" + modelID,
		Headers:  p.authHeaders(),
		Prepare:  p.prepareRequest,
	})
	if err == nil {
		return &sdk.ModelTestResult{Supported: true, Message: "supported"}, nil
	}
	var apiErr *utils.APIError
	if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusNotFound {
		return nil, fmt.Errorf("openai-responses: test model request failed: %w", err)
	}

	status, probeErr := utils.ProbeStatus(ctx, p.httpClient, &utils.RequestOptions{
		Method:   http.MethodPost,
		BaseURL:  p.baseURL,
		Path:     "/responses",
		Headers:  p.authHeaders(),
		Prepare:  p.prepareRequest,
		Body: map[string]any{
			"model":             modelID,
			"input":             "hi",
			"max_output_tokens": 1,
		},
	})
	if probeErr != nil {
		return nil, fmt.Errorf("openai-responses: probe model request failed: %w", probeErr)
	}
	return sdk.ClassifyProbeStatus(status)
}

func (p *Provider) ChatModel(id string) *sdk.Model {
	return &sdk.Model{
		ID:       id,
		Provider: p,
		Type:     sdk.ModelTypeChat,
	}
}

func (p *Provider) authHeaders() map[string]string {
	if p.prepareRequest != nil {
		return nil
	}
	if p.apiKey == "" {
		return nil
	}
	return utils.AuthHeader(p.apiKey)
}

// ---------- DoGenerate ----------

func (p *Provider) DoGenerate(ctx context.Context, params sdk.GenerateParams) (*sdk.GenerateResult, error) { //nolint:gocritic // interface method
	if params.Model == nil {
		return nil, fmt.Errorf("openai-responses: model is required")
	}

	req := p.buildRequest(&params)

	resp, err := utils.FetchJSON[responsesResponse](ctx, p.httpClient, &utils.RequestOptions{
		Method:   http.MethodPost,
		BaseURL:  p.baseURL,
		Path:     "/responses",
		Headers:  p.authHeaders(),
		Prepare:  p.prepareRequest,
		Body:     req,
	})
	if err != nil {
		var apiErr *utils.APIError
		if errors.As(err, &apiErr) {
			return nil, fmt.Errorf("openai-responses: request failed: %s", apiErr.Detail())
		}
		return nil, fmt.Errorf("openai-responses: request failed: %w", err)
	}

	if resp.Error != nil {
		return nil, fmt.Errorf("openai-responses: api error [%s]: %s", resp.Error.Code, resp.Error.Message)
	}

	return p.parseResponse(resp)
}

// ---------- buildRequest ----------

func (p *Provider) buildRequest(params *sdk.GenerateParams) *responsesRequest {
	req := &responsesRequest{
		Model:           params.Model.ID,
		Input:           convertToResponsesInput(params),
		Temperature:     params.Temperature,
		TopP:            params.TopP,
		MaxOutputTokens: params.MaxTokens,
	}

	if len(params.Tools) > 0 {
		req.Tools = convertResponsesTools(params.Tools)
		req.ToolChoice = params.ToolChoice
	}

	if params.ResponseFormat != nil {
		tf := &responsesTextFmt{}
		switch params.ResponseFormat.Type {
		case sdk.ResponseFormatJSONObject:
			tf.Format = &responsesTextFormat{Type: "json_object"}
		case sdk.ResponseFormatJSONSchema:
			tf.Format = &responsesTextFormat{
				Type:   "json_schema",
				Name:   "response",
				Schema: params.ResponseFormat.JSONSchema,
			}
		}
		if tf.Format != nil {
			req.Text = tf
		}
	}

	if params.ReasoningEffort != nil {
		req.Reasoning = &responsesReasoning{Effort: *params.ReasoningEffort}
	}

	return req
}

func convertResponsesTools(tools []sdk.Tool) []responsesTool {
	out := make([]responsesTool, 0, len(tools))
	for _, t := range tools {
		out = append(out, responsesTool{
			Type:        "function",
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.Parameters,
		})
	}
	return out
}

// ---------- input conversion ----------

func convertToResponsesInput(params *sdk.GenerateParams) []json.RawMessage {
	var items []json.RawMessage

	if params.System != "" {
		items = appendRaw(items, responsesSystemMessage{
			Role:    "system",
			Content: params.System,
		})
	}

	for _, msg := range params.Messages {
		items = append(items, convertResponsesMessage(msg)...)
	}
	return items
}

func convertResponsesMessage(msg sdk.Message) []json.RawMessage {
	switch msg.Role {
	case sdk.MessageRoleSystem:
		return []json.RawMessage{marshalRaw(responsesSystemMessage{
			Role:    "system",
			Content: textFromParts(msg.Content),
		})}

	case sdk.MessageRoleUser:
		return convertResponsesUserMessage(msg)

	case sdk.MessageRoleAssistant:
		return convertResponsesAssistantMessage(msg)

	case sdk.MessageRoleTool:
		return convertResponsesToolResults(msg)

	default:
		return nil
	}
}

func convertResponsesUserMessage(msg sdk.Message) []json.RawMessage {
	var parts []responsesUserContentPart
	for _, part := range msg.Content {
		switch p := part.(type) {
		case sdk.TextPart:
			parts = append(parts, responsesUserContentPart{Type: "input_text", Text: p.Text})
		case sdk.ImagePart:
			parts = append(parts, responsesUserContentPart{Type: "input_image", ImageURL: p.Image})
		case sdk.FilePart:
			parts = append(parts, responsesUserContentPart{Type: "input_text", Text: p.Data})
		}
	}
	return []json.RawMessage{marshalRaw(responsesUserMessage{
		Role:    "user",
		Content: parts,
	})}
}

func convertResponsesAssistantMessage(msg sdk.Message) []json.RawMessage {
	var items []json.RawMessage
	var textParts []responsesOutputTextPart
	var reasoningSummary []responsesReasoningSummaryText
	var encryptedContent string

	for _, part := range msg.Content {
		switch p := part.(type) {
		case sdk.TextPart:
			textParts = append(textParts, responsesOutputTextPart{
				Type: "output_text",
				Text: p.Text,
			})

		case sdk.ReasoningPart:
			reasoningSummary = append(reasoningSummary, responsesReasoningSummaryText{
				Type: "summary_text",
				Text: p.Text,
			})
			if ec := extractOpenAIEncryptedContent(p.ProviderMetadata); ec != "" {
				encryptedContent = ec
			}

		case sdk.ToolCallPart:
			args, err := json.Marshal(p.Input)
			if err != nil {
				continue
			}
			id := p.ToolCallID
			if id == "" {
				id = generateID()
			}
			items = appendRaw(items, responsesFunctionCall{
				Type:      "function_call",
				CallID:    id,
				Name:      p.ToolName,
				Arguments: string(args),
			})
		}
	}

	var prefix []json.RawMessage
	if len(reasoningSummary) > 0 {
		ri := responsesReasoningItem{
			Type:             "reasoning",
			Summary:          reasoningSummary,
			EncryptedContent: encryptedContent,
		}
		prefix = append(prefix, marshalRaw(ri))
	}
	if len(textParts) > 0 {
		prefix = append(prefix, marshalRaw(responsesAssistantMessage{
			Role:    "assistant",
			Content: textParts,
		}))
	}
	items = append(prefix, items...)

	return items
}

func convertResponsesToolResults(msg sdk.Message) []json.RawMessage {
	var items []json.RawMessage
	for _, part := range msg.Content {
		if trp, ok := part.(sdk.ToolResultPart); ok {
			output, _ := json.Marshal(trp.Result)
			items = appendRaw(items, responsesFunctionCallOutput{
				Type:   "function_call_output",
				CallID: trp.ToolCallID,
				Output: string(output),
			})
		}
	}
	return items
}

// ---------- parseResponse ----------

func (p *Provider) parseResponse(resp *responsesResponse) (*sdk.GenerateResult, error) {
	result := &sdk.GenerateResult{
		Response: sdk.ResponseMetadata{
			ID:        resp.ID,
			ModelID:   resp.Model,
			Timestamp: time.Unix(resp.CreatedAt, 0),
		},
	}

	if resp.Usage != nil {
		result.Usage = convertResponsesUsage(resp.Usage)
	}

	hasFunctionCall := false
	var incompleteReason string
	if resp.IncompleteDetails != nil {
		incompleteReason = resp.IncompleteDetails.Reason
	}

	for i := range resp.Output {
		item := &resp.Output[i]
		switch item.Type {
		case outputTypeMessage:
			for _, c := range item.Content {
				if c.Type == "output_text" {
					result.Text += c.Text
				}
				for _, ann := range c.Annotations {
					if ann.Type == "url_citation" {
						result.Sources = append(result.Sources, sdk.Source{
							SourceType: "url",
							ID:         generateID(),
							URL:        ann.URL,
							Title:      ann.Title,
						})
					}
				}
			}

		case outputTypeReasoning:
			for _, s := range item.Summary {
				if s.Type == "summary_text" {
					result.Reasoning += s.Text
				}
			}
			if item.EncryptedContent != "" {
				result.ReasoningProviderMetadata = map[string]any{
					"openai": map[string]any{
						"reasoningEncryptedContent": item.EncryptedContent,
						"itemId":                    item.ID,
					},
				}
			}

		case outputTypeFunctionCall:
			hasFunctionCall = true
			var input any
			if err := json.Unmarshal([]byte(item.Arguments), &input); err != nil {
				return result, fmt.Errorf("openai-responses: unmarshal tool call arguments for %q: %w", item.Name, err)
			}
			callID := item.CallID
			if callID == "" {
				callID = generateID()
			}
			result.ToolCalls = append(result.ToolCalls, sdk.ToolCall{
				ToolCallID: callID,
				ToolName:   item.Name,
				Input:      input,
			})
		}
	}

	result.FinishReason = mapResponsesFinishReason(incompleteReason, hasFunctionCall)
	if incompleteReason != "" {
		result.RawFinishReason = incompleteReason
	}

	return result, nil
}

// ---------- DoStream ----------

func (p *Provider) DoStream(ctx context.Context, params sdk.GenerateParams) (*sdk.StreamResult, error) { //nolint:gocritic,gocyclo // interface method
	if params.Model == nil {
		return nil, fmt.Errorf("openai-responses: model is required")
	}

	req := p.buildRequest(&params)
	req.Stream = true

	ch := make(chan sdk.StreamPart, 64)

	go func() {
		defer close(ch)

		var (
			responseID       string
			responseModel    string
			responseCreated  int64
			usage            sdk.Usage
			incompleteReason string
			hasFunctionCall  bool

			textStartSent      bool
			reasoningStartSent bool

			// Track ongoing function calls by output_index
			pendingToolCalls = map[int]*streamingToolCall{}
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
			if reasoningStartSent {
				send(&sdk.ReasoningEndPart{ID: responseID})
				reasoningStartSent = false
			}
			if textStartSent {
				send(&sdk.TextEndPart{ID: responseID})
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
			Method:   http.MethodPost,
			BaseURL:  p.baseURL,
			Path:     "/responses",
			Headers:  p.authHeaders(),
			Prepare:  p.prepareRequest,
			Body:     req,
		}, func(ev *utils.SSEEvent) error {
			eventType := ev.Event
			if eventType == "" {
				// Try to infer event type from the data if event field is missing
				var probe struct {
					Type string `json:"type"`
				}
				if json.Unmarshal([]byte(ev.Data), &probe) == nil && probe.Type != "" {
					eventType = probe.Type
				}
			}

			switch eventType {
			case "response.created":
				var chunk responsesCreatedChunk
				if err := json.Unmarshal([]byte(ev.Data), &chunk); err != nil {
					return nil
				}
				responseID = chunk.Response.ID
				responseModel = chunk.Response.Model
				responseCreated = chunk.Response.CreatedAt

			case "response.output_item.added":
				var chunk responsesOutputItemAddedChunk
				if err := json.Unmarshal([]byte(ev.Data), &chunk); err != nil {
					return nil
				}
				switch chunk.Item.Type {
				case outputTypeMessage:
					if !textStartSent {
						send(&sdk.TextStartPart{ID: chunk.Item.ID})
						textStartSent = true
					}
				case outputTypeReasoning:
					if !reasoningStartSent {
						var meta map[string]any
						if chunk.Item.EncryptedContent != "" {
							meta = map[string]any{
								"openai": map[string]any{
									"reasoningEncryptedContent": chunk.Item.EncryptedContent,
									"itemId":                    chunk.Item.ID,
								},
							}
						}
						send(&sdk.ReasoningStartPart{ID: chunk.Item.ID, ProviderMetadata: meta})
						reasoningStartSent = true
					}
				case outputTypeFunctionCall:
					if reasoningStartSent {
						send(&sdk.ReasoningEndPart{ID: responseID})
						reasoningStartSent = false
					}
					if textStartSent {
						send(&sdk.TextEndPart{ID: responseID})
						textStartSent = false
					}
					callID := chunk.Item.CallID
					if callID == "" {
						callID = generateID()
					}
					pendingToolCalls[chunk.OutputIndex] = &streamingToolCall{
						id:   callID,
						name: chunk.Item.Name,
					}
					send(&sdk.ToolInputStartPart{
						ID:       callID,
						ToolName: chunk.Item.Name,
					})
				}

			case "response.output_text.delta":
				var chunk responsesTextDeltaChunk
				if err := json.Unmarshal([]byte(ev.Data), &chunk); err != nil {
					return nil
				}
				if reasoningStartSent {
					send(&sdk.ReasoningEndPart{ID: responseID})
					reasoningStartSent = false
				}
				if !textStartSent {
					send(&sdk.TextStartPart{ID: chunk.ItemID})
					textStartSent = true
				}
				send(&sdk.TextDeltaPart{ID: chunk.ItemID, Text: chunk.Delta})

			case "response.reasoning_summary_text.delta":
				var chunk responsesReasoningSummaryDeltaChunk
				if err := json.Unmarshal([]byte(ev.Data), &chunk); err != nil {
					return nil
				}
				if !reasoningStartSent {
					send(&sdk.ReasoningStartPart{ID: chunk.ItemID})
					reasoningStartSent = true
				}
				send(&sdk.ReasoningDeltaPart{ID: chunk.ItemID, Text: chunk.Delta})

			case "response.function_call_arguments.delta":
				var chunk responsesFuncArgsDeltaChunk
				if err := json.Unmarshal([]byte(ev.Data), &chunk); err != nil {
					return nil
				}
				stc := pendingToolCalls[chunk.OutputIndex]
				if stc == nil {
					return nil
				}
				stc.args += chunk.Delta
				send(&sdk.ToolInputDeltaPart{
					ID:    stc.id,
					Delta: chunk.Delta,
				})

			case "response.output_item.done":
				var chunk responsesOutputItemDoneChunk
				if err := json.Unmarshal([]byte(ev.Data), &chunk); err != nil {
					return nil
				}
				switch chunk.Item.Type {
				case outputTypeMessage:
					if textStartSent {
						send(&sdk.TextEndPart{ID: chunk.Item.ID})
						textStartSent = false
					}
				case outputTypeReasoning:
					if reasoningStartSent {
						send(&sdk.ReasoningEndPart{ID: chunk.Item.ID})
						reasoningStartSent = false
					}
				case outputTypeFunctionCall:
					hasFunctionCall = true
					stc := pendingToolCalls[chunk.OutputIndex]
					if stc != nil && !stc.finished {
						send(&sdk.ToolInputEndPart{ID: stc.id})
						args := chunk.Item.Arguments
						if args == "" {
							args = stc.args
						}
						var input any
						if err := json.Unmarshal([]byte(args), &input); err != nil {
							send(&sdk.ErrorPart{Error: fmt.Errorf("openai-responses: unmarshal tool call arguments for %q: %w", stc.name, err)})
						}
						send(&sdk.StreamToolCallPart{
							ToolCallID: stc.id,
							ToolName:   stc.name,
							Input:      input,
						})
						stc.finished = true
					}
				}

			case "response.output_text.annotation.added":
				var chunk responsesAnnotationAddedChunk
				if err := json.Unmarshal([]byte(ev.Data), &chunk); err != nil {
					return nil
				}
				if chunk.Annotation.Type == "url_citation" {
					send(&sdk.StreamSourcePart{
						Source: sdk.Source{
							SourceType: "url",
							ID:         generateID(),
							URL:        chunk.Annotation.URL,
							Title:      chunk.Annotation.Title,
						},
					})
				}

			case "response.completed", "response.incomplete":
				var chunk responsesCompletedChunk
				if err := json.Unmarshal([]byte(ev.Data), &chunk); err != nil {
					return nil
				}
				if chunk.Response.IncompleteDetails != nil {
					incompleteReason = chunk.Response.IncompleteDetails.Reason
				}
				if chunk.Response.Usage != nil {
					usage = convertResponsesUsage(chunk.Response.Usage)
				}

				flush()

				finishReason := mapResponsesFinishReason(incompleteReason, hasFunctionCall)
				send(&sdk.FinishStepPart{
					FinishReason:    finishReason,
					RawFinishReason: incompleteReason,
					Usage:           usage,
					Response: sdk.ResponseMetadata{
						ID:        responseID,
						ModelID:   responseModel,
						Timestamp: time.Unix(responseCreated, 0),
					},
				})

				return utils.ErrStreamDone

			case "error":
				var chunk responsesErrorChunk
				if err := json.Unmarshal([]byte(ev.Data), &chunk); err != nil {
					return nil
				}
				send(&sdk.ErrorPart{Error: fmt.Errorf("openai-responses: %s: %s", chunk.Error.Code, chunk.Error.Message)})
				return utils.ErrStreamDone
			}

			return nil
		})

		if err != nil {
			var apiErr *utils.APIError
			if errors.As(err, &apiErr) {
				send(&sdk.ErrorPart{Error: fmt.Errorf("openai-responses: stream failed: %s", apiErr.Detail())})
			} else {
				send(&sdk.ErrorPart{Error: fmt.Errorf("openai-responses: stream failed: %w", err)})
			}
		}

		flush()

		finishReason := mapResponsesFinishReason(incompleteReason, hasFunctionCall)
		send(&sdk.FinishPart{
			FinishReason:    finishReason,
			RawFinishReason: incompleteReason,
			TotalUsage:      usage,
		})
	}()

	return &sdk.StreamResult{Stream: ch}, nil
}

// ---------- helpers ----------

func mapResponsesFinishReason(incompleteReason string, hasFunctionCall bool) sdk.FinishReason {
	switch incompleteReason {
	case "max_output_tokens":
		return sdk.FinishReasonLength
	case "content_filter":
		return sdk.FinishReasonContentFilter
	case "":
		if hasFunctionCall {
			return sdk.FinishReasonToolCalls
		}
		return sdk.FinishReasonStop
	default:
		if hasFunctionCall {
			return sdk.FinishReasonToolCalls
		}
		return sdk.FinishReasonOther
	}
}

func convertResponsesUsage(u *responsesUsage) sdk.Usage {
	inputTokens := u.InputTokens
	outputTokens := u.OutputTokens
	cachedTokens := 0
	reasoningTokens := 0

	if u.InputTokensDetails != nil {
		cachedTokens = u.InputTokensDetails.CachedTokens
	}
	if u.OutputTokensDetails != nil {
		reasoningTokens = u.OutputTokensDetails.ReasoningTokens
	}

	return sdk.Usage{
		InputTokens:       inputTokens,
		OutputTokens:      outputTokens,
		TotalTokens:       inputTokens + outputTokens,
		ReasoningTokens:   reasoningTokens,
		CachedInputTokens: cachedTokens,
		InputTokenDetails: sdk.InputTokenDetail{
			CacheReadTokens: cachedTokens,
			NoCacheTokens:   inputTokens - cachedTokens,
		},
		OutputTokenDetails: sdk.OutputTokenDetail{
			ReasoningTokens: reasoningTokens,
			TextTokens:      outputTokens - reasoningTokens,
		},
	}
}

func extractOpenAIEncryptedContent(meta map[string]any) string {
	if meta == nil {
		return ""
	}
	om, ok := meta["openai"].(map[string]any)
	if !ok {
		return ""
	}
	ec, _ := om["reasoningEncryptedContent"].(string)
	return ec
}

func textFromParts(parts []sdk.MessagePart) string {
	var text string
	for _, p := range parts {
		if tp, ok := p.(sdk.TextPart); ok {
			text += tp.Text
		}
	}
	return text
}

func marshalRaw(v any) json.RawMessage {
	data, _ := json.Marshal(v)
	return data
}

func appendRaw(items []json.RawMessage, v any) []json.RawMessage {
	return append(items, marshalRaw(v))
}
