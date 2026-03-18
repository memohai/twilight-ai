package messages

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/memohai/twilight-ai/internal/utils"
	"github.com/memohai/twilight-ai/sdk"
)

const (
	defaultBaseURL      = "https://api.anthropic.com/v1"
	defaultAnthropicVer = "2023-06-01"

	// Content block types for Anthropic API
	blockTypeText     = "text"
	blockTypeThinking = "thinking"
	blockTypeToolUse  = "tool_use"
)

// ThinkingConfig controls extended thinking for Anthropic models.
type ThinkingConfig struct {
	Type         string // "enabled", "adaptive", or "disabled"
	BudgetTokens int    // required when Type is "enabled"
}

type Provider struct {
	apiKey     string
	authToken  string
	baseURL    string
	httpClient *http.Client
	headers    map[string]string
	thinking   *ThinkingConfig
}

type Option func(*Provider)

func WithAPIKey(apiKey string) Option {
	return func(p *Provider) {
		p.apiKey = apiKey
	}
}

// WithAuthToken sets a Bearer token for authentication instead of x-api-key.
// Useful for proxies like OpenRouter that require Authorization: Bearer.
func WithAuthToken(token string) Option {
	return func(p *Provider) {
		p.authToken = token
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

// WithHeaders sets additional HTTP headers for requests.
func WithHeaders(headers map[string]string) Option {
	return func(p *Provider) {
		p.headers = headers
	}
}

// WithThinking enables extended thinking for all requests made by this provider.
func WithThinking(cfg ThinkingConfig) Option {
	return func(p *Provider) {
		p.thinking = &cfg
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
	return "anthropic-messages"
}

func (p *Provider) ListModels(ctx context.Context) ([]sdk.Model, error) {
	resp, err := utils.FetchJSON[modelsListResponse](ctx, p.httpClient, &utils.RequestOptions{
		Method:  http.MethodGet,
		BaseURL: p.baseURL,
		Path:    "/models",
		Headers: p.requestHeaders(),
	})
	if err != nil {
		return nil, fmt.Errorf("anthropic: list models request failed: %w", err)
	}

	models := make([]sdk.Model, 0, len(resp.Data))
	for _, m := range resp.Data {
		models = append(models, sdk.Model{
			ID:          m.ID,
			DisplayName: m.DisplayName,
			Provider:    p,
			Type:        sdk.ModelTypeChat,
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
		Headers: p.requestHeaders(),
	})
	if err != nil {
		return classifyError(err)
	}
	return &sdk.ProviderTestResult{Status: sdk.ProviderStatusOK, Message: "ok"}
}

func (p *Provider) TestModel(ctx context.Context, modelID string) (*sdk.ModelTestResult, error) {
	_, err := utils.FetchJSON[anthropicModelObject](ctx, p.httpClient, &utils.RequestOptions{
		Method:  http.MethodGet,
		BaseURL: p.baseURL,
		Path:    "/models/" + modelID,
		Headers: p.requestHeaders(),
	})
	if err != nil {
		var apiErr *utils.APIError
		if errors.As(err, &apiErr) && apiErr.StatusCode == http.StatusNotFound {
			return &sdk.ModelTestResult{Supported: false, Message: "model not found"}, nil
		}
		return nil, fmt.Errorf("anthropic: test model request failed: %w", err)
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

func (p *Provider) requestHeaders() map[string]string {
	h := map[string]string{
		"anthropic-version": defaultAnthropicVer,
		"Content-Type":      "application/json",
	}
	if p.authToken != "" {
		h["Authorization"] = "Bearer " + p.authToken
	} else if p.apiKey != "" {
		h["x-api-key"] = p.apiKey
	}
	for k, v := range p.headers {
		h[k] = v
	}
	return h
}

// ---------- DoGenerate ----------

func (p *Provider) DoGenerate(ctx context.Context, params sdk.GenerateParams) (*sdk.GenerateResult, error) { //nolint:gocritic // interface method
	if params.Model == nil {
		return nil, fmt.Errorf("anthropic: model is required")
	}

	req := p.buildRequest(&params)

	resp, err := utils.FetchJSON[messagesResponse](ctx, p.httpClient, &utils.RequestOptions{
		Method:  http.MethodPost,
		BaseURL: p.baseURL,
		Path:    "/messages",
		Headers: p.requestHeaders(),
		Body:    req,
	})
	if err != nil {
		var apiErr *utils.APIError
		if errors.As(err, &apiErr) {
			return nil, fmt.Errorf("anthropic: messages request failed: %s", apiErr.Detail())
		}
		return nil, fmt.Errorf("anthropic: messages request failed: %w", err)
	}

	return p.parseResponse(resp)
}

// ---------- buildRequest ----------

func (p *Provider) buildRequest(params *sdk.GenerateParams) *messagesRequest {
	system, messages := convertMessages(params)

	req := &messagesRequest{
		Model:       params.Model.ID,
		System:      system,
		Messages:    messages,
		MaxTokens:   params.MaxTokens,
		Temperature: params.Temperature,
		TopP:        params.TopP,
	}

	if len(params.StopSequences) > 0 {
		req.StopSequences = params.StopSequences
	}
	if len(params.Tools) > 0 {
		req.Tools = convertTools(params.Tools)
		req.ToolChoice = convertToolChoice(params.ToolChoice)
	}

	if p.thinking != nil && p.thinking.Type != "" && p.thinking.Type != "disabled" {
		req.Thinking = &anthropicThinking{
			Type:         p.thinking.Type,
			BudgetTokens: p.thinking.BudgetTokens,
		}
	}

	return req
}

func convertTools(tools []sdk.Tool) []anthropicTool {
	out := make([]anthropicTool, 0, len(tools))
	for _, t := range tools {
		out = append(out, anthropicTool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.Parameters,
		})
	}
	return out
}

func convertToolChoice(choice any) *anthropicToolChoice {
	if choice == nil {
		return nil
	}
	switch v := choice.(type) {
	case string:
		switch v {
		case "auto":
			return &anthropicToolChoice{Type: "auto"}
		case "required":
			return &anthropicToolChoice{Type: "any"}
		case "none":
			return nil
		default:
			return &anthropicToolChoice{Type: "auto"}
		}
	case map[string]any:
		tc := &anthropicToolChoice{Type: "tool"}
		if fn, ok := v["function"].(map[string]any); ok {
			if name, ok := fn["name"].(string); ok {
				tc.Name = name
			}
		}
		return tc
	default:
		return nil
	}
}

// ---------- message conversion ----------

// convertMessages splits SDK messages into Anthropic's system blocks and
// alternating user/assistant messages. Tool result messages are merged into
// user messages, as required by the Anthropic API.
func convertMessages(params *sdk.GenerateParams) ([]contentBlock, []anthropicMessage) {
	var system []contentBlock
	var out []anthropicMessage

	if params.System != "" {
		system = append(system, contentBlock{Type: blockTypeText, Text: params.System})
	}

	for _, msg := range params.Messages {
		switch msg.Role {
		case sdk.MessageRoleSystem:
			for _, part := range msg.Content {
				if tp, ok := part.(sdk.TextPart); ok {
					system = append(system, contentBlock{Type: blockTypeText, Text: tp.Text})
				}
			}

		case sdk.MessageRoleUser:
			blocks := convertUserContent(msg.Content)
			out = appendUserBlocks(out, blocks)

		case sdk.MessageRoleAssistant:
			out = append(out, convertAssistantMessage(msg))

		case sdk.MessageRoleTool:
			blocks := convertToolResults(msg.Content)
			out = appendUserBlocks(out, blocks)
		}
	}

	return system, out
}

// appendUserBlocks appends content blocks to the last user message if it exists,
// or creates a new user message.
func appendUserBlocks(messages []anthropicMessage, blocks []contentBlock) []anthropicMessage {
	if len(messages) > 0 && messages[len(messages)-1].Role == "user" {
		messages[len(messages)-1].Content = append(messages[len(messages)-1].Content, blocks...)
		return messages
	}
	return append(messages, anthropicMessage{
		Role:    "user",
		Content: blocks,
	})
}

func convertUserContent(parts []sdk.MessagePart) []contentBlock {
	var blocks []contentBlock
	for _, part := range parts {
		switch p := part.(type) {
		case sdk.TextPart:
			blocks = append(blocks, contentBlock{Type: blockTypeText, Text: p.Text})
		case sdk.ImagePart:
			blocks = append(blocks, contentBlock{
				Type: "image",
				Source: &imageSource{
					Type:      "base64",
					MediaType: p.MediaType,
					Data:      p.Image,
				},
			})
		case sdk.FilePart:
			blocks = append(blocks, contentBlock{Type: blockTypeText, Text: p.Data})
		}
	}
	return blocks
}

func convertAssistantMessage(msg sdk.Message) anthropicMessage {
	var blocks []contentBlock

	for _, part := range msg.Content {
		switch p := part.(type) {
		case sdk.TextPart:
			blocks = append(blocks, contentBlock{Type: blockTypeText, Text: p.Text})
		case sdk.ReasoningPart:
			sig := extractAnthropicSignature(p.ProviderMetadata)
			blocks = append(blocks, contentBlock{
				Type:      blockTypeThinking,
				Thinking:  p.Text,
				Signature: sig,
			})
		case sdk.ToolCallPart:
			id := p.ToolCallID
			if id == "" {
				id = generateID()
			}
			blocks = append(blocks, contentBlock{
				Type:  blockTypeToolUse,
				ID:    id,
				Name:  p.ToolName,
				Input: p.Input,
			})
		}
	}

	return anthropicMessage{Role: "assistant", Content: blocks}
}

func convertToolResults(parts []sdk.MessagePart) []contentBlock {
	var blocks []contentBlock
	for _, part := range parts {
		if trp, ok := part.(sdk.ToolResultPart); ok {
			content, _ := json.Marshal(trp.Result)
			block := contentBlock{
				Type:      "tool_result",
				ToolUseID: trp.ToolCallID,
				Content:   string(content),
				IsError:   trp.IsError,
			}
			blocks = append(blocks, block)
		}
	}
	return blocks
}

// ---------- parseResponse ----------

func (p *Provider) parseResponse(resp *messagesResponse) (*sdk.GenerateResult, error) {
	result := &sdk.GenerateResult{
		Usage:           convertUsage(&resp.Usage),
		FinishReason:    mapFinishReason(resp.StopReason),
		RawFinishReason: resp.StopReason,
		Response: sdk.ResponseMetadata{
			ID:      resp.ID,
			ModelID: resp.Model,
		},
	}

	for i := range resp.Content {
		block := &resp.Content[i]
		switch block.Type {
		case blockTypeText:
			result.Text += block.Text
		case blockTypeThinking:
			result.Reasoning += block.Thinking
			if block.Signature != "" {
				result.ReasoningProviderMetadata = map[string]any{
					"anthropic": map[string]any{"signature": block.Signature},
				}
			}
		case "redacted_thinking":
			// Redacted thinking blocks don't contain readable text
		case blockTypeToolUse:
			result.ToolCalls = append(result.ToolCalls, sdk.ToolCall{
				ToolCallID: block.ID,
				ToolName:   block.Name,
				Input:      block.Input,
			})
		}
	}

	return result, nil
}

// ---------- DoStream ----------

func (p *Provider) DoStream(ctx context.Context, params sdk.GenerateParams) (*sdk.StreamResult, error) { //nolint:gocritic // interface method
	if params.Model == nil {
		return nil, fmt.Errorf("anthropic: model is required")
	}

	req := p.buildRequest(&params)
	req.Stream = true

	ch := make(chan sdk.StreamPart, 64)

	go func() {
		defer close(ch)

		h := &streamHandler{
			ch:           ch,
			ctx:          ctx,
			activeBlocks: map[int]*streamingBlock{},
		}

		if !h.send(&sdk.StartPart{}) {
			return
		}
		if !h.send(&sdk.StartStepPart{}) {
			return
		}

		err := utils.FetchSSE(ctx, p.httpClient, &utils.RequestOptions{
			Method:  http.MethodPost,
			BaseURL: p.baseURL,
			Path:    "/messages",
			Headers: p.requestHeaders(),
			Body:    req,
		}, h.handleEvent)

		if err != nil {
			var apiErr *utils.APIError
			if errors.As(err, &apiErr) {
				h.send(&sdk.ErrorPart{Error: fmt.Errorf("anthropic: stream failed: %s", apiErr.Detail())})
			} else {
				h.send(&sdk.ErrorPart{Error: fmt.Errorf("anthropic: stream failed: %w", err)})
			}
		}

		h.send(&sdk.FinishPart{
			FinishReason:    h.finishReason,
			RawFinishReason: h.rawFinishReason,
			TotalUsage:      h.usage,
		})
	}()

	return &sdk.StreamResult{Stream: ch}, nil
}

type streamHandler struct {
	ch           chan sdk.StreamPart
	ctx          context.Context
	activeBlocks map[int]*streamingBlock

	rawFinishReason string
	finishReason    sdk.FinishReason
	usage           sdk.Usage
	messageID       string
	messageModel    string
}

func (h *streamHandler) send(part sdk.StreamPart) bool {
	select {
	case h.ch <- part:
		return true
	case <-h.ctx.Done():
		return false
	}
}

func (h *streamHandler) handleEvent(ev *utils.SSEEvent) error {
	var event streamEvent
	if err := json.Unmarshal([]byte(ev.Data), &event); err != nil {
		h.send(&sdk.ErrorPart{Error: fmt.Errorf("anthropic: unmarshal event: %w", err)})
		return err
	}

	switch event.Type {
	case "message_start":
		h.onMessageStart(&event)
	case "content_block_start":
		h.onBlockStart(&event)
	case "content_block_delta":
		h.onBlockDelta(&event)
	case "content_block_stop":
		h.onBlockStop(&event)
	case "message_delta":
		h.onMessageDelta(&event)
	case "message_stop":
		return utils.ErrStreamDone
	case "ping":
		// ignore
	case "error":
		h.onError(&event)
	}
	return nil
}

func (h *streamHandler) onMessageStart(event *streamEvent) {
	if event.Message == nil {
		return
	}
	h.messageID = event.Message.ID
	h.messageModel = event.Message.Model
	h.usage = convertUsage(&event.Message.Usage)
}

func (h *streamHandler) onBlockStart(event *streamEvent) {
	if event.Index == nil || event.ContentBlock == nil {
		return
	}
	idx := *event.Index
	cb := event.ContentBlock
	switch cb.Type {
	case blockTypeText:
		h.activeBlocks[idx] = &streamingBlock{blockType: blockTypeText}
		h.send(&sdk.TextStartPart{ID: h.messageID})
	case blockTypeThinking:
		h.activeBlocks[idx] = &streamingBlock{blockType: blockTypeThinking}
		h.send(&sdk.ReasoningStartPart{ID: h.messageID})
	case blockTypeToolUse:
		h.activeBlocks[idx] = &streamingBlock{
			blockType: blockTypeToolUse,
			toolID:    cb.ID,
			toolName:  cb.Name,
		}
		h.send(&sdk.ToolInputStartPart{
			ID:       cb.ID,
			ToolName: cb.Name,
		})
	}
}

func (h *streamHandler) onBlockDelta(event *streamEvent) {
	if event.Index == nil || event.Delta == nil {
		return
	}
	idx := *event.Index
	delta := event.Delta
	sb := h.activeBlocks[idx]

	switch delta.Type {
	case "text_delta":
		h.send(&sdk.TextDeltaPart{ID: h.messageID, Text: delta.Text})
	case "thinking_delta":
		h.send(&sdk.ReasoningDeltaPart{ID: h.messageID, Text: delta.Thinking})
	case "input_json_delta":
		if sb != nil {
			sb.args += delta.PartialJSON
			h.send(&sdk.ToolInputDeltaPart{
				ID:    sb.toolID,
				Delta: delta.PartialJSON,
			})
		}
	case "signature_delta":
		if sb != nil {
			sb.signature += delta.Signature
		}
	}
}

func (h *streamHandler) onBlockStop(event *streamEvent) {
	if event.Index == nil {
		return
	}
	idx := *event.Index
	sb, ok := h.activeBlocks[idx]
	if !ok {
		return
	}
	delete(h.activeBlocks, idx)

	switch sb.blockType {
	case blockTypeText:
		h.send(&sdk.TextEndPart{ID: h.messageID})
	case blockTypeThinking:
		var meta map[string]any
		if sb.signature != "" {
			meta = map[string]any{
				"anthropic": map[string]any{"signature": sb.signature},
			}
		}
		h.send(&sdk.ReasoningEndPart{ID: h.messageID, ProviderMetadata: meta})
	case blockTypeToolUse:
		h.send(&sdk.ToolInputEndPart{ID: sb.toolID})
		var input any
		if sb.args != "" {
			if err := json.Unmarshal([]byte(sb.args), &input); err != nil {
				h.send(&sdk.ErrorPart{Error: fmt.Errorf("anthropic: unmarshal tool args for %q: %w", sb.toolName, err)})
			}
		}
		h.send(&sdk.StreamToolCallPart{
			ToolCallID: sb.toolID,
			ToolName:   sb.toolName,
			Input:      input,
		})
	}
}

func (h *streamHandler) onMessageDelta(event *streamEvent) {
	if event.Delta != nil {
		h.rawFinishReason = event.Delta.StopReason
		h.finishReason = mapFinishReason(h.rawFinishReason)
	}
	if event.Usage != nil {
		h.usage.OutputTokens = event.Usage.OutputTokens
		h.usage.TotalTokens = h.usage.InputTokens + h.usage.OutputTokens
	}
	h.send(&sdk.FinishStepPart{
		FinishReason:    h.finishReason,
		RawFinishReason: h.rawFinishReason,
		Usage:           h.usage,
		Response: sdk.ResponseMetadata{
			ID:      h.messageID,
			ModelID: h.messageModel,
		},
	})
}

func (h *streamHandler) onError(event *streamEvent) {
	errMsg := "unknown error"
	if event.Delta != nil && event.Delta.Text != "" {
		errMsg = event.Delta.Text
	}
	h.send(&sdk.ErrorPart{Error: fmt.Errorf("anthropic: stream error: %s", errMsg)})
}

type streamingBlock struct {
	blockType string
	toolID    string
	toolName  string
	args      string
	signature string
}

// ---------- helpers ----------

func generateID() string {
	b := make([]byte, 12)
	rand.Read(b)
	return fmt.Sprintf("toolu_%x", b)
}

func convertUsage(u *messagesUsage) sdk.Usage {
	total := u.InputTokens + u.OutputTokens
	return sdk.Usage{
		InputTokens:       u.InputTokens,
		OutputTokens:      u.OutputTokens,
		TotalTokens:       total,
		CachedInputTokens: u.CacheReadInputTokens,
		InputTokenDetails: sdk.InputTokenDetail{
			CacheReadTokens:  u.CacheReadInputTokens,
			CacheWriteTokens: u.CacheCreationInputTokens,
		},
	}
}

func mapFinishReason(reason string) sdk.FinishReason {
	switch reason {
	case "end_turn", "stop_sequence":
		return sdk.FinishReasonStop
	case "tool_use":
		return sdk.FinishReasonToolCalls
	case "max_tokens":
		return sdk.FinishReasonLength
	default:
		return sdk.FinishReasonUnknown
	}
}

func extractAnthropicSignature(meta map[string]any) string {
	if meta == nil {
		return ""
	}
	am, ok := meta["anthropic"].(map[string]any)
	if !ok {
		return ""
	}
	sig, _ := am["signature"].(string)
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
