package sdk

import (
	"encoding/json"
	"fmt"
)

type MessageRole string

const (
	MessageRoleUser      MessageRole = "user"
	MessageRoleAssistant MessageRole = "assistant"
	MessageRoleSystem    MessageRole = "system"
	MessageRoleTool      MessageRole = "tool"
)

type MessagePartType string

const (
	MessagePartTypeText       MessagePartType = "text"
	MessagePartTypeReasoning  MessagePartType = "reasoning"
	MessagePartTypeImage      MessagePartType = "image"
	MessagePartTypeFile       MessagePartType = "file"
	MessagePartTypeToolCall   MessagePartType = "tool-call"
	MessagePartTypeToolResult MessagePartType = "tool-result"
)

type MessagePart interface {
	PartType() MessagePartType
}

// CacheControl specifies caching behaviour for a content block.
// Anthropic currently supports Type "ephemeral" with an optional TTL.
// Leave TTL empty for the default 5-minute cache; set TTL to "1h" for a
// 1-hour cache (billed at a higher rate).
type CacheControl struct {
	Type string `json:"type"`          // "ephemeral"
	TTL  string `json:"ttl,omitempty"` // "" (5 min, default) | "1h"
}

// --- Text ---

type TextPart struct {
	Text             string         `json:"text"`
	CacheControl     *CacheControl  `json:"cacheControl,omitempty"`
	ProviderMetadata map[string]any `json:"providerMetadata,omitempty"`
}

func (p TextPart) PartType() MessagePartType { return MessagePartTypeText }

// --- Reasoning ---

type ReasoningPart struct {
	Text             string         `json:"text"`
	ProviderMetadata map[string]any `json:"providerMetadata,omitempty"`
}

func (p ReasoningPart) PartType() MessagePartType { return MessagePartTypeReasoning }

// --- Image ---

type ImagePart struct {
	Image        string        `json:"image"`
	MediaType    string        `json:"mediaType,omitempty"`
	CacheControl *CacheControl `json:"cacheControl,omitempty"`
}

func (p ImagePart) PartType() MessagePartType { return MessagePartTypeImage }

// --- File ---

type FilePart struct {
	Data         string        `json:"data"`
	MediaType    string        `json:"mediaType,omitempty"`
	Filename     string        `json:"filename,omitempty"`
	CacheControl *CacheControl `json:"cacheControl,omitempty"`
}

func (p FilePart) PartType() MessagePartType { return MessagePartTypeFile }

// --- Tool Call (in assistant messages) ---

type ToolCallPart struct {
	ToolCallID       string         `json:"toolCallId"`
	ToolName         string         `json:"toolName"`
	Input            any            `json:"input"`
	ProviderMetadata map[string]any `json:"providerMetadata,omitempty"`
}

func (p ToolCallPart) PartType() MessagePartType { return MessagePartTypeToolCall }

// --- Tool Result (in tool messages) ---

type ToolResultPart struct {
	ToolCallID string `json:"toolCallId"`
	ToolName   string `json:"toolName"`
	Result     any    `json:"result"`
	IsError    bool   `json:"isError,omitempty"`
}

func (p ToolResultPart) PartType() MessagePartType { return MessagePartTypeToolResult }

// --- Message ---

type Message struct {
	Role    MessageRole   `json:"role"`
	Content []MessagePart `json:"content"`
	Usage   *Usage        `json:"usage,omitempty"`
}

// --- Convenience constructors ---

// UserMessage creates a user message with one or more text parts.
func UserMessage(text string, extra ...MessagePart) Message {
	parts := make([]MessagePart, 0, 1+len(extra))
	parts = append(parts, TextPart{Text: text})
	parts = append(parts, extra...)
	return Message{Role: MessageRoleUser, Content: parts}
}

// SystemMessage creates a system message with a single text part.
func SystemMessage(text string) Message {
	return Message{Role: MessageRoleSystem, Content: []MessagePart{TextPart{Text: text}}}
}

// AssistantMessage creates an assistant message with a single text part.
func AssistantMessage(text string) Message {
	return Message{Role: MessageRoleAssistant, Content: []MessagePart{TextPart{Text: text}}}
}

// ToolMessage creates a tool-role message containing one or more ToolResultParts.
func ToolMessage(results ...ToolResultPart) Message {
	parts := make([]MessagePart, len(results))
	for i, r := range results {
		parts[i] = r
	}
	return Message{Role: MessageRoleTool, Content: parts}
}

// --- JSON ---

func (m Message) MarshalJSON() ([]byte, error) {
	// Single TextPart with no metadata and no cache control → emit content as a plain string.
	var content any
	if len(m.Content) == 1 {
		if tp, ok := m.Content[0].(TextPart); ok && len(tp.ProviderMetadata) == 0 && tp.CacheControl == nil {
			content = tp.Text
		}
	}
	if content == nil {
		parts := make([]json.RawMessage, 0, len(m.Content))
		for _, p := range m.Content {
			raw, err := marshalPart(p)
			if err != nil {
				return nil, err
			}
			parts = append(parts, raw)
		}
		content = parts
	}
	if m.Usage != nil {
		return json.Marshal(struct {
			Role    MessageRole `json:"role"`
			Content any         `json:"content"`
			Usage   *Usage      `json:"usage,omitempty"`
		}{Role: m.Role, Content: content, Usage: m.Usage})
	}
	return json.Marshal(struct {
		Role    MessageRole `json:"role"`
		Content any         `json:"content"`
	}{Role: m.Role, Content: content})
}

func (m *Message) UnmarshalJSON(data []byte) error {
	var raw struct {
		Role    MessageRole     `json:"role"`
		Content json.RawMessage `json:"content"`
		Usage   *Usage          `json:"usage,omitempty"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	m.Role = raw.Role
	m.Usage = raw.Usage

	// content can be a string or an array of parts.
	if len(raw.Content) > 0 && raw.Content[0] == '"' {
		var s string
		if err := json.Unmarshal(raw.Content, &s); err != nil {
			return fmt.Errorf("unmarshal string content: %w", err)
		}
		m.Content = []MessagePart{TextPart{Text: s}}
		return nil
	}

	var parts []json.RawMessage
	if err := json.Unmarshal(raw.Content, &parts); err != nil {
		return fmt.Errorf("unmarshal content array: %w", err)
	}
	m.Content = make([]MessagePart, 0, len(parts))
	for _, r := range parts {
		p, err := unmarshalPart(r)
		if err != nil {
			return err
		}
		m.Content = append(m.Content, p)
	}
	return nil
}

func marshalPart(p MessagePart) (json.RawMessage, error) {
	type typed struct {
		Type MessagePartType `json:"type"`
	}
	base, err := json.Marshal(p)
	if err != nil {
		return nil, err
	}
	typeJSON, _ := json.Marshal(typed{Type: p.PartType()})

	// merge {"type":"..."} into the part's JSON
	merged := make(map[string]json.RawMessage)
	if err := json.Unmarshal(typeJSON, &merged); err != nil {
		return nil, err
	}
	if err := json.Unmarshal(base, &merged); err != nil {
		return nil, err
	}
	return json.Marshal(merged)
}

func unmarshalPart(data json.RawMessage) (MessagePart, error) {
	var probe struct {
		Type MessagePartType `json:"type"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return nil, fmt.Errorf("unmarshal message part type: %w", err)
	}
	switch probe.Type {
	case MessagePartTypeText:
		var p TextPart
		if err := json.Unmarshal(data, &p); err != nil {
			return nil, err
		}
		return p, nil
	case MessagePartTypeReasoning:
		var p ReasoningPart
		if err := json.Unmarshal(data, &p); err != nil {
			return nil, err
		}
		return p, nil
	case MessagePartTypeImage:
		var p ImagePart
		if err := json.Unmarshal(data, &p); err != nil {
			return nil, err
		}
		return p, nil
	case MessagePartTypeFile:
		var p FilePart
		if err := json.Unmarshal(data, &p); err != nil {
			return nil, err
		}
		return p, nil
	case MessagePartTypeToolCall:
		var p ToolCallPart
		if err := json.Unmarshal(data, &p); err != nil {
			return nil, err
		}
		return p, nil
	case MessagePartTypeToolResult:
		var p ToolResultPart
		if err := json.Unmarshal(data, &p); err != nil {
			return nil, err
		}
		return p, nil
	default:
		return nil, fmt.Errorf("unknown message part type: %q", probe.Type)
	}
}
