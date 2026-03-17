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

// --- Text ---

type TextPart struct {
	Text string `json:"text"`
}

func (p TextPart) PartType() MessagePartType { return MessagePartTypeText }

// --- Reasoning ---

type ReasoningPart struct {
	Text      string `json:"text"`
	Signature string `json:"signature,omitempty"`
}

func (p ReasoningPart) PartType() MessagePartType { return MessagePartTypeReasoning }

// --- Image ---

type ImagePart struct {
	Image     string `json:"image"`
	MediaType string `json:"mediaType,omitempty"`
}

func (p ImagePart) PartType() MessagePartType { return MessagePartTypeImage }

// --- File ---

type FilePart struct {
	Data      string `json:"data"`
	MediaType string `json:"mediaType,omitempty"`
	Filename  string `json:"filename,omitempty"`
}

func (p FilePart) PartType() MessagePartType { return MessagePartTypeFile }

// --- Tool Call (in assistant messages) ---

type ToolCallPart struct {
	ToolCallID string `json:"toolCallId"`
	ToolName   string `json:"toolName"`
	Input      any    `json:"input"`
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
}

// --- Convenience constructors ---

// UserMessage creates a user message with one or more text parts.
func UserMessage(text string, extra ...MessagePart) Message {
	parts := []MessagePart{TextPart{Text: text}}
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
	parts := make([]json.RawMessage, 0, len(m.Content))
	for _, p := range m.Content {
		raw, err := marshalPart(p)
		if err != nil {
			return nil, err
		}
		parts = append(parts, raw)
	}
	return json.Marshal(struct {
		Role    MessageRole       `json:"role"`
		Content []json.RawMessage `json:"content"`
	}{Role: m.Role, Content: parts})
}

func (m *Message) UnmarshalJSON(data []byte) error {
	var raw struct {
		Role    MessageRole       `json:"role"`
		Content []json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	m.Role = raw.Role
	m.Content = make([]MessagePart, 0, len(raw.Content))
	for _, r := range raw.Content {
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
	json.Unmarshal(typeJSON, &merged)
	json.Unmarshal(base, &merged)
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
		return p, json.Unmarshal(data, &p)
	case MessagePartTypeReasoning:
		var p ReasoningPart
		return p, json.Unmarshal(data, &p)
	case MessagePartTypeImage:
		var p ImagePart
		return p, json.Unmarshal(data, &p)
	case MessagePartTypeFile:
		var p FilePart
		return p, json.Unmarshal(data, &p)
	case MessagePartTypeToolCall:
		var p ToolCallPart
		return p, json.Unmarshal(data, &p)
	case MessagePartTypeToolResult:
		var p ToolResultPart
		return p, json.Unmarshal(data, &p)
	default:
		return nil, fmt.Errorf("unknown message part type: %q", probe.Type)
	}
}
