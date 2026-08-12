package sdk_test

import (
	"encoding/json"
	"testing"

	"github.com/memohai/twilight-ai/sdk"
)

func TestMessage_JSON_TextOnly(t *testing.T) {
	msg := sdk.Message{
		Role:    sdk.MessageRoleUser,
		Content: []sdk.MessagePart{sdk.TextPart{Text: "Hello"}},
	}

	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	t.Logf("json: %s", data)

	var got sdk.Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if got.Role != sdk.MessageRoleUser {
		t.Errorf("role: got %q, want %q", got.Role, sdk.MessageRoleUser)
	}
	if len(got.Content) != 1 {
		t.Fatalf("parts: got %d, want 1", len(got.Content))
	}
	tp, ok := got.Content[0].(sdk.TextPart)
	if !ok {
		t.Fatalf("part type: got %T, want TextPart", got.Content[0])
	}
	if tp.Text != "Hello" {
		t.Errorf("text: got %q, want %q", tp.Text, "Hello")
	}
}

func TestMessage_JSON_MultiPart(t *testing.T) {
	msg := sdk.Message{
		Role: sdk.MessageRoleUser,
		Content: []sdk.MessagePart{
			sdk.TextPart{Text: "Describe this image"},
			sdk.ImagePart{Image: "https://example.com/cat.png", MediaType: "image/png"},
		},
	}

	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	t.Logf("json: %s", data)

	var got sdk.Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if len(got.Content) != 2 {
		t.Fatalf("parts: got %d, want 2", len(got.Content))
	}
	if tp, ok := got.Content[0].(sdk.TextPart); !ok || tp.Text != "Describe this image" {
		t.Errorf("part[0]: got %+v", got.Content[0])
	}
	if ip, ok := got.Content[1].(sdk.ImagePart); !ok || ip.Image != "https://example.com/cat.png" || ip.MediaType != "image/png" {
		t.Errorf("part[1]: got %+v", got.Content[1])
	}
}

func TestMessage_JSON_AllPartTypes(t *testing.T) {
	msg := sdk.Message{
		Role: sdk.MessageRoleAssistant,
		Content: []sdk.MessagePart{
			sdk.TextPart{Text: "answer"},
			sdk.ReasoningPart{Text: "thinking", ProviderMetadata: map[string]any{"anthropic": map[string]any{"signature": "sig123"}}},
			sdk.ImagePart{Image: "data:image/png;base64,abc", MediaType: "image/png"},
			sdk.FilePart{Data: "base64data", MediaType: "application/pdf", Filename: "doc.pdf"},
			sdk.ToolCallPart{ToolCallID: "tc1", ToolName: "search", Input: map[string]any{"q": "go"}},
			sdk.ToolResultPart{ToolCallID: "tc1", ToolName: "search", Result: "found", IsError: false},
		},
	}

	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	t.Logf("json: %s", data)

	var got sdk.Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if len(got.Content) != 6 {
		t.Fatalf("parts: got %d, want 6", len(got.Content))
	}

	expectTypes := []sdk.MessagePartType{
		sdk.MessagePartTypeText,
		sdk.MessagePartTypeReasoning,
		sdk.MessagePartTypeImage,
		sdk.MessagePartTypeFile,
		sdk.MessagePartTypeToolCall,
		sdk.MessagePartTypeToolResult,
	}
	for i, want := range expectTypes {
		if got.Content[i].PartType() != want {
			t.Errorf("part[%d]: type got %q, want %q", i, got.Content[i].PartType(), want)
		}
	}

	rp := got.Content[1].(sdk.ReasoningPart)
	if rp.Text != "thinking" {
		t.Errorf("reasoning text: got %q, want %q", rp.Text, "thinking")
	}
	if rp.ProviderMetadata == nil {
		t.Fatal("reasoning: expected providerMetadata to be non-nil")
	}
	am, ok := rp.ProviderMetadata["anthropic"].(map[string]any)
	if !ok {
		t.Fatalf("reasoning: expected providerMetadata[\"anthropic\"] to be map, got %T", rp.ProviderMetadata["anthropic"])
	}
	if sig, _ := am["signature"].(string); sig != "sig123" {
		t.Errorf("reasoning signature: got %q, want %q", sig, "sig123")
	}

	fp := got.Content[3].(sdk.FilePart)
	if fp.Data != "base64data" || fp.Filename != "doc.pdf" {
		t.Errorf("file: got %+v", fp)
	}

	tcp := got.Content[4].(sdk.ToolCallPart)
	if tcp.ToolCallID != "tc1" || tcp.ToolName != "search" {
		t.Errorf("tool call: got %+v", tcp)
	}

	trp := got.Content[5].(sdk.ToolResultPart)
	if trp.ToolCallID != "tc1" || trp.Result != "found" {
		t.Errorf("tool result: got %+v", trp)
	}
}

func TestToolApprovalPause_JSONRoundTrip(t *testing.T) {
	pause := sdk.ToolApprovalPause{
		BatchID: "batch-1",
		System:  "release rules",
		Messages: []sdk.Message{
			sdk.UserMessage("deploy"),
			{
				Role: sdk.MessageRoleAssistant,
				Content: []sdk.MessagePart{sdk.ToolCallPart{
					ToolCallID:       "call-1",
					ToolName:         "deploy",
					Input:            map[string]any{"target": "production"},
					ProviderMetadata: map[string]any{"signature": "message-signature"},
				}},
			},
		},
		Pending: []sdk.DeferredToolApproval{{
			ToolCall: sdk.ToolCall{
				ToolCallID:       "call-1",
				ToolName:         "deploy",
				Input:            map[string]any{"target": "production"},
				ProviderMetadata: map[string]any{"signature": "pending-signature"},
			},
			Approval: sdk.ToolApprovalResult{
				Decision:   sdk.ToolApprovalDecisionDeferred,
				ApprovalID: "approval-1",
				Reason:     "needs release owner",
				Metadata:   map[string]any{"workspace": "release"},
			},
		}},
	}

	data, err := json.Marshal(pause)
	if err != nil {
		t.Fatalf("marshal pause: %v", err)
	}
	var got sdk.ToolApprovalPause
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal pause: %v", err)
	}
	if got.BatchID != "batch-1" || got.System != "release rules" || len(got.Messages) != 2 || len(got.Pending) != 1 {
		t.Fatalf("pause round trip: %#v", got)
	}
	call, ok := got.Messages[1].Content[0].(sdk.ToolCallPart)
	if !ok || call.ToolCallID != "call-1" || call.ToolName != "deploy" ||
		call.Input.(map[string]any)["target"] != "production" ||
		call.ProviderMetadata["signature"] != "message-signature" {
		t.Fatalf("paused message call: %#v", got.Messages[1].Content)
	}
	pending := got.Pending[0]
	if pending.ToolCall.ToolCallID != "call-1" || pending.ToolCall.ToolName != "deploy" ||
		pending.ToolCall.Input.(map[string]any)["target"] != "production" ||
		pending.ToolCall.ProviderMetadata["signature"] != "pending-signature" ||
		pending.Approval.Decision != sdk.ToolApprovalDecisionDeferred ||
		pending.Approval.ApprovalID != "approval-1" ||
		pending.Approval.Reason != "needs release owner" ||
		pending.Approval.Metadata["workspace"] != "release" {
		t.Fatalf("pending approval: %#v", pending)
	}
}

func TestMessage_JSON_FromRawJSON(t *testing.T) {
	raw := `{
		"role": "user",
		"content": [
			{"type": "text", "text": "What is this?"},
			{"type": "image", "image": "https://example.com/photo.jpg", "mediaType": "image/jpeg"}
		]
	}`

	var msg sdk.Message
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if msg.Role != sdk.MessageRoleUser {
		t.Errorf("role: got %q", msg.Role)
	}
	if len(msg.Content) != 2 {
		t.Fatalf("parts: got %d", len(msg.Content))
	}
	if tp, ok := msg.Content[0].(sdk.TextPart); !ok || tp.Text != "What is this?" {
		t.Errorf("part[0]: %+v", msg.Content[0])
	}
	if ip, ok := msg.Content[1].(sdk.ImagePart); !ok || ip.Image != "https://example.com/photo.jpg" {
		t.Errorf("part[1]: %+v", msg.Content[1])
	}
}

func TestMessage_JSON_StringContent_Unmarshal(t *testing.T) {
	raw := `{"role": "user", "content": "Hello, world!"}`

	var msg sdk.Message
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if msg.Role != sdk.MessageRoleUser {
		t.Errorf("role: got %q, want %q", msg.Role, sdk.MessageRoleUser)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("parts: got %d, want 1", len(msg.Content))
	}
	tp, ok := msg.Content[0].(sdk.TextPart)
	if !ok {
		t.Fatalf("part type: got %T, want TextPart", msg.Content[0])
	}
	if tp.Text != "Hello, world!" {
		t.Errorf("text: got %q, want %q", tp.Text, "Hello, world!")
	}
}

func TestMessage_JSON_SingleTextPart_MarshalAsString(t *testing.T) {
	msg := sdk.Message{
		Role:    sdk.MessageRoleUser,
		Content: []sdk.MessagePart{sdk.TextPart{Text: "Hello"}},
	}

	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal raw: %v", err)
	}

	var content string
	if err := json.Unmarshal(raw["content"], &content); err != nil {
		t.Fatalf("content should be a string, got: %s", raw["content"])
	}
	if content != "Hello" {
		t.Errorf("content: got %q, want %q", content, "Hello")
	}
}

func TestMessage_JSON_StringContent_RoundTrip(t *testing.T) {
	msg := sdk.UserMessage("round trip test")

	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	t.Logf("json: %s", data)

	var got sdk.Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if got.Role != sdk.MessageRoleUser {
		t.Errorf("role: got %q", got.Role)
	}
	if len(got.Content) != 1 {
		t.Fatalf("parts: got %d, want 1", len(got.Content))
	}
	tp, ok := got.Content[0].(sdk.TextPart)
	if !ok {
		t.Fatalf("part type: got %T, want TextPart", got.Content[0])
	}
	if tp.Text != "round trip test" {
		t.Errorf("text: got %q", tp.Text)
	}
}

func TestMessage_JSON_TextPartWithMetadata_MarshalAsArray(t *testing.T) {
	msg := sdk.Message{
		Role: sdk.MessageRoleAssistant,
		Content: []sdk.MessagePart{
			sdk.TextPart{Text: "answer", ProviderMetadata: map[string]any{"key": "val"}},
		},
	}

	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal raw: %v", err)
	}

	// Should be an array because TextPart has ProviderMetadata.
	if raw["content"][0] != '[' {
		t.Errorf("content should be an array when TextPart has metadata, got: %s", raw["content"])
	}
}

func TestUsage_JSON(t *testing.T) {
	u := sdk.Usage{
		InputTokens:  10,
		OutputTokens: 20,
		TotalTokens:  30,
		InputTokenDetails: sdk.InputTokenDetail{
			CacheReadTokens: 5,
		},
		OutputTokenDetails: sdk.OutputTokenDetail{
			ReasoningTokens: 8,
			TextTokens:      12,
		},
	}

	data, err := json.Marshal(u)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	t.Logf("json: %s", data)

	var got sdk.Usage
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if got.InputTokens != 10 || got.OutputTokens != 20 || got.TotalTokens != 30 {
		t.Errorf("tokens: %+v", got)
	}
	if got.InputTokenDetails.CacheReadTokens != 5 {
		t.Errorf("cache read: got %d", got.InputTokenDetails.CacheReadTokens)
	}
	if got.OutputTokenDetails.ReasoningTokens != 8 {
		t.Errorf("reasoning: got %d", got.OutputTokenDetails.ReasoningTokens)
	}
}

func TestGenerateResult_JSON(t *testing.T) {
	r := sdk.GenerateResult{
		Text:         "Hello world",
		FinishReason: sdk.FinishReasonStop,
		Usage: sdk.Usage{
			InputTokens:  5,
			OutputTokens: 2,
			TotalTokens:  7,
		},
		ToolCalls: []sdk.ToolCall{{
			ToolCallID: "tc1",
			ToolName:   "search",
			Input:      map[string]any{"query": "go"},
		}},
	}

	data, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	t.Logf("json: %s", data)

	var got sdk.GenerateResult
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if got.Text != "Hello world" {
		t.Errorf("text: got %q", got.Text)
	}
	if got.FinishReason != sdk.FinishReasonStop {
		t.Errorf("finish: got %q", got.FinishReason)
	}
	if len(got.ToolCalls) != 1 || got.ToolCalls[0].ToolName != "search" {
		t.Errorf("tool calls: %+v", got.ToolCalls)
	}
}
