package sdk

import "testing"

func TestBuildStepMessagesPreservesToolCallProviderMetadata(t *testing.T) {
	meta := map[string]any{"google": map[string]any{"thoughtSignature": "sig-1"}}
	msgs := buildStepMessages("", "", nil, []ToolCall{{
		ToolCallID:       "call-1",
		ToolName:         "lookup",
		Input:            map[string]any{"q": "memoh"},
		ProviderMetadata: meta,
	}}, nil, nil)

	if len(msgs) != 1 || len(msgs[0].Content) != 1 {
		t.Fatalf("unexpected messages: %#v", msgs)
	}
	part, ok := msgs[0].Content[0].(ToolCallPart)
	if !ok {
		t.Fatalf("content part = %T, want ToolCallPart", msgs[0].Content[0])
	}
	gotGoogle, ok := part.ProviderMetadata["google"].(map[string]any)
	if !ok {
		t.Fatalf("provider metadata = %#v, want google map", part.ProviderMetadata)
	}
	if gotGoogle["thoughtSignature"] != "sig-1" {
		t.Fatalf("thoughtSignature = %#v, want sig-1", gotGoogle["thoughtSignature"])
	}
}
