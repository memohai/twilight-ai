package sdk

import "testing"

func TestAddUsageAccumulatesCacheWriteTTLDetails(t *testing.T) {
	total := Usage{
		InputTokenDetails: InputTokenDetail{
			CacheWriteTokens:   100,
			CacheWrite5mTokens: 70,
			CacheWrite1hTokens: 30,
		},
	}
	step := Usage{
		InputTokenDetails: InputTokenDetail{
			CacheWriteTokens:   200,
			CacheWrite5mTokens: 120,
			CacheWrite1hTokens: 80,
		},
	}

	got := addUsage(&total, &step)

	if got.InputTokenDetails.CacheWriteTokens != 300 {
		t.Fatalf("CacheWriteTokens = %d, want 300", got.InputTokenDetails.CacheWriteTokens)
	}
	if got.InputTokenDetails.CacheWrite5mTokens != 190 {
		t.Fatalf("CacheWrite5mTokens = %d, want 190", got.InputTokenDetails.CacheWrite5mTokens)
	}
	if got.InputTokenDetails.CacheWrite1hTokens != 110 {
		t.Fatalf("CacheWrite1hTokens = %d, want 110", got.InputTokenDetails.CacheWrite1hTokens)
	}
}

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
