package sdk_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/memohai/twilight-ai/sdk"
)

// pausedFixture returns a ToolApprovalPause in the exact shape a deferred
// pause produces: the conversation ends with an assistant message carrying
// two tool calls and a tool message resolving only the first; Pending lists
// the unresolved call.
func pausedFixture() sdk.ToolApprovalPause {
	deployCall := sdk.ToolCall{ToolCallID: "c2", ToolName: "deploy", Input: map[string]any{"env": "prod"}}
	return sdk.ToolApprovalPause{
		Messages: []sdk.Message{
			sdk.UserMessage("deploy and notify"),
			{
				Role: sdk.MessageRoleAssistant,
				Content: []sdk.MessagePart{
					sdk.ToolCallPart{ToolCallID: "c1", ToolName: "notify", Input: map[string]any{"msg": "hi"}},
					sdk.ToolCallPart{ToolCallID: "c2", ToolName: "deploy", Input: map[string]any{"env": "prod"}},
				},
			},
			sdk.ToolMessage(sdk.ToolResultPart{ToolCallID: "c1", ToolName: "notify", Result: "sent"}),
		},
		Pending: []sdk.DeferredToolApproval{
			{ToolCall: deployCall, Approval: sdk.ToolApprovalResult{Decision: sdk.ToolApprovalDecisionDeferred, ApprovalID: "a-c2"}},
		},
	}
}

func resumeTools(executed map[string]any, mu *sync.Mutex) []sdk.Tool {
	return []sdk.Tool{
		{
			Name: "notify", Parameters: &jsonschema.Schema{Type: "object"},
			Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
				mu.Lock()
				executed["notify"] = input
				mu.Unlock()
				return "sent", nil
			},
		},
		{
			Name: "deploy", Parameters: &jsonschema.Schema{Type: "object"}, RequireApproval: true,
			Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
				mu.Lock()
				executed["deploy"] = input
				mu.Unlock()
				return "deployed", nil
			},
		},
	}
}

// toolResultsByID collects every ToolResultPart in the conversation, keyed by
// tool call ID, in encounter order.
func toolResultsByID(messages []sdk.Message) (order []string, byID map[string]sdk.ToolResultPart) {
	byID = map[string]sdk.ToolResultPart{}
	for _, m := range messages {
		if m.Role != sdk.MessageRoleTool {
			continue
		}
		for _, p := range m.Content {
			if trp, ok := p.(sdk.ToolResultPart); ok {
				order = append(order, trp.ToolCallID)
				byID[trp.ToolCallID] = trp
			}
		}
	}
	return order, byID
}

// TestDeferredApprovalLifecycle drives the full cycle through the public API:
// a run pauses on two deferred approvals with its resolved sibling executed,
// a resume applies mixed decisions and pauses again on a new gated call, and
// a second resume completes the run. The pause travels as the portable
// ToolApprovalPause value — including a JSON round-trip, as a host persisting
// it would do. The scripted provider asserts the conversation is
// protocol-complete at every model call.
// lifecycleProvider scripts three model calls for the lifecycle test and
// asserts the conversation is protocol-complete at each one.
func lifecycleProvider(t *testing.T) *mockProvider {
	t.Helper()
	assertResults := func(call int, params sdk.GenerateParams, wantIDs []string, rejectedID, rejectedContains string) {
		t.Helper()
		order, byID := toolResultsByID(params.Messages)
		if len(order) != len(wantIDs) {
			t.Fatalf("model call %d saw tool results %v, want %v", call, order, wantIDs)
		}
		for i, want := range wantIDs {
			if order[i] != want {
				t.Fatalf("model call %d saw tool results %v, want %v", call, order, wantIDs)
			}
		}
		if r := byID[rejectedID]; !r.IsError || !strings.Contains(r.Result.(string), rejectedContains) {
			t.Fatalf("rejected %s as seen by the model: %#v", rejectedID, r)
		}
	}
	return &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		switch call {
		case 1:
			return &sdk.GenerateResult{
				FinishReason: sdk.FinishReasonToolCalls,
				ToolCalls: []sdk.ToolCall{
					{ToolCallID: "c1", ToolName: "lookup", Input: nil},
					{ToolCallID: "c2", ToolName: "deploy", Input: map[string]any{"env": "prod"}},
					{ToolCallID: "c3", ToolName: "notify", Input: nil},
				},
			}, nil
		case 2:
			assertResults(2, params, []string{"c1", "c2", "c3"}, "c3", "no notifications")
			return &sdk.GenerateResult{
				FinishReason: sdk.FinishReasonToolCalls,
				ToolCalls:    []sdk.ToolCall{{ToolCallID: "c4", ToolName: "deploy", Input: map[string]any{"env": "prod", "retry": true}}},
			}, nil
		case 3:
			assertResults(3, params, []string{"c1", "c2", "c3", "c4"}, "c4", "one deploy is enough")
			return &sdk.GenerateResult{Text: "released", FinishReason: sdk.FinishReasonStop}, nil
		}
		t.Fatalf("unexpected provider call %d", call)
		return nil, nil
	}}
}

func TestDeferredApprovalLifecycle(t *testing.T) {
	mp := lifecycleProvider(t)

	counts := map[string]int{}
	var mu sync.Mutex
	tool := func(name string, gated bool) sdk.Tool {
		return sdk.Tool{
			Name: name, Parameters: &jsonschema.Schema{Type: "object"}, RequireApproval: gated,
			Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
				mu.Lock()
				counts[name]++
				mu.Unlock()
				return name + "-ok", nil
			},
		}
	}
	tools := []sdk.Tool{tool("lookup", false), tool("deploy", true), tool("notify", true)}
	handler := sdk.WithApprovalHandler(func(_ context.Context, tc sdk.ToolCall) (sdk.ToolApprovalResult, error) {
		return sdk.ToolApprovalResult{Decision: sdk.ToolApprovalDecisionDeferred, ApprovalID: "approval-" + tc.ToolCallID}, nil
	})

	// Run 1 pauses with deploy and notify pending; lookup already executed.
	r1, err := sdk.GenerateTextResult(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("release v2")}),
		sdk.WithTools(tools), sdk.WithMaxSteps(5), handler)
	if err != nil {
		t.Fatalf("run 1: %v", err)
	}
	if r1.FinishReason != sdk.FinishReasonPaused || r1.Pause == nil {
		t.Fatalf("run 1: finish=%q pause=%v", r1.FinishReason, r1.Pause)
	}
	if len(r1.Pause.Pending) != 2 || r1.Pause.Pending[0].ToolCall.ToolCallID != "c2" || r1.Pause.Pending[1].ToolCall.ToolCallID != "c3" {
		t.Fatalf("run 1 pending: %#v", r1.Pause.Pending)
	}
	if counts["lookup"] != 1 {
		t.Fatalf("lookup must execute before the pause: %v", counts)
	}

	// The pause is plain data: persist and reload it as a host would.
	blob, err := json.Marshal(r1.Pause)
	if err != nil {
		t.Fatalf("marshal pause: %v", err)
	}
	var pause1 sdk.ToolApprovalPause
	if err := json.Unmarshal(blob, &pause1); err != nil {
		t.Fatalf("unmarshal pause: %v", err)
	}

	// Resume 1 approves deploy and rejects notify; the model requests another
	// gated deploy and the run pauses again.
	r2, err := sdk.ResumeText(context.Background(), pause1,
		map[string]sdk.ToolDecision{
			"c2": {Decision: sdk.ToolDecisionApproved},
			"c3": {Decision: sdk.ToolDecisionRejected, Reason: "no notifications during release"},
		},
		sdk.WithModel(mockModel(mp)), sdk.WithTools(tools), sdk.WithMaxSteps(5), handler)
	if err != nil {
		t.Fatalf("resume 1: %v", err)
	}
	if counts["deploy"] != 1 {
		t.Fatalf("approved deploy must execute exactly once: %v", counts)
	}
	if counts["notify"] != 0 {
		t.Fatalf("rejected notify must not execute: %v", counts)
	}
	if r2.Resume == nil || len(r2.Resume.Results) != 2 {
		t.Fatalf("resolution: %#v", r2.Resume)
	}
	if r2.FinishReason != sdk.FinishReasonPaused || r2.Pause == nil || len(r2.Pause.Pending) != 1 || r2.Pause.Pending[0].ToolCall.ToolCallID != "c4" {
		t.Fatalf("second pause: finish=%q pause=%#v", r2.FinishReason, r2.Pause)
	}
	// The resolution is a report, not a synthetic step: only the model step.
	if len(r2.Steps) != 1 || len(r2.Steps[0].ToolCalls) != 1 || r2.Steps[0].ToolCalls[0].ToolCallID != "c4" {
		t.Fatalf("resumed run steps: %#v", r2.Steps)
	}

	// Resume 2 rejects the retry; the model finishes. The second pause is
	// self-contained — no manual conversation assembly.
	r3, err := sdk.ResumeText(context.Background(), *r2.Pause,
		map[string]sdk.ToolDecision{
			"c4": {Decision: sdk.ToolDecisionRejected, Reason: "one deploy is enough"},
		},
		sdk.WithModel(mockModel(mp)), sdk.WithTools(tools), sdk.WithMaxSteps(5), handler)
	if err != nil {
		t.Fatalf("resume 2: %v", err)
	}
	if r3.Text != "released" || r3.FinishReason != sdk.FinishReasonStop || r3.Pause != nil {
		t.Fatalf("final result: text=%q finish=%q pause=%v", r3.Text, r3.FinishReason, r3.Pause)
	}
	if counts["lookup"] != 1 || counts["deploy"] != 1 || counts["notify"] != 0 {
		t.Fatalf("execution counts after the lifecycle: %v", counts)
	}
	if mp.calls != 3 {
		t.Fatalf("provider calls: %d, want 3", mp.calls)
	}
}

// Streaming resume applies the decisions before the stream opens: the
// resolution is available immediately, the stream carries the provider's
// normal lifecycle, and validation failures return synchronously.
func TestResumeTextStream(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		return &sdk.GenerateResult{Text: "all done", FinishReason: sdk.FinishReasonStop}, nil
	}}
	executed := map[string]any{}
	var mu sync.Mutex

	sr, err := sdk.ResumeTextStream(context.Background(), pausedFixture(),
		map[string]sdk.ToolDecision{
			"c2": {Decision: sdk.ToolDecisionApproved},
		},
		sdk.WithModel(mockModel(mp)),
		sdk.WithTools(resumeTools(executed, &mu)),
		sdk.WithMaxSteps(5),
	)
	if err != nil {
		t.Fatalf("ResumeTextStream: %v", err)
	}
	if _, ok := executed["deploy"]; !ok {
		t.Error("approved call must execute before the stream opens")
	}
	if sr.Resume == nil || len(sr.Resume.Results) != 1 {
		t.Fatalf("resolution must be available immediately: %#v", sr.Resume)
	}

	// The stream opens with the resume phase's tool parts (they happened
	// before the model call), then the provider lifecycle.
	var parts []sdk.StreamPart
	var text string
	var gotFinish bool
	for part := range sr.Stream {
		parts = append(parts, part)
		switch p := part.(type) {
		case *sdk.TextDeltaPart:
			text += p.Text
		case *sdk.ErrorPart:
			t.Fatalf("stream error: %v", p.Error)
		case *sdk.FinishPart:
			gotFinish = true
		}
	}
	if len(parts) == 0 {
		t.Fatal("empty stream")
	}
	resumeResult, ok := parts[0].(*sdk.StreamToolResultPart)
	if !ok || resumeResult.ToolCallID != "c2" {
		t.Errorf("stream must open with the resumed call's result part, got %T", parts[0])
	}
	sawStart := false
	for _, p := range parts[1:] {
		if _, ok := p.(*sdk.StartPart); ok {
			sawStart = true
			break
		}
	}
	if !sawStart {
		t.Error("provider StartPart must follow the resume parts")
	}
	if text != "all done" || !gotFinish || sr.Pause != nil {
		t.Fatalf("text=%q finish=%v pause=%#v", text, gotFinish, sr.Pause)
	}

	// Validation failures surface synchronously, before any stream exists.
	mp2 := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		t.Fatal("provider must not be called on validation failure")
		return nil, nil
	}}
	if _, err := sdk.ResumeTextStream(context.Background(), pausedFixture(),
		map[string]sdk.ToolDecision{},
		sdk.WithModel(mockModel(mp2)),
		sdk.WithTools(resumeTools(executed, &mu)),
		sdk.WithMaxSteps(5),
	); err == nil || !strings.Contains(err.Error(), "missing decisions") {
		t.Fatalf("expected synchronous validation error, got %v", err)
	}
}

// The resume contract fails closed, before any tool execution or model call.
func TestResumeValidation(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		t.Fatal("provider must not be called when resume validation fails")
		return nil, nil
	}}
	executed := map[string]any{}
	var mu sync.Mutex
	tools := resumeTools(executed, &mu)

	assistant := func(parts ...sdk.MessagePart) sdk.Message {
		return sdk.Message{Role: sdk.MessageRoleAssistant, Content: parts}
	}
	pauseWith := func(msgs []sdk.Message, pending ...sdk.ToolCall) sdk.ToolApprovalPause {
		p := sdk.ToolApprovalPause{Messages: msgs}
		for _, tc := range pending {
			p.Pending = append(p.Pending, sdk.DeferredToolApproval{ToolCall: tc})
		}
		return p
	}

	cases := []struct {
		name      string
		pause     sdk.ToolApprovalPause
		tools     []sdk.Tool
		decisions map[string]sdk.ToolDecision
		wantErr   string
	}{
		{
			name:      "missing decision",
			pause:     pausedFixture(),
			tools:     tools,
			decisions: map[string]sdk.ToolDecision{},
			wantErr:   "missing decisions for pending tool calls: c2",
		},
		{
			name:  "unknown decision",
			pause: pausedFixture(),
			tools: tools,
			decisions: map[string]sdk.ToolDecision{
				"c2":    {Decision: sdk.ToolDecisionApproved},
				"ghost": {Decision: sdk.ToolDecisionApproved},
			},
			wantErr: "decisions for unknown tool calls: ghost",
		},
		{
			name:      "zero-value decision fails closed",
			pause:     pausedFixture(),
			tools:     tools,
			decisions: map[string]sdk.ToolDecision{"c2": {}},
			wantErr:   "must be explicitly approved or rejected: c2",
		},
		{
			name:      "approved tool missing from the tool set",
			pause:     pausedFixture(),
			tools:     tools[:1], // deploy omitted
			decisions: map[string]sdk.ToolDecision{"c2": {Decision: sdk.ToolDecisionApproved}},
			wantErr:   "approved tools missing from the tool set",
		},
		{
			name:      "pause without pending calls",
			pause:     sdk.ToolApprovalPause{Messages: []sdk.Message{sdk.UserMessage("hi")}},
			tools:     tools,
			decisions: map[string]sdk.ToolDecision{},
			wantErr:   "pause has no pending tool calls",
		},
		{
			name: "pause disagrees with its conversation",
			pause: pauseWith(
				// Tail leaves NOTHING unresolved, but Pending claims c9.
				[]sdk.Message{
					sdk.UserMessage("go"),
					assistant(sdk.ToolCallPart{ToolCallID: "c1", ToolName: "notify", Input: nil}),
					sdk.ToolMessage(sdk.ToolResultPart{ToolCallID: "c1", ToolName: "notify", Result: "sent"}),
				},
				sdk.ToolCall{ToolCallID: "c9", ToolName: "deploy"},
			),
			tools:     tools,
			decisions: map[string]sdk.ToolDecision{"c9": {Decision: sdk.ToolDecisionApproved}},
			wantErr:   "pause.Pending lists 1 calls but pause.Messages leaves 0 unresolved",
		},
		{
			name: "empty tool call ID",
			pause: pauseWith(
				[]sdk.Message{
					sdk.UserMessage("go"),
					assistant(sdk.ToolCallPart{ToolCallID: "", ToolName: "deploy", Input: nil}),
				},
				sdk.ToolCall{ToolCallID: "", ToolName: "deploy"},
			),
			tools:     tools,
			decisions: map[string]sdk.ToolDecision{},
			wantErr:   "cannot be addressed",
		},
		{
			name: "duplicate tool call ID",
			pause: pauseWith(
				[]sdk.Message{
					sdk.UserMessage("go"),
					assistant(
						sdk.ToolCallPart{ToolCallID: "c1", ToolName: "deploy", Input: nil},
						sdk.ToolCallPart{ToolCallID: "c1", ToolName: "notify", Input: nil},
					),
				},
				sdk.ToolCall{ToolCallID: "c1", ToolName: "deploy"},
				sdk.ToolCall{ToolCallID: "c1", ToolName: "notify"},
			),
			tools:     tools,
			decisions: map[string]sdk.ToolDecision{"c1": {Decision: sdk.ToolDecisionApproved}},
			wantErr:   "duplicate tool call ID",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := sdk.ResumeText(context.Background(), tc.pause, tc.decisions,
				sdk.WithModel(mockModel(mp)),
				sdk.WithTools(tc.tools),
				sdk.WithMaxSteps(5),
			)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error = %v, want containing %q", err, tc.wantErr)
			}
			mu.Lock()
			if len(executed) != 0 {
				t.Fatalf("no tool may execute on validation failure: %v", executed)
			}
			mu.Unlock()
		})
	}

	// WithMessages conflicts with the pause: the conversation has one source.
	t.Run("WithMessages conflicts", func(t *testing.T) {
		_, err := sdk.ResumeText(context.Background(), pausedFixture(),
			map[string]sdk.ToolDecision{"c2": {Decision: sdk.ToolDecisionApproved}},
			sdk.WithModel(mockModel(mp)),
			sdk.WithMessages([]sdk.Message{sdk.UserMessage("conflicting")}),
			sdk.WithTools(tools),
		)
		if err == nil || !strings.Contains(err.Error(), "WithMessages conflicts") {
			t.Fatalf("error = %v, want WithMessages conflict", err)
		}
	})
}

// Resume with MaxSteps==0 applies the decisions and makes one model call;
// per the MaxSteps==0 contract, new tool calls that single call returns are
// not auto-executed — identically in both transports.
func TestResume_MaxStepsZero_NoNewToolExecution(t *testing.T) {
	newMP := func() *mockProvider {
		return &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
			return &sdk.GenerateResult{
				FinishReason: sdk.FinishReasonToolCalls,
				ToolCalls:    []sdk.ToolCall{{ToolCallID: "c9", ToolName: "notify", Input: nil}},
			}, nil
		}}
	}
	decisions := map[string]sdk.ToolDecision{
		"c2": {Decision: sdk.ToolDecisionApproved},
	}

	run := func(t *testing.T, stream bool) bool {
		executed := map[string]any{}
		var mu sync.Mutex
		opts := []sdk.GenerateOption{
			sdk.WithModel(mockModel(newMP())),
			sdk.WithTools(resumeTools(executed, &mu)),
		}
		if stream {
			sr, err := sdk.ResumeTextStream(context.Background(), pausedFixture(), decisions, opts...)
			if err != nil {
				t.Fatalf("stream: %v", err)
			}
			for range sr.Stream {
			}
		} else {
			if _, err := sdk.ResumeText(context.Background(), pausedFixture(), decisions, opts...); err != nil {
				t.Fatalf("generate: %v", err)
			}
		}
		mu.Lock()
		defer mu.Unlock()
		_, ok := executed["notify"]
		return ok
	}

	if gen, stream := run(t, false), run(t, true); gen || stream {
		t.Errorf("MaxSteps==0 must not auto-execute new tool calls: generate=%v stream=%v", gen, stream)
	}
}

// ApplyToolDecisions runs only the decision phase: side effects happen once,
// and generation over the completed conversation can be retried without
// re-executing approved tools. No model is required.
func TestApplyToolDecisions_SplitPhases(t *testing.T) {
	executed := map[string]any{}
	var mu sync.Mutex

	pause := pausedFixture()
	resolution, err := sdk.ApplyToolDecisions(context.Background(), pause,
		map[string]sdk.ToolDecision{
			"c2": {Decision: sdk.ToolDecisionApproved},
		},
		resumeTools(executed, &mu),
	)
	if err != nil {
		t.Fatalf("apply: %v", err)
	}
	if _, ok := executed["deploy"]; !ok {
		t.Fatal("approved call must execute during apply")
	}
	if len(resolution.Results) != 1 || resolution.Results[0].ToolCallID != "c2" {
		t.Fatalf("resolution results: %#v", resolution.Results)
	}
	if len(resolution.Messages) != 1 || resolution.Messages[0].Role != sdk.MessageRoleTool {
		t.Fatalf("resolution messages: %#v", resolution.Messages)
	}

	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		return &sdk.GenerateResult{Text: "done", FinishReason: sdk.FinishReasonStop}, nil
	}}
	completed := make([]sdk.Message, 0, len(pause.Messages)+len(resolution.Messages))
	completed = append(completed, pause.Messages...)
	completed = append(completed, resolution.Messages...)
	for retry := 0; retry < 2; retry++ {
		result, err := sdk.GenerateTextResult(context.Background(),
			sdk.WithModel(mockModel(mp)),
			sdk.WithMessages(completed),
			sdk.WithTools(resumeTools(executed, &mu)),
			sdk.WithMaxSteps(5),
		)
		if err != nil || result.Text != "done" {
			t.Fatalf("generate retry %d: err=%v", retry, err)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if len(executed) != 1 {
		t.Fatalf("tools must execute exactly once across retries: %v", executed)
	}
}

// Rejected decisions produce error results carrying the reason, and the
// resolution completes the conversation protocol.
func TestApplyToolDecisions_Rejection(t *testing.T) {
	executed := map[string]any{}
	var mu sync.Mutex

	resolution, err := sdk.ApplyToolDecisions(context.Background(), pausedFixture(),
		map[string]sdk.ToolDecision{
			"c2": {Decision: sdk.ToolDecisionRejected, Reason: "not during freeze"},
		},
		resumeTools(executed, &mu),
	)
	if err != nil {
		t.Fatalf("apply: %v", err)
	}
	if _, ok := executed["deploy"]; ok {
		t.Error("rejected call must not execute")
	}
	trp := resolution.Messages[0].Content[0].(sdk.ToolResultPart)
	if !trp.IsError || !strings.Contains(trp.Result.(string), "not during freeze") {
		t.Errorf("rejected result should be an error carrying the reason: %#v", trp)
	}
}

// ---------- integration test (requires OPENAI_API_KEY) ----------

// TestClient_ResumeText verifies against a real provider that the resumed
// conversation shape — an earlier assistant message's calls completed across
// split tool messages — is accepted end to end.
func TestClient_ResumeText(t *testing.T) {
	m := model(t)
	executed := false
	tools := []sdk.Tool{{
		Name:            "get_release_status",
		Description:     "Returns the current release status.",
		Parameters:      &jsonschema.Schema{Type: "object"},
		RequireApproval: true,
		Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
			executed = true
			return "v2.1 shipped to all regions", nil
		},
	}}

	paused, err := sdk.GenerateTextResult(context.Background(),
		sdk.WithModel(m),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("Check the release status and summarize it in one sentence.")}),
		sdk.WithTools(tools),
		sdk.WithMaxSteps(3), sdk.WithToolChoice("required"),
		sdk.WithApprovalHandler(func(_ context.Context, tc sdk.ToolCall) (sdk.ToolApprovalResult, error) {
			return sdk.ToolApprovalResult{Decision: sdk.ToolApprovalDecisionDeferred, ApprovalID: "a-" + tc.ToolCallID}, nil
		}),
	)
	if err != nil {
		t.Fatalf("GenerateTextResult: %v", err)
	}
	if paused.FinishReason != sdk.FinishReasonPaused || paused.Pause == nil {
		t.Fatalf("expected a paused run, got finish=%q pause=%v", paused.FinishReason, paused.Pause)
	}

	decisions := map[string]sdk.ToolDecision{}
	for _, d := range paused.Pause.Pending {
		decisions[d.ToolCall.ToolCallID] = sdk.ToolDecision{Decision: sdk.ToolDecisionApproved}
	}
	// Tool choice reverts to auto so the model can produce the final answer.
	resumed, err := sdk.ResumeText(context.Background(), *paused.Pause, decisions,
		sdk.WithModel(m), sdk.WithTools(tools), sdk.WithMaxSteps(3),
	)
	if err != nil {
		t.Fatalf("ResumeText: %v", err)
	}
	if !executed {
		t.Error("approved tool did not execute on resume")
	}
	t.Logf("resumed answer: %q", resumed.Text)
	if resumed.Text == "" {
		t.Error("expected a final answer after resume")
	}
}

// Regression battery for the review findings around the resume lifecycle:
// PrepareStep must not corrupt the pause, OnFinish must observe the
// resolution, System must travel with the pause, and a canceled context must
// surface as an error rather than a fabricated resolution.
func TestResume_LifecycleEdges(t *testing.T) {
	t.Run("pause after a completed step with PrepareStep", func(t *testing.T) {
		mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
			switch call {
			case 1:
				return &sdk.GenerateResult{
					FinishReason: sdk.FinishReasonToolCalls,
					ToolCalls:    []sdk.ToolCall{{ToolCallID: "c1", ToolName: "notify", Input: nil}},
				}, nil
			case 2:
				return &sdk.GenerateResult{
					FinishReason: sdk.FinishReasonToolCalls,
					ToolCalls:    []sdk.ToolCall{{ToolCallID: "c2", ToolName: "deploy", Input: map[string]any{"env": "prod"}}},
				}, nil
			case 3:
				// The resumed conversation must contain each message exactly once.
				order, _ := toolResultsByID(params.Messages)
				if len(order) != 2 || order[0] != "c1" || order[1] != "c2" {
					t.Fatalf("model saw tool results %v, want [c1 c2]", order)
				}
				return &sdk.GenerateResult{Text: "done", FinishReason: sdk.FinishReasonStop}, nil
			}
			t.Fatalf("unexpected provider call %d", call)
			return nil, nil
		}}
		executed := map[string]any{}
		var mu sync.Mutex
		tools := resumeTools(executed, &mu)
		handler := sdk.WithApprovalHandler(func(_ context.Context, tc sdk.ToolCall) (sdk.ToolApprovalResult, error) {
			return sdk.ToolApprovalResult{Decision: sdk.ToolApprovalDecisionDeferred, ApprovalID: "a-" + tc.ToolCallID}, nil
		})

		paused, err := sdk.GenerateTextResult(context.Background(),
			sdk.WithModel(mockModel(mp)),
			sdk.WithSystem("deploy bot rules"),
			sdk.WithMessages([]sdk.Message{sdk.UserMessage("release")}),
			sdk.WithTools(tools), sdk.WithMaxSteps(5), handler,
			sdk.WithPrepareStep(func(p *sdk.GenerateParams) *sdk.GenerateParams { return nil }),
		)
		if err != nil {
			t.Fatalf("paused run: %v", err)
		}
		if paused.Pause == nil {
			t.Fatal("expected a pause")
		}
		if paused.Pause.System != "deploy bot rules" {
			t.Fatalf("pause system: %q", paused.Pause.System)
		}

		resumed, err := sdk.ResumeText(context.Background(), *paused.Pause,
			map[string]sdk.ToolDecision{"c2": {Decision: sdk.ToolDecisionApproved}},
			sdk.WithModel(mockModel(mp)), sdk.WithTools(tools), sdk.WithMaxSteps(5), handler,
		)
		if err != nil {
			t.Fatalf("resume: %v", err)
		}
		if resumed.Text != "done" {
			t.Fatalf("text: %q", resumed.Text)
		}
	})

	t.Run("OnFinish observes the resolution", func(t *testing.T) {
		mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
			return &sdk.GenerateResult{Text: "done", FinishReason: sdk.FinishReasonStop}, nil
		}}
		executed := map[string]any{}
		var mu sync.Mutex

		var seen *sdk.ToolApprovalResolution
		_, err := sdk.ResumeText(context.Background(), pausedFixture(),
			map[string]sdk.ToolDecision{"c2": {Decision: sdk.ToolDecisionApproved}},
			sdk.WithModel(mockModel(mp)),
			sdk.WithTools(resumeTools(executed, &mu)),
			sdk.WithMaxSteps(5),
			sdk.WithOnFinish(func(r *sdk.GenerateResult) { seen = r.Resume }),
		)
		if err != nil {
			t.Fatalf("resume: %v", err)
		}
		if seen == nil || len(seen.Results) != 1 || seen.Results[0].ToolCallID != "c2" {
			t.Fatalf("OnFinish must observe the resolution: %#v", seen)
		}
	})

	t.Run("conflicting WithSystem is rejected", func(t *testing.T) {
		executed := map[string]any{}
		var mu sync.Mutex
		pause := pausedFixture()
		pause.System = "original rules"
		_, err := sdk.ResumeText(context.Background(), pause,
			map[string]sdk.ToolDecision{"c2": {Decision: sdk.ToolDecisionApproved}},
			sdk.WithModel(mockModel(&mockProvider{})),
			sdk.WithSystem("different rules"),
			sdk.WithTools(resumeTools(executed, &mu)),
		)
		if err == nil || !strings.Contains(err.Error(), "WithSystem conflicts") {
			t.Fatalf("error = %v, want WithSystem conflict", err)
		}
	})

	t.Run("canceled context fails instead of fabricating results", func(t *testing.T) {
		executed := map[string]any{}
		var mu sync.Mutex
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := sdk.ApplyToolDecisions(ctx, pausedFixture(),
			map[string]sdk.ToolDecision{"c2": {Decision: sdk.ToolDecisionApproved}},
			resumeTools(executed, &mu),
		)
		if err == nil || !strings.Contains(err.Error(), "context ended while applying decisions") {
			t.Fatalf("error = %v, want context-ended error", err)
		}
	})

	t.Run("failed execution is distinguishable in the resolution", func(t *testing.T) {
		tools := []sdk.Tool{{
			Name: "deploy", Parameters: &jsonschema.Schema{Type: "object"}, RequireApproval: true,
			Execute: func(ctx *sdk.ToolExecContext, in any) (any, error) {
				return nil, errors.New("quota exceeded")
			},
		}, {
			Name: "notify", Parameters: &jsonschema.Schema{Type: "object"},
			Execute: func(ctx *sdk.ToolExecContext, in any) (any, error) { return "sent", nil },
		}}
		resolution, err := sdk.ApplyToolDecisions(context.Background(), pausedFixture(),
			map[string]sdk.ToolDecision{"c2": {Decision: sdk.ToolDecisionApproved}}, tools)
		if err != nil {
			t.Fatalf("apply: %v", err)
		}
		if len(resolution.Results) != 1 || !resolution.Results[0].IsError {
			t.Fatalf("failed execution must carry IsError: %#v", resolution.Results)
		}
	})
}
