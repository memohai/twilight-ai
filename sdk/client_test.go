package sdk_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/memohai/twilight-ai/internal/testutil"
	"github.com/memohai/twilight-ai/provider/openai/completions"
	"github.com/memohai/twilight-ai/sdk"
)

func TestMain(m *testing.M) {
	testutil.LoadEnv()
	os.Exit(m.Run())
}

func envOrSkip(t *testing.T, key string) string {
	t.Helper()
	v := os.Getenv(key)
	if v == "" {
		t.Skipf("skipping: %s not set", key)
	}
	return v
}

func newProvider(t *testing.T) *completions.Provider {
	t.Helper()
	apiKey := envOrSkip(t, "OPENAI_API_KEY")
	opts := []completions.Option{completions.WithAPIKey(apiKey)}
	if base := os.Getenv("OPENAI_BASE_URL"); base != "" {
		opts = append(opts, completions.WithBaseURL(base))
	}
	return completions.New(opts...)
}

func model(t *testing.T) *sdk.Model {
	t.Helper()
	id := os.Getenv("OPENAI_MODEL")
	if id == "" {
		id = "gpt-4o-mini"
	}
	return newProvider(t).ChatModel(id)
}

// ---------- integration tests (require OPENAI_API_KEY) ----------

func TestClient_GenerateText(t *testing.T) {
	text, err := sdk.GenerateText(context.Background(),
		sdk.WithModel(model(t)),
		sdk.WithMessages([]sdk.Message{
			sdk.UserMessage("Say hi in one word."),
		}),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}
	t.Logf("response: %q", text)
	if text == "" {
		t.Error("expected non-empty response")
	}
}

func TestClient_GenerateTextResult(t *testing.T) {
	result, err := sdk.GenerateTextResult(context.Background(),
		sdk.WithModel(model(t)),
		sdk.WithMessages([]sdk.Message{
			sdk.UserMessage("Say hi in one word."),
		}),
	)
	if err != nil {
		t.Fatalf("GenerateTextResult: %v", err)
	}
	t.Logf("text=%q finish=%s input=%d output=%d",
		result.Text, result.FinishReason,
		result.Usage.InputTokens, result.Usage.OutputTokens)

	if result.Text == "" {
		t.Error("expected non-empty text")
	}
	if result.FinishReason != sdk.FinishReasonStop {
		t.Errorf("expected stop, got %s", result.FinishReason)
	}
}

func TestClient_StreamText(t *testing.T) {
	sr, err := sdk.StreamText(context.Background(),
		sdk.WithModel(model(t)),
		sdk.WithMessages([]sdk.Message{
			sdk.UserMessage("Count from 1 to 3."),
		}),
	)
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	var text string
	for part := range sr.Stream {
		switch p := part.(type) {
		case *sdk.TextDeltaPart:
			text += p.Text
		case *sdk.ErrorPart:
			t.Fatalf("stream error: %v", p.Error)
		case *sdk.FinishPart:
			t.Logf("finish=%s tokens=%d", p.FinishReason, p.TotalUsage.TotalTokens)
		}
	}
	t.Logf("streamed: %q", text)
	if text == "" {
		t.Error("expected non-empty streamed text")
	}
}

func TestClient_StreamText_ToResult(t *testing.T) {
	sr, err := sdk.StreamText(context.Background(),
		sdk.WithModel(model(t)),
		sdk.WithMessages([]sdk.Message{
			sdk.UserMessage("Say hello in one word."),
		}),
	)
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	result, err := sr.ToResult()
	if err != nil {
		t.Fatalf("ToResult: %v", err)
	}
	t.Logf("text=%q finish=%s", result.Text, result.FinishReason)
	if result.Text == "" {
		t.Error("expected non-empty text")
	}
}

func TestClient_WithSystem(t *testing.T) {
	text, err := sdk.GenerateText(context.Background(),
		sdk.WithModel(model(t)),
		sdk.WithSystem("You always respond with exactly one word."),
		sdk.WithMessages([]sdk.Message{
			sdk.UserMessage("Greet me."),
		}),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}
	t.Logf("response: %q", text)
	if text == "" {
		t.Error("expected non-empty response")
	}
}

func TestClient_NoModel(t *testing.T) {
	_, err := sdk.GenerateText(context.Background(),
		sdk.WithMessages([]sdk.Message{
			sdk.UserMessage("Hi"),
		}),
	)
	if err == nil {
		t.Fatal("expected error for nil model")
	}
}

// ---------- mockProvider for unit tests ----------

type mockProvider struct {
	calls         int
	handler       func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error)
	streamHandler func(call int, params sdk.GenerateParams) (*sdk.StreamResult, error)
}

func (m *mockProvider) Name() string { return "mock" }
func (m *mockProvider) ListModels(_ context.Context) ([]sdk.Model, error) {
	return nil, nil
}
func (m *mockProvider) Test(_ context.Context) *sdk.ProviderTestResult {
	return &sdk.ProviderTestResult{Status: sdk.ProviderStatusOK, Message: "ok"}
}
func (m *mockProvider) TestModel(_ context.Context, _ string) (*sdk.ModelTestResult, error) {
	return &sdk.ModelTestResult{Supported: true, Message: "supported"}, nil
}

func (m *mockProvider) DoGenerate(_ context.Context, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
	m.calls++
	return m.handler(m.calls, params)
}

func (m *mockProvider) DoStream(_ context.Context, params sdk.GenerateParams) (*sdk.StreamResult, error) {
	if m.streamHandler != nil {
		m.calls++
		return m.streamHandler(m.calls, params)
	}
	result, err := m.DoGenerate(context.Background(), params)
	if err != nil {
		return nil, err
	}

	ch := make(chan sdk.StreamPart, 16)
	go func() {
		defer close(ch)
		ch <- &sdk.StartPart{}
		ch <- &sdk.StartStepPart{}
		if result.Text != "" {
			ch <- &sdk.TextStartPart{ID: "mock"}
			ch <- &sdk.TextDeltaPart{ID: "mock", Text: result.Text}
			ch <- &sdk.TextEndPart{ID: "mock"}
		}
		for _, tc := range result.ToolCalls {
			ch <- &sdk.StreamToolCallPart{
				ToolCallID: tc.ToolCallID,
				ToolName:   tc.ToolName,
				Input:      tc.Input,
			}
		}
		ch <- &sdk.FinishStepPart{
			FinishReason: result.FinishReason,
			Usage:        result.Usage,
			Response:     result.Response,
		}
		ch <- &sdk.FinishPart{
			FinishReason: result.FinishReason,
			TotalUsage:   result.Usage,
		}
	}()
	return &sdk.StreamResult{Stream: ch}, nil
}

func mockModel(p *mockProvider) *sdk.Model {
	return &sdk.Model{ID: "mock-model", Provider: p}
}

// ---------- unit tests: tool auto-execution ----------

func TestClient_GenerateTextResult_ToolAutoExec(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		if call == 1 {
			return &sdk.GenerateResult{
				FinishReason: sdk.FinishReasonToolCalls,
				ToolCalls: []sdk.ToolCall{{
					ToolCallID: "c1",
					ToolName:   "add",
					Input:      map[string]any{"a": float64(2), "b": float64(3)},
				}},
				Usage: sdk.Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
			}, nil
		}
		// Second call: model sees tool result and responds with text.
		return &sdk.GenerateResult{
			Text:         "The sum is 5.",
			FinishReason: sdk.FinishReasonStop,
			Usage:        sdk.Usage{InputTokens: 20, OutputTokens: 8, TotalTokens: 28},
		}, nil
	}}

	result, err := sdk.GenerateTextResult(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("Add 2 and 3")}),
		sdk.WithTools([]sdk.Tool{{
			Name:       "add",
			Parameters: &jsonschema.Schema{Type: "object"},
			Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
				m := input.(map[string]any)
				return m["a"].(float64) + m["b"].(float64), nil
			},
		}}),
		sdk.WithMaxSteps(5),
	)
	if err != nil {
		t.Fatalf("GenerateTextResult: %v", err)
	}

	if result.Text != "The sum is 5." {
		t.Errorf("text: got %q", result.Text)
	}
	if mp.calls != 2 {
		t.Errorf("expected 2 provider calls, got %d", mp.calls)
	}
	if result.Usage.TotalTokens != 43 {
		t.Errorf("expected accumulated total tokens 43, got %d", result.Usage.TotalTokens)
	}
}

func TestClient_GenerateTextResult_NoAutoExec_WhenMaxStepsZero(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		return &sdk.GenerateResult{
			FinishReason: sdk.FinishReasonToolCalls,
			ToolCalls: []sdk.ToolCall{{
				ToolCallID: "c1",
				ToolName:   "add",
				Input:      map[string]any{"a": float64(1), "b": float64(2)},
			}},
		}, nil
	}}

	result, err := sdk.GenerateTextResult(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("Add 1 and 2")}),
		sdk.WithTools([]sdk.Tool{{
			Name:       "add",
			Parameters: &jsonschema.Schema{Type: "object"},
			Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
				return nil, fmt.Errorf("should not be called")
			},
		}}),
		// MaxSteps defaults to 0 = single call, no auto-execution
	)
	if err != nil {
		t.Fatalf("GenerateTextResult: %v", err)
	}

	if mp.calls != 1 {
		t.Errorf("expected 1 provider call, got %d", mp.calls)
	}
	if result.FinishReason != sdk.FinishReasonToolCalls {
		t.Errorf("expected tool-calls finish, got %s", result.FinishReason)
	}
}

func TestClient_GenerateTextResult_UnlimitedSteps(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		if call <= 3 {
			return &sdk.GenerateResult{
				FinishReason: sdk.FinishReasonToolCalls,
				ToolCalls: []sdk.ToolCall{{
					ToolCallID: fmt.Sprintf("c%d", call),
					ToolName:   "step",
					Input:      map[string]any{"n": float64(call)},
				}},
				Usage: sdk.Usage{TotalTokens: 10},
			}, nil
		}
		return &sdk.GenerateResult{
			Text:         "done",
			FinishReason: sdk.FinishReasonStop,
			Usage:        sdk.Usage{TotalTokens: 10},
		}, nil
	}}

	result, err := sdk.GenerateTextResult(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("go")}),
		sdk.WithTools([]sdk.Tool{{
			Name:       "step",
			Parameters: &jsonschema.Schema{Type: "object"},
			Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
				return "ok", nil
			},
		}}),
		sdk.WithMaxSteps(-1),
	)
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	if mp.calls != 4 {
		t.Errorf("expected 4 calls (3 tool rounds + 1 final), got %d", mp.calls)
	}
	if result.Text != "done" {
		t.Errorf("text: got %q", result.Text)
	}
	if result.Usage.TotalTokens != 40 {
		t.Errorf("total tokens: got %d, want 40", result.Usage.TotalTokens)
	}
}

// ---------- unit tests: callbacks ----------

func TestClient_GenerateTextResult_Callbacks(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		if call == 1 {
			return &sdk.GenerateResult{
				FinishReason: sdk.FinishReasonToolCalls,
				ToolCalls: []sdk.ToolCall{{
					ToolCallID: "c1", ToolName: "ping", Input: nil,
				}},
				Usage: sdk.Usage{TotalTokens: 5},
			}, nil
		}
		return &sdk.GenerateResult{
			Text:         "pong",
			FinishReason: sdk.FinishReasonStop,
			Usage:        sdk.Usage{TotalTokens: 5},
		}, nil
	}}

	var stepResults []*sdk.StepResult
	var finishResult *sdk.GenerateResult

	_, err := sdk.GenerateTextResult(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("ping")}),
		sdk.WithTools([]sdk.Tool{{
			Name:       "ping",
			Parameters: &jsonschema.Schema{Type: "object"},
			Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
				return "pong", nil
			},
		}}),
		sdk.WithMaxSteps(5),
		sdk.WithOnStep(func(sr *sdk.StepResult) *sdk.GenerateParams {
			stepResults = append(stepResults, sr)
			return nil
		}),
		sdk.WithOnFinish(func(r *sdk.GenerateResult) {
			finishResult = r
		}),
	)
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	if len(stepResults) != 2 {
		t.Fatalf("expected 2 step callbacks, got %d", len(stepResults))
	}
	if stepResults[0].FinishReason != sdk.FinishReasonToolCalls {
		t.Errorf("step 0 finish: got %s", stepResults[0].FinishReason)
	}
	if stepResults[1].FinishReason != sdk.FinishReasonStop {
		t.Errorf("step 1 finish: got %s", stepResults[1].FinishReason)
	}
	if finishResult == nil {
		t.Fatal("onFinish not called")
	}
	if finishResult.Usage.TotalTokens != 10 {
		t.Errorf("onFinish total tokens: got %d, want 10", finishResult.Usage.TotalTokens)
	}
}

func TestClient_GenerateTextResult_PrepareStep(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		if call == 1 {
			return &sdk.GenerateResult{
				FinishReason: sdk.FinishReasonToolCalls,
				ToolCalls: []sdk.ToolCall{{
					ToolCallID: "c1", ToolName: "fetch", Input: nil,
				}},
			}, nil
		}
		if params.System != "injected-system" {
			t.Errorf("prepareStep did not inject system: got %q", params.System)
		}
		return &sdk.GenerateResult{
			Text:         "ok",
			FinishReason: sdk.FinishReasonStop,
		}, nil
	}}

	_, err := sdk.GenerateTextResult(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("go")}),
		sdk.WithTools([]sdk.Tool{{
			Name:       "fetch",
			Parameters: &jsonschema.Schema{Type: "object"},
			Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
				return "data", nil
			},
		}}),
		sdk.WithMaxSteps(5),
		sdk.WithPrepareStep(func(p *sdk.GenerateParams) *sdk.GenerateParams {
			p.System = "injected-system"
			return nil
		}),
	)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
}

func TestClient_GenerateTextResult_PreservesDeveloperRoleAcrossToolSteps(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		if call == 1 {
			if len(params.Messages) != 2 || params.Messages[0].Role != sdk.MessageRoleDeveloper {
				t.Fatalf("first step messages: %+v", params.Messages)
			}
			return &sdk.GenerateResult{
				FinishReason: sdk.FinishReasonToolCalls,
				ToolCalls: []sdk.ToolCall{{
					ToolCallID: "c1", ToolName: "fetch", Input: nil,
				}},
			}, nil
		}

		wantRoles := []sdk.MessageRole{
			sdk.MessageRoleDeveloper,
			sdk.MessageRoleUser,
			sdk.MessageRoleAssistant,
			sdk.MessageRoleTool,
		}
		if len(params.Messages) != len(wantRoles) {
			t.Fatalf("second step message count: got %d, want %d", len(params.Messages), len(wantRoles))
		}
		for i, want := range wantRoles {
			if params.Messages[i].Role != want {
				t.Fatalf("second step message %d role: got %q, want %q", i, params.Messages[i].Role, want)
			}
		}
		return &sdk.GenerateResult{Text: "ok", FinishReason: sdk.FinishReasonStop}, nil
	}}

	_, err := sdk.GenerateTextResult(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{
			sdk.DeveloperMessage("application policy"),
			sdk.UserMessage("go"),
		}),
		sdk.WithTools([]sdk.Tool{{
			Name:       "fetch",
			Parameters: &jsonschema.Schema{Type: "object"},
			Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
				return "data", nil
			},
		}}),
		sdk.WithMaxSteps(5),
	)
	if err != nil {
		t.Fatalf("GenerateTextResult: %v", err)
	}
}

// ---------- unit tests: approval flow ----------

func TestClient_GenerateTextResult_ApprovalApproved(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		if call == 1 {
			return &sdk.GenerateResult{
				FinishReason: sdk.FinishReasonToolCalls,
				ToolCalls: []sdk.ToolCall{{
					ToolCallID: "c1", ToolName: "dangerous", Input: nil,
				}},
			}, nil
		}
		return &sdk.GenerateResult{
			Text:         "executed",
			FinishReason: sdk.FinishReasonStop,
		}, nil
	}}

	executed := false
	result, err := sdk.GenerateTextResult(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("do it")}),
		sdk.WithTools([]sdk.Tool{{
			Name:            "dangerous",
			Parameters:      &jsonschema.Schema{Type: "object"},
			RequireApproval: true,
			Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
				executed = true
				return "done", nil
			},
		}}),
		sdk.WithMaxSteps(5),
		sdk.WithApprovalHandlerBool(func(_ context.Context, tc sdk.ToolCall) (bool, error) {
			return true, nil
		}),
	)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if !executed {
		t.Error("tool was not executed despite approval")
	}
	if result.Text != "executed" {
		t.Errorf("text: got %q", result.Text)
	}
}

func TestClient_GenerateTextResult_ApprovalDenied(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		if call == 1 {
			return &sdk.GenerateResult{
				FinishReason: sdk.FinishReasonToolCalls,
				ToolCalls: []sdk.ToolCall{{
					ToolCallID: "c1", ToolName: "dangerous", Input: nil,
				}},
			}, nil
		}
		return &sdk.GenerateResult{
			Text:         "denied-response",
			FinishReason: sdk.FinishReasonStop,
		}, nil
	}}

	executed := false
	result, err := sdk.GenerateTextResult(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("do it")}),
		sdk.WithTools([]sdk.Tool{{
			Name:            "dangerous",
			Parameters:      &jsonschema.Schema{Type: "object"},
			RequireApproval: true,
			Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
				executed = true
				return "done", nil
			},
		}}),
		sdk.WithMaxSteps(5),
		sdk.WithApprovalHandlerBool(func(_ context.Context, tc sdk.ToolCall) (bool, error) {
			return false, nil
		}),
	)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if executed {
		t.Error("tool should not have been executed after denial")
	}
	if result.Text != "denied-response" {
		t.Errorf("text: got %q", result.Text)
	}
}

func TestClient_GenerateTextResult_ApprovalDeferred(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		if call != 1 {
			t.Fatalf("unexpected provider call %d", call)
		}
		return &sdk.GenerateResult{
			FinishReason: sdk.FinishReasonToolCalls,
			ToolCalls: []sdk.ToolCall{{
				ToolCallID: "c1", ToolName: "dangerous", Input: nil,
			}},
		}, nil
	}}

	executed := false
	result, err := sdk.GenerateTextResult(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("do it")}),
		sdk.WithTools([]sdk.Tool{{
			Name:            "dangerous",
			Parameters:      &jsonschema.Schema{Type: "object"},
			RequireApproval: true,
			Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
				executed = true
				return "done", nil
			},
		}}),
		sdk.WithMaxSteps(5),
		sdk.WithApprovalHandler(func(_ context.Context, tc sdk.ToolCall) (sdk.ToolApprovalResult, error) {
			return sdk.ToolApprovalResult{
				Decision:   sdk.ToolApprovalDecisionDeferred,
				ApprovalID: "approval-1",
			}, nil
		}),
	)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if executed {
		t.Error("tool should not have executed while approval is deferred")
	}
	if result.FinishReason != sdk.FinishReasonPaused {
		t.Fatalf("finish reason: got %q, want paused", result.FinishReason)
	}
	if result.Pause == nil || len(result.Pause.Pending) != 1 {
		t.Fatalf("pause: %#v", result.Pause)
	}
	if d := result.Pause.Pending[0]; d.ToolCall.ToolCallID != "c1" || d.Approval.ApprovalID != "approval-1" {
		t.Fatalf("deferred[0]: %#v", d)
	}
	if len(result.Messages) != 1 || len(result.Messages[0].Content) == 0 {
		t.Fatalf("expected assistant tool-call message, got %#v", result.Messages)
	}
	if mp.calls != 1 {
		t.Fatalf("expected one provider call, got %d", mp.calls)
	}
}

func TestClient_GenerateTextResult_ApprovalNoHandler(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		if call == 1 {
			return &sdk.GenerateResult{
				FinishReason: sdk.FinishReasonToolCalls,
				ToolCalls: []sdk.ToolCall{{
					ToolCallID: "c1", ToolName: "dangerous", Input: nil,
				}},
			}, nil
		}
		return &sdk.GenerateResult{
			Text:         "handled-denial",
			FinishReason: sdk.FinishReasonStop,
		}, nil
	}}

	executed := false
	_, err := sdk.GenerateTextResult(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("do it")}),
		sdk.WithTools([]sdk.Tool{{
			Name:            "dangerous",
			Parameters:      &jsonschema.Schema{Type: "object"},
			RequireApproval: true,
			Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
				executed = true
				return "done", nil
			},
		}}),
		sdk.WithMaxSteps(5),
		// No approval handler: should deny
	)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if executed {
		t.Error("tool should not execute when RequireApproval=true and no handler")
	}
}

// Regression: a deferral must not truncate or discard the batch. One step
// mixes an approval-free call, an approved call, a rejected call, and two
// deferred calls: every gated call reaches the handler, resolved calls
// execute (or record their denial), the step's tool message covers exactly
// the resolved calls, and the run pauses with every pending approval listed.
func TestClient_GenerateTextResult_PausesOnDeferredApprovals(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		if call != 1 {
			t.Fatalf("unexpected provider call %d", call)
		}
		return &sdk.GenerateResult{
			FinishReason: sdk.FinishReasonToolCalls,
			ToolCalls: []sdk.ToolCall{
				{ToolCallID: "c1", ToolName: "safe", Input: nil},
				{ToolCallID: "c2", ToolName: "write", Input: nil},
				{ToolCallID: "c3", ToolName: "risky", Input: nil},
				{ToolCallID: "c4", ToolName: "deploy", Input: nil},
				{ToolCallID: "c5", ToolName: "notify", Input: nil},
			},
		}, nil
	}}

	executed := map[string]bool{}
	var mu sync.Mutex
	execute := func(name string) sdk.ToolExecuteFunc {
		return func(ctx *sdk.ToolExecContext, input any) (any, error) {
			mu.Lock()
			executed[name] = true
			mu.Unlock()
			return name + "-ok", nil
		}
	}
	var handlerSaw []string

	result, err := sdk.GenerateTextResult(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("go")}),
		sdk.WithTools([]sdk.Tool{
			{Name: "safe", Parameters: &jsonschema.Schema{Type: "object"}, Execute: execute("safe")},
			{Name: "write", Parameters: &jsonschema.Schema{Type: "object"}, RequireApproval: true, Execute: execute("write")},
			{Name: "risky", Parameters: &jsonschema.Schema{Type: "object"}, RequireApproval: true, Execute: execute("risky")},
			{Name: "deploy", Parameters: &jsonschema.Schema{Type: "object"}, RequireApproval: true, Execute: execute("deploy")},
			{Name: "notify", Parameters: &jsonschema.Schema{Type: "object"}, RequireApproval: true, Execute: execute("notify")},
		}),
		sdk.WithMaxSteps(5),
		sdk.WithApprovalHandler(func(_ context.Context, tc sdk.ToolCall) (sdk.ToolApprovalResult, error) {
			handlerSaw = append(handlerSaw, tc.ToolName)
			switch tc.ToolName {
			case "write":
				return sdk.ToolApprovalResult{Decision: sdk.ToolApprovalDecisionApproved}, nil
			case "risky":
				return sdk.ToolApprovalResult{Decision: sdk.ToolApprovalDecisionRejected, Reason: "nope"}, nil
			default:
				return sdk.ToolApprovalResult{Decision: sdk.ToolApprovalDecisionDeferred, ApprovalID: "approval-" + tc.ToolName}, nil
			}
		}),
	)
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	if len(handlerSaw) != 4 || handlerSaw[0] != "write" || handlerSaw[1] != "risky" || handlerSaw[2] != "deploy" || handlerSaw[3] != "notify" {
		t.Fatalf("approval handler saw %v, want [write risky deploy notify]", handlerSaw)
	}
	if !executed["safe"] || !executed["write"] {
		t.Errorf("resolved calls must execute despite sibling deferrals: %v", executed)
	}
	if executed["risky"] || executed["deploy"] || executed["notify"] {
		t.Errorf("rejected/deferred calls must not run: %v", executed)
	}
	if result.FinishReason != sdk.FinishReasonPaused {
		t.Fatalf("finish reason: got %q, want paused", result.FinishReason)
	}
	if result.Pause == nil || len(result.Pause.Pending) != 2 {
		t.Fatalf("pause: %#v", result.Pause)
	}
	if d := result.Pause.Pending[0]; d.ToolCall.ToolCallID != "c4" || d.Approval.ApprovalID != "approval-deploy" {
		t.Errorf("deferred[0]: %#v", d)
	}
	if d := result.Pause.Pending[1]; d.ToolCall.ToolCallID != "c5" || d.Approval.ApprovalID != "approval-notify" {
		t.Errorf("deferred[1]: %#v", d)
	}

	step := result.Steps[len(result.Steps)-1]
	if step.FinishReason != sdk.FinishReasonToolCalls {
		t.Errorf("paused step keeps the provider finish reason, got %q", step.FinishReason)
	}
	if len(step.ToolResults) != 3 || step.ToolResults[0].ToolCallID != "c1" || step.ToolResults[1].ToolCallID != "c2" || step.ToolResults[2].ToolCallID != "c3" {
		t.Fatalf("resolved tool results: %#v", step.ToolResults)
	}
	if len(step.Messages) != 2 || step.Messages[1].Role != sdk.MessageRoleTool {
		t.Fatalf("step messages: %#v", step.Messages)
	}
	gotIDs := make([]string, 0, len(step.Messages[1].Content))
	rejectedIsError := false
	for _, part := range step.Messages[1].Content {
		trp, ok := part.(sdk.ToolResultPart)
		if !ok {
			t.Fatalf("unexpected part in tool message: %#v", part)
		}
		gotIDs = append(gotIDs, trp.ToolCallID)
		if trp.ToolCallID == "c3" {
			rejectedIsError = trp.IsError
		}
	}
	if len(gotIDs) != 3 || gotIDs[0] != "c1" || gotIDs[1] != "c2" || gotIDs[2] != "c3" {
		t.Errorf("tool message covers %v, want [c1 c2 c3]", gotIDs)
	}
	if !rejectedIsError {
		t.Error("rejected call's result should be an error result")
	}
	if mp.calls != 1 {
		t.Fatalf("expected one provider call, got %d", mp.calls)
	}
}

// ---------- unit tests: streaming with tool execution ----------

func TestClient_StreamText_ToolAutoExec(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		if call == 1 {
			return &sdk.GenerateResult{
				FinishReason: sdk.FinishReasonToolCalls,
				ToolCalls: []sdk.ToolCall{{
					ToolCallID: "c1",
					ToolName:   "greet",
					Input:      map[string]any{"name": "Alice"},
				}},
				Usage: sdk.Usage{TotalTokens: 10},
			}, nil
		}
		return &sdk.GenerateResult{
			Text:         "Hello, Alice!",
			FinishReason: sdk.FinishReasonStop,
			Usage:        sdk.Usage{TotalTokens: 10},
		}, nil
	}}

	sr, err := sdk.StreamText(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("greet Alice")}),
		sdk.WithTools([]sdk.Tool{{
			Name:       "greet",
			Parameters: &jsonschema.Schema{Type: "object"},
			Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
				m := input.(map[string]any)
				return fmt.Sprintf("greeting for %s", m["name"]), nil
			},
		}}),
		sdk.WithMaxSteps(5),
	)
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	var text string
	var gotToolResult bool
	var gotFinish bool
	for part := range sr.Stream {
		switch p := part.(type) {
		case *sdk.TextDeltaPart:
			text += p.Text
		case *sdk.StreamToolResultPart:
			gotToolResult = true
			if p.ToolName != "greet" {
				t.Errorf("tool result name: got %q", p.ToolName)
			}
		case *sdk.FinishPart:
			gotFinish = true
		case *sdk.ErrorPart:
			t.Fatalf("stream error: %v", p.Error)
		}
	}

	if text != "Hello, Alice!" {
		t.Errorf("text: got %q", text)
	}
	if !gotToolResult {
		t.Error("expected StreamToolResultPart")
	}
	if !gotFinish {
		t.Error("expected FinishPart")
	}
	if mp.calls != 2 {
		t.Errorf("expected 2 provider calls, got %d", mp.calls)
	}
}

func TestClient_StreamText_ToolProgress(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		if call == 1 {
			return &sdk.GenerateResult{
				FinishReason: sdk.FinishReasonToolCalls,
				ToolCalls: []sdk.ToolCall{{
					ToolCallID: "c1", ToolName: "run_cmd", Input: nil,
				}},
			}, nil
		}
		return &sdk.GenerateResult{
			Text:         "command finished",
			FinishReason: sdk.FinishReasonStop,
		}, nil
	}}

	sr, err := sdk.StreamText(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("run")}),
		sdk.WithTools([]sdk.Tool{{
			Name:       "run_cmd",
			Parameters: &jsonschema.Schema{Type: "object"},
			Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
				ctx.SendProgress("line 1\n")
				ctx.SendProgress("line 2\n")
				return "exit 0", nil
			},
		}}),
		sdk.WithMaxSteps(5),
	)
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	var progressParts []string
	for part := range sr.Stream {
		switch p := part.(type) {
		case *sdk.ToolProgressPart:
			progressParts = append(progressParts, p.Content.(string))
		case *sdk.ErrorPart:
			t.Fatalf("stream error: %v", p.Error)
		}
	}

	if len(progressParts) != 2 {
		t.Fatalf("expected 2 progress parts, got %d", len(progressParts))
	}
	if progressParts[0] != "line 1\n" || progressParts[1] != "line 2\n" {
		t.Errorf("progress: got %v", progressParts)
	}
}

func TestClient_StreamText_ApprovalFlow(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		if call == 1 {
			return &sdk.GenerateResult{
				FinishReason: sdk.FinishReasonToolCalls,
				ToolCalls: []sdk.ToolCall{{
					ToolCallID: "c1", ToolName: "rm_rf", Input: nil,
				}},
			}, nil
		}
		return &sdk.GenerateResult{
			Text:         "denied gracefully",
			FinishReason: sdk.FinishReasonStop,
		}, nil
	}}

	sr, err := sdk.StreamText(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("delete")}),
		sdk.WithTools([]sdk.Tool{{
			Name:            "rm_rf",
			Parameters:      &jsonschema.Schema{Type: "object"},
			RequireApproval: true,
			Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
				return "deleted", nil
			},
		}}),
		sdk.WithMaxSteps(5),
		sdk.WithApprovalHandlerBool(func(_ context.Context, tc sdk.ToolCall) (bool, error) {
			return false, nil
		}),
	)
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	var gotApprovalReq, gotDenied bool
	for part := range sr.Stream {
		switch p := part.(type) {
		case *sdk.ToolApprovalRequestPart:
			gotApprovalReq = true
		case *sdk.ToolOutputDeniedPart:
			gotDenied = true
		case *sdk.ErrorPart:
			t.Fatalf("stream error: %v", p.Error)
		}
	}

	if !gotApprovalReq {
		t.Error("expected ToolApprovalRequestPart")
	}
	if !gotDenied {
		t.Error("expected ToolOutputDeniedPart")
	}
}

func TestClient_StreamText_ApprovalDeferred(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		if call != 1 {
			t.Fatalf("unexpected provider call %d", call)
		}
		return &sdk.GenerateResult{
			FinishReason: sdk.FinishReasonToolCalls,
			ToolCalls: []sdk.ToolCall{{
				ToolCallID: "c1", ToolName: "rm_rf", Input: nil,
			}},
		}, nil
	}}

	executed := false
	sr, err := sdk.StreamText(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("delete")}),
		sdk.WithTools([]sdk.Tool{{
			Name:            "rm_rf",
			Parameters:      &jsonschema.Schema{Type: "object"},
			RequireApproval: true,
			Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
				executed = true
				return "deleted", nil
			},
		}}),
		sdk.WithMaxSteps(5),
		sdk.WithApprovalHandler(func(_ context.Context, tc sdk.ToolCall) (sdk.ToolApprovalResult, error) {
			return sdk.ToolApprovalResult{
				Decision:   sdk.ToolApprovalDecisionDeferred,
				ApprovalID: "approval-1",
			}, nil
		}),
	)
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	var gotApprovalReq, gotFinish bool
	for part := range sr.Stream {
		switch p := part.(type) {
		case *sdk.ToolApprovalRequestPart:
			gotApprovalReq = true
			if p.ApprovalID != "approval-1" {
				t.Fatalf("approval id: got %q", p.ApprovalID)
			}
		case *sdk.StreamToolResultPart:
			t.Fatal("did not expect tool result while approval is deferred")
		case *sdk.ErrorPart:
			t.Fatalf("stream error: %v", p.Error)
		case *sdk.FinishPart:
			gotFinish = true
		}
	}
	if executed {
		t.Error("tool should not have executed while approval is deferred")
	}
	if !gotApprovalReq {
		t.Error("expected ToolApprovalRequestPart")
	}
	if !gotFinish {
		t.Error("expected FinishPart")
	}
	if sr.Pause == nil || len(sr.Pause.Pending) != 1 || sr.Pause.Pending[0].Approval.ApprovalID != "approval-1" {
		t.Fatalf("missing pause: %#v", sr.Pause)
	}
	if len(sr.Messages) != 1 {
		t.Fatalf("expected assistant tool-call message, got %#v", sr.Messages)
	}
	if mp.calls != 1 {
		t.Fatalf("expected one provider call, got %d", mp.calls)
	}
}

// Regression: stream mode announces one ToolApprovalRequestPart per deferred
// call and marks the pause in-band — FinishPart carries FinishReasonPaused
// while the paused step keeps the provider's finish reason.
func TestClient_StreamText_PausesOnDeferredApprovals(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		if call != 1 {
			t.Fatalf("unexpected provider call %d", call)
		}
		return &sdk.GenerateResult{
			FinishReason: sdk.FinishReasonToolCalls,
			ToolCalls: []sdk.ToolCall{
				{ToolCallID: "c1", ToolName: "tool_a", Input: nil},
				{ToolCallID: "c2", ToolName: "tool_b", Input: nil},
			},
		}, nil
	}}

	execute := func(ctx *sdk.ToolExecContext, input any) (any, error) {
		t.Error("no tool should execute while its approval is deferred")
		return nil, nil
	}

	sr, err := sdk.StreamText(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("go")}),
		sdk.WithTools([]sdk.Tool{
			{Name: "tool_a", Parameters: &jsonschema.Schema{Type: "object"}, RequireApproval: true, Execute: execute},
			{Name: "tool_b", Parameters: &jsonschema.Schema{Type: "object"}, RequireApproval: true, Execute: execute},
		}),
		sdk.WithMaxSteps(5),
		sdk.WithApprovalHandler(func(_ context.Context, tc sdk.ToolCall) (sdk.ToolApprovalResult, error) {
			return sdk.ToolApprovalResult{
				Decision:   sdk.ToolApprovalDecisionDeferred,
				ApprovalID: "approval-" + tc.ToolName,
			}, nil
		}),
	)
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	var approvalIDs []string
	var finish *sdk.FinishPart
	for part := range sr.Stream {
		switch p := part.(type) {
		case *sdk.ToolApprovalRequestPart:
			approvalIDs = append(approvalIDs, p.ApprovalID)
		case *sdk.StreamToolResultPart:
			t.Fatal("did not expect tool result while approvals are deferred")
		case *sdk.ErrorPart:
			t.Fatalf("stream error: %v", p.Error)
		case *sdk.FinishPart:
			finish = p
		}
	}

	if len(approvalIDs) != 2 || approvalIDs[0] != "approval-tool_a" || approvalIDs[1] != "approval-tool_b" {
		t.Fatalf("approval request parts: %v, want [approval-tool_a approval-tool_b]", approvalIDs)
	}
	if finish == nil || finish.FinishReason != sdk.FinishReasonPaused {
		t.Fatalf("FinishPart: %#v, want paused", finish)
	}
	if sr.Pause == nil || len(sr.Pause.Pending) != 2 || sr.Pause.Pending[1].ToolCall.ToolCallID != "c2" {
		t.Fatalf("pause: %#v", sr.Pause)
	}
	if len(sr.Steps) != 1 || sr.Steps[0].FinishReason != sdk.FinishReasonToolCalls {
		t.Fatalf("paused step: %#v", sr.Steps)
	}
	if mp.calls != 1 {
		t.Fatalf("expected one provider call, got %d", mp.calls)
	}
}

// A handler error after an earlier deferral must not produce SDK approval
// events, execute tools, or commit a step. Per-call handlers can still leave
// their own earlier writes behind; atomic host persistence requires the batch
// handler and a transaction.
func TestClient_StreamText_ApprovalHandlerErrorAfterDeferral(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		return &sdk.GenerateResult{
			FinishReason: sdk.FinishReasonToolCalls,
			ToolCalls: []sdk.ToolCall{
				{ToolCallID: "c1", ToolName: "tool_a", Input: nil},
				{ToolCallID: "c2", ToolName: "tool_b", Input: nil},
			},
		}, nil
	}}

	executed := false
	hostWrites := 0
	execute := func(ctx *sdk.ToolExecContext, input any) (any, error) {
		executed = true
		return "done", nil
	}

	sr, err := sdk.StreamText(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("go")}),
		sdk.WithTools([]sdk.Tool{
			{Name: "tool_a", Parameters: &jsonschema.Schema{Type: "object"}, RequireApproval: true, Execute: execute},
			{Name: "tool_b", Parameters: &jsonschema.Schema{Type: "object"}, RequireApproval: true, Execute: execute},
		}),
		sdk.WithMaxSteps(5),
		sdk.WithApprovalHandler(func(_ context.Context, tc sdk.ToolCall) (sdk.ToolApprovalResult, error) {
			if tc.ToolName == "tool_a" {
				hostWrites++ // Simulates a per-call pending row the SDK cannot roll back.
				return sdk.ToolApprovalResult{Decision: sdk.ToolApprovalDecisionDeferred, ApprovalID: "approval-a"}, nil
			}
			return sdk.ToolApprovalResult{}, errors.New("approval store unavailable")
		}),
	)
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	var gotError bool
	for part := range sr.Stream {
		switch p := part.(type) {
		case *sdk.ToolApprovalRequestPart:
			t.Errorf("approval request for %q leaked despite the batch failing", p.ToolName)
		case *sdk.ErrorPart:
			gotError = true
		}
	}
	if !gotError {
		t.Error("expected ErrorPart for the handler failure")
	}
	if executed {
		t.Error("no tool should execute when the approval phase fails")
	}
	if len(sr.Steps) != 0 {
		t.Errorf("no step should be committed, got %d", len(sr.Steps))
	}
	if sr.Pause != nil {
		t.Errorf("no pause should be reported: %#v", sr.Pause)
	}
	if hostWrites != 1 {
		t.Fatalf("host writes: %d, want 1 (per-call callbacks are not transactional)", hostWrites)
	}
}

func TestClient_StreamText_OnStepCallback(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		if call == 1 {
			return &sdk.GenerateResult{
				FinishReason: sdk.FinishReasonToolCalls,
				ToolCalls: []sdk.ToolCall{{
					ToolCallID: "c1", ToolName: "noop", Input: nil,
				}},
			}, nil
		}
		return &sdk.GenerateResult{
			Text:         "done",
			FinishReason: sdk.FinishReasonStop,
		}, nil
	}}

	var mu sync.Mutex
	var steps []*sdk.StepResult

	sr, err := sdk.StreamText(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("go")}),
		sdk.WithTools([]sdk.Tool{{
			Name:       "noop",
			Parameters: &jsonschema.Schema{Type: "object"},
			Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
				return "ok", nil
			},
		}}),
		sdk.WithMaxSteps(5),
		sdk.WithOnStep(func(sr *sdk.StepResult) *sdk.GenerateParams {
			mu.Lock()
			steps = append(steps, sr)
			mu.Unlock()
			return nil
		}),
	)
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	// Consume the stream
	for range sr.Stream {
	}

	mu.Lock()
	defer mu.Unlock()
	if len(steps) != 2 {
		t.Fatalf("expected 2 step callbacks, got %d", len(steps))
	}
}

func TestClient_OnStepCommittedErrorStopsLoop(t *testing.T) {
	runners := []struct {
		name string
		run  func(...sdk.GenerateOption) error
	}{
		{"generate", func(opts ...sdk.GenerateOption) error {
			_, err := sdk.GenerateTextResult(context.Background(), opts...)
			return err
		}},
		{"stream", func(opts ...sdk.GenerateOption) error {
			sr, err := sdk.StreamText(context.Background(), opts...)
			if err != nil {
				return err
			}
			_, err = sr.ToResult()
			return err
		}},
	}

	for _, runner := range runners {
		t.Run(runner.name, func(t *testing.T) {
			commitErr := errors.New("checkpoint unavailable")
			mp := &mockProvider{handler: func(int, sdk.GenerateParams) (*sdk.GenerateResult, error) {
				return &sdk.GenerateResult{
					FinishReason: sdk.FinishReasonToolCalls,
					ToolCalls:    []sdk.ToolCall{{ToolCallID: "c1", ToolName: "noop"}},
				}, nil
			}}
			err := runner.run(
				sdk.WithModel(mockModel(mp)),
				sdk.WithTools([]sdk.Tool{{
					Name: "noop", Parameters: &jsonschema.Schema{Type: "object"},
					Execute: func(*sdk.ToolExecContext, any) (any, error) { return "ok", nil },
				}}),
				sdk.WithMaxSteps(5),
				sdk.WithOnStepCommitted(func(_ context.Context, i int, step *sdk.StepResult) error {
					if i != 0 || len(step.ToolResults) != 1 {
						return fmt.Errorf("unexpected committed step %d: %#v", i, step.ToolResults)
					}
					return commitErr
				}),
			)
			if !errors.Is(err, commitErr) {
				t.Fatalf("error: got %v, want %v", err, commitErr)
			}
			if mp.calls != 1 {
				t.Fatalf("provider calls: got %d, want 1", mp.calls)
			}
		})
	}
}

func TestClient_StreamText_DoesNotCommitIncompleteStep(t *testing.T) {
	mp := &mockProvider{streamHandler: func(int, sdk.GenerateParams) (*sdk.StreamResult, error) {
		ch := make(chan sdk.StreamPart, 1)
		ch <- &sdk.TextDeltaPart{ID: "partial", Text: "partial"}
		close(ch)
		return &sdk.StreamResult{Stream: ch}, nil
	}}
	committed := false

	sr, err := sdk.StreamText(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMaxSteps(5),
		sdk.WithOnStepCommitted(func(context.Context, int, *sdk.StepResult) error {
			committed = true
			return nil
		}),
	)
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}
	_, err = sr.ToResult()
	if err == nil || err.Error() != "twilightai: stream step 0 ended before finish-step" {
		t.Fatalf("stream error: got %v", err)
	}
	if committed {
		t.Fatal("incomplete step was committed")
	}
}

// ---------- unit tests: Steps, Messages fields ----------

func TestClient_GenerateTextResult_StepsAndMessages(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		if call == 1 {
			return &sdk.GenerateResult{
				Text:         "Let me add that.",
				FinishReason: sdk.FinishReasonToolCalls,
				ToolCalls: []sdk.ToolCall{{
					ToolCallID: "c1", ToolName: "add",
					Input: map[string]any{"a": float64(1), "b": float64(2)},
				}},
				Usage: sdk.Usage{TotalTokens: 10},
			}, nil
		}
		return &sdk.GenerateResult{
			Text:         "The answer is 3.",
			FinishReason: sdk.FinishReasonStop,
			Usage:        sdk.Usage{TotalTokens: 10},
		}, nil
	}}

	result, err := sdk.GenerateTextResult(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("1+2?")}),
		sdk.WithTools([]sdk.Tool{{
			Name:       "add",
			Parameters: &jsonschema.Schema{Type: "object"},
			Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
				return float64(3), nil
			},
		}}),
		sdk.WithMaxSteps(5),
	)
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	// Steps
	if len(result.Steps) != 2 {
		t.Fatalf("expected 2 steps, got %d", len(result.Steps))
	}

	// Step 0: tool call step → assistant msg (text + tool call) + tool msg (result)
	s0 := result.Steps[0]
	if s0.FinishReason != sdk.FinishReasonToolCalls {
		t.Errorf("step 0 finish: got %s", s0.FinishReason)
	}
	if len(s0.Messages) != 2 {
		t.Fatalf("step 0 messages: expected 2, got %d", len(s0.Messages))
	}
	if s0.Messages[0].Role != sdk.MessageRoleAssistant {
		t.Errorf("step 0 msg[0] role: got %s", s0.Messages[0].Role)
	}
	if s0.Messages[1].Role != sdk.MessageRoleTool {
		t.Errorf("step 0 msg[1] role: got %s", s0.Messages[1].Role)
	}

	// Step 1: final text step → assistant msg only
	s1 := result.Steps[1]
	if s1.FinishReason != sdk.FinishReasonStop {
		t.Errorf("step 1 finish: got %s", s1.FinishReason)
	}
	if len(s1.Messages) != 1 {
		t.Fatalf("step 1 messages: expected 1, got %d", len(s1.Messages))
	}
	if s1.Messages[0].Role != sdk.MessageRoleAssistant {
		t.Errorf("step 1 msg[0] role: got %s", s1.Messages[0].Role)
	}

	// All output messages = step0 msgs + step1 msgs
	if len(result.Messages) != 3 {
		t.Fatalf("total messages: expected 3, got %d", len(result.Messages))
	}
}

func TestClient_StreamText_StepsAndMessages(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		if call == 1 {
			return &sdk.GenerateResult{
				FinishReason: sdk.FinishReasonToolCalls,
				ToolCalls: []sdk.ToolCall{{
					ToolCallID: "c1", ToolName: "ping", Input: nil,
				}},
			}, nil
		}
		return &sdk.GenerateResult{
			Text:         "pong",
			FinishReason: sdk.FinishReasonStop,
		}, nil
	}}

	sr, err := sdk.StreamText(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("ping")}),
		sdk.WithTools([]sdk.Tool{{
			Name:       "ping",
			Parameters: &jsonschema.Schema{Type: "object"},
			Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
				return "pong", nil
			},
		}}),
		sdk.WithMaxSteps(5),
	)
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	for range sr.Stream {
	}

	if len(sr.Steps) != 2 {
		t.Fatalf("expected 2 steps, got %d", len(sr.Steps))
	}
	if sr.Steps[0].FinishReason != sdk.FinishReasonToolCalls {
		t.Errorf("step 0 finish: got %s", sr.Steps[0].FinishReason)
	}
	if len(sr.Steps[0].Messages) != 2 {
		t.Errorf("step 0 messages: expected 2, got %d", len(sr.Steps[0].Messages))
	}
	if sr.Steps[1].FinishReason != sdk.FinishReasonStop {
		t.Errorf("step 1 finish: got %s", sr.Steps[1].FinishReason)
	}
	if len(sr.Steps[1].Messages) != 1 {
		t.Errorf("step 1 messages: expected 1, got %d", len(sr.Steps[1].Messages))
	}
	if len(sr.Messages) != 3 {
		t.Fatalf("total messages: expected 3, got %d", len(sr.Messages))
	}
}

// ---------- unit tests: callback return override ----------

func TestClient_GenerateTextResult_OnStepOverride(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		if call == 1 {
			return &sdk.GenerateResult{
				FinishReason: sdk.FinishReasonToolCalls,
				ToolCalls: []sdk.ToolCall{{
					ToolCallID: "c1", ToolName: "x", Input: nil,
				}},
			}, nil
		}
		if params.System != "overridden-by-onstep" {
			t.Errorf("onStep override not applied: system=%q", params.System)
		}
		return &sdk.GenerateResult{
			Text:         "ok",
			FinishReason: sdk.FinishReasonStop,
		}, nil
	}}

	_, err := sdk.GenerateTextResult(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("go")}),
		sdk.WithTools([]sdk.Tool{{
			Name:       "x",
			Parameters: &jsonschema.Schema{Type: "object"},
			Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
				return "ok", nil
			},
		}}),
		sdk.WithMaxSteps(5),
		sdk.WithOnStep(func(sr *sdk.StepResult) *sdk.GenerateParams {
			if sr.FinishReason == sdk.FinishReasonToolCalls {
				return &sdk.GenerateParams{
					Model:  mockModel(mp),
					System: "overridden-by-onstep",
				}
			}
			return nil
		}),
	)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
}

func TestClient_GenerateTextResult_PrepareStepOverride(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		if call == 1 {
			return &sdk.GenerateResult{
				FinishReason: sdk.FinishReasonToolCalls,
				ToolCalls: []sdk.ToolCall{{
					ToolCallID: "c1", ToolName: "x", Input: nil,
				}},
			}, nil
		}
		if params.System != "replaced-by-preparestep" {
			t.Errorf("prepareStep override not applied: system=%q", params.System)
		}
		return &sdk.GenerateResult{
			Text:         "ok",
			FinishReason: sdk.FinishReasonStop,
		}, nil
	}}

	_, err := sdk.GenerateTextResult(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("go")}),
		sdk.WithTools([]sdk.Tool{{
			Name:       "x",
			Parameters: &jsonschema.Schema{Type: "object"},
			Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
				return "ok", nil
			},
		}}),
		sdk.WithMaxSteps(5),
		sdk.WithPrepareStep(func(p *sdk.GenerateParams) *sdk.GenerateParams {
			newParams := *p
			newParams.System = "replaced-by-preparestep"
			return &newParams
		}),
	)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
}

// Regression: an observe-only PrepareStep must not duplicate the accumulated
// conversation in the pause.
func TestClient_GenerateTextResult_PauseSurvivesPrepareStep(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		switch call {
		case 1:
			return &sdk.GenerateResult{
				FinishReason: sdk.FinishReasonToolCalls,
				ToolCalls:    []sdk.ToolCall{{ToolCallID: "c1", ToolName: "lookup", Input: nil}},
			}, nil
		case 2:
			return &sdk.GenerateResult{
				FinishReason: sdk.FinishReasonToolCalls,
				ToolCalls:    []sdk.ToolCall{{ToolCallID: "c2", ToolName: "deploy", Input: nil}},
			}, nil
		}
		t.Fatalf("unexpected provider call %d", call)
		return nil, nil
	}}

	result, err := sdk.GenerateTextResult(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithSystem("release bot rules"),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("release v2")}),
		sdk.WithTools([]sdk.Tool{
			{Name: "lookup", Parameters: &jsonschema.Schema{Type: "object"},
				Execute: func(ctx *sdk.ToolExecContext, in any) (any, error) { return "ok", nil }},
			{Name: "deploy", Parameters: &jsonschema.Schema{Type: "object"}, RequireApproval: true,
				Execute: func(ctx *sdk.ToolExecContext, in any) (any, error) { return "ok", nil }},
		}),
		sdk.WithMaxSteps(5),
		// Observe-only PrepareStep: must not corrupt the pause.
		sdk.WithPrepareStep(func(p *sdk.GenerateParams) *sdk.GenerateParams { return nil }),
		sdk.WithApprovalHandler(func(_ context.Context, tc sdk.ToolCall) (sdk.ToolApprovalResult, error) {
			return sdk.ToolApprovalResult{Decision: sdk.ToolApprovalDecisionDeferred, ApprovalID: "a-" + tc.ToolCallID}, nil
		}),
	)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if result.Pause == nil {
		t.Fatal("expected a pause")
	}

	// Exactly: user, assistant(c1), tool(c1), assistant(c2) — no duplicates.
	wantRoles := []sdk.MessageRole{
		sdk.MessageRoleUser, sdk.MessageRoleAssistant, sdk.MessageRoleTool, sdk.MessageRoleAssistant,
	}
	if len(result.Pause.Messages) != len(wantRoles) {
		t.Fatalf("pause messages: got %d, want %d: %#v", len(result.Pause.Messages), len(wantRoles), result.Pause.Messages)
	}
	for i, want := range wantRoles {
		if result.Pause.Messages[i].Role != want {
			t.Fatalf("pause message %d role: got %q, want %q", i, result.Pause.Messages[i].Role, want)
		}
	}
	// Tool call c1 appears exactly once across assistant messages.
	c1Count := 0
	for _, m := range result.Pause.Messages {
		if m.Role != sdk.MessageRoleAssistant {
			continue
		}
		for _, p := range m.Content {
			if tcp, ok := p.(sdk.ToolCallPart); ok && tcp.ToolCallID == "c1" {
				c1Count++
			}
		}
	}
	if c1Count != 1 {
		t.Fatalf("tool call c1 appears %d times in the pause, want 1", c1Count)
	}
	if result.Pause.System != "release bot rules" {
		t.Fatalf("pause system: %q", result.Pause.System)
	}
}

// A pause captures the effective context of the model call that deferred,
// including PrepareStep history compaction and system overrides. Rebuilding it
// from the run's original input would silently restore discarded history.
func TestClient_GenerateTextResult_PauseUsesPreparedContext(t *testing.T) {
	tests := []struct {
		name       string
		prepare    func(*sdk.GenerateParams) *sdk.GenerateParams
		failCommit bool
	}{
		{
			name: "mutate params and return nil",
			prepare: func(p *sdk.GenerateParams) *sdk.GenerateParams {
				p.Messages = []sdk.Message{sdk.UserMessage("compressed history")}
				p.System = "prepared rules"
				return nil
			},
		},
		{
			name: "return replacement params",
			prepare: func(p *sdk.GenerateParams) *sdk.GenerateParams {
				next := *p
				next.Messages = []sdk.Message{sdk.UserMessage("compressed history")}
				next.System = "prepared rules"
				return &next
			},
		},
		{
			name: "commit barrier failure",
			prepare: func(p *sdk.GenerateParams) *sdk.GenerateParams {
				next := *p
				next.Messages = []sdk.Message{sdk.UserMessage("compressed history")}
				next.System = "prepared rules"
				return &next
			},
			failCommit: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			options := preparedPauseOptions(tc.prepare)
			if tc.failCommit {
				options = append(options, sdk.WithOnStepCommitted(func(_ context.Context, _ int, step *sdk.StepResult) error {
					if len(step.DeferredToolApprovals) > 0 {
						call := step.Messages[0].Content[1].(sdk.ToolCallPart)
						call.ToolCallID = "corrupted-message"
						call.Input.(map[string]any)["target"] = "corrupted"
						step.Messages[0].Content[1] = call
						toolResult := step.Messages[1].Content[0].(sdk.ToolResultPart)
						toolResult.Result.(map[string]any)["status"] = "corrupted"
						step.Messages[1].Content[0] = toolResult
						step.DeferredToolApprovals[0].ToolCall.ToolCallID = "corrupted-pending"
						step.DeferredToolApprovals[0].ToolCall.Input.(map[string]any)["target"] = "corrupted"
						step.DeferredToolApprovals[0].Approval.Metadata["queue"].(map[string]any)["name"] = "corrupted"
						return errors.New("pause persistence failed")
					}
					return nil
				}))
			}

			result, err := sdk.GenerateTextResult(context.Background(), options...)
			if tc.failCommit {
				if err == nil || !strings.Contains(err.Error(), "pause persistence failed") {
					t.Fatalf("error = %v, want commit barrier failure", err)
				}
			} else if err != nil {
				t.Fatalf("GenerateTextResult: %v", err)
			}
			assertPreparedPause(t, result.Pause)
		})
	}
}

func TestClient_StreamText_PauseUsesPreparedContext(t *testing.T) {
	for _, failCommit := range []bool{false, true} {
		name := "success"
		if failCommit {
			name = "commit barrier failure"
		}
		t.Run(name, func(t *testing.T) {
			options := preparedPauseOptions(replaceWithPreparedContext)
			if failCommit {
				options = append(options, sdk.WithOnStepCommitted(func(_ context.Context, _ int, step *sdk.StepResult) error {
					if len(step.DeferredToolApprovals) > 0 {
						return errors.New("pause persistence failed")
					}
					return nil
				}))
			}
			sr, err := sdk.StreamText(context.Background(), options...)
			if err != nil {
				t.Fatalf("StreamText: %v", err)
			}

			sawPausedFinish := false
			sawCommitError := false
			for part := range sr.Stream {
				switch p := part.(type) {
				case *sdk.ErrorPart:
					if !failCommit || !strings.Contains(p.Error.Error(), "pause persistence failed") {
						t.Fatalf("stream error: %v", p.Error)
					}
					sawCommitError = true
					assertPreparedPause(t, sr.Pause)
				case *sdk.FinishPart:
					if p.FinishReason == sdk.FinishReasonPaused {
						sawPausedFinish = true
						assertPreparedPause(t, sr.Pause)
					}
				}
			}
			if failCommit {
				if !sawCommitError || sawPausedFinish {
					t.Fatalf("commit error=%v paused finish=%v", sawCommitError, sawPausedFinish)
				}
			} else if !sawPausedFinish || sawCommitError {
				t.Fatalf("paused finish=%v commit error=%v", sawPausedFinish, sawCommitError)
			}
		})
	}
}

func replaceWithPreparedContext(p *sdk.GenerateParams) *sdk.GenerateParams {
	next := *p
	next.Messages = []sdk.Message{sdk.UserMessage("compressed history")}
	next.System = "prepared rules"
	return &next
}

func preparedPauseOptions(prepare func(*sdk.GenerateParams) *sdk.GenerateParams) []sdk.GenerateOption {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		switch call {
		case 1:
			return &sdk.GenerateResult{
				FinishReason: sdk.FinishReasonToolCalls,
				ToolCalls:    []sdk.ToolCall{{ToolCallID: "c1", ToolName: "lookup", Input: nil}},
			}, nil
		case 2:
			if params.System != "prepared rules" {
				return nil, fmt.Errorf("provider system: %q, want prepared rules", params.System)
			}
			if err := validateSingleTextMessage(params.Messages, "compressed history"); err != nil {
				return nil, err
			}
			return &sdk.GenerateResult{
				FinishReason: sdk.FinishReasonToolCalls,
				ToolCalls: []sdk.ToolCall{
					{ToolCallID: "c2", ToolName: "audit", Input: map[string]any{"target": "release"}},
					{ToolCallID: "c3", ToolName: "deploy", Input: map[string]any{"target": "production"}},
				},
			}, nil
		default:
			return nil, fmt.Errorf("unexpected provider call %d", call)
		}
	}}

	return []sdk.GenerateOption{
		sdk.WithModel(mockModel(mp)),
		sdk.WithSystem("original rules"),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("uncompressed history")}),
		sdk.WithTools([]sdk.Tool{
			{Name: "lookup", Parameters: &jsonschema.Schema{Type: "object"},
				Execute: func(ctx *sdk.ToolExecContext, in any) (any, error) { return "lookup-ok", nil }},
			{Name: "audit", Parameters: &jsonschema.Schema{Type: "object"},
				Execute: func(ctx *sdk.ToolExecContext, in any) (any, error) { return map[string]any{"status": "ok"}, nil }},
			{Name: "deploy", Parameters: &jsonschema.Schema{Type: "object"}, RequireApproval: true,
				Execute: func(ctx *sdk.ToolExecContext, in any) (any, error) { return "deploy-ok", nil }},
		}),
		sdk.WithMaxSteps(5),
		sdk.WithPrepareStep(prepare),
		sdk.WithApprovalHandler(func(_ context.Context, call sdk.ToolCall) (sdk.ToolApprovalResult, error) {
			// Approval callbacks retain their historical input ownership. A host
			// mutation must not rewrite the provider call stored in the pause.
			call.Input.(map[string]any)["target"] = "handler-mutated"
			return sdk.ToolApprovalResult{
				Decision:   sdk.ToolApprovalDecisionDeferred,
				ApprovalID: "approval-" + call.ToolCallID,
				Metadata:   map[string]any{"queue": map[string]any{"name": "deployments"}},
			}, nil
		}),
	}
}

func assertSingleTextMessage(t *testing.T, messages []sdk.Message, want string) {
	t.Helper()
	if err := validateSingleTextMessage(messages, want); err != nil {
		t.Fatal(err)
	}
}

func validateSingleTextMessage(messages []sdk.Message, want string) error {
	if len(messages) != 1 || messages[0].Role != sdk.MessageRoleUser || len(messages[0].Content) != 1 {
		return fmt.Errorf("messages: %#v, want one user text message", messages)
	}
	text, ok := messages[0].Content[0].(sdk.TextPart)
	if !ok || text.Text != want {
		return fmt.Errorf("message content: %#v, want %q", messages[0].Content, want)
	}
	return nil
}

func assertPreparedPause(t *testing.T, pause *sdk.ToolApprovalPause) {
	t.Helper()
	if pause == nil {
		t.Fatal("expected a pause")
	}
	if pause.System != "prepared rules" {
		t.Fatalf("pause system: %q, want prepared rules", pause.System)
	}
	if len(pause.Messages) != 3 {
		t.Fatalf("pause messages: got %d, want 3: %#v", len(pause.Messages), pause.Messages)
	}
	assertSingleTextMessage(t, pause.Messages[:1], "compressed history")
	if pause.Messages[1].Role != sdk.MessageRoleAssistant || len(pause.Messages[1].Content) != 2 {
		t.Fatalf("paused assistant message: %#v", pause.Messages[1])
	}
	for i, wantID := range []string{"c2", "c3"} {
		call, ok := pause.Messages[1].Content[i].(sdk.ToolCallPart)
		if !ok || call.ToolCallID != wantID {
			t.Fatalf("paused tool call %d: %#v, want %s", i, pause.Messages[1].Content[i], wantID)
		}
		if got := call.Input.(map[string]any)["target"]; got != []string{"release", "production"}[i] {
			t.Fatalf("paused tool call %d target: %v", i, got)
		}
	}
	if pause.Messages[2].Role != sdk.MessageRoleTool || len(pause.Messages[2].Content) != 1 {
		t.Fatalf("resolved sibling message: %#v", pause.Messages[2])
	}
	result, ok := pause.Messages[2].Content[0].(sdk.ToolResultPart)
	if !ok || result.ToolCallID != "c2" || result.Result.(map[string]any)["status"] != "ok" {
		t.Fatalf("resolved sibling result: %#v", pause.Messages[2].Content[0])
	}
	if len(pause.Pending) != 1 || pause.Pending[0].ToolCall.ToolCallID != "c3" {
		t.Fatalf("pending approvals: %#v", pause.Pending)
	}
	if got := pause.Pending[0].ToolCall.Input.(map[string]any)["target"]; got != "production" {
		t.Fatalf("pending input target: %v", got)
	}
	queue := pause.Pending[0].Approval.Metadata["queue"].(map[string]any)
	if queue["name"] != "deployments" {
		t.Fatalf("pending metadata: %#v", pause.Pending[0].Approval.Metadata)
	}
}

// Regression: sr.Pause must be readable at the moment FinishPart(paused)
// arrives — the documented in-band pause signal.
func TestClient_StreamText_PauseVisibleAtFinishPart(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		return &sdk.GenerateResult{
			FinishReason: sdk.FinishReasonToolCalls,
			ToolCalls:    []sdk.ToolCall{{ToolCallID: "c1", ToolName: "gated", Input: nil}},
		}, nil
	}}

	sr, err := sdk.StreamText(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("go")}),
		sdk.WithTools([]sdk.Tool{{
			Name: "gated", Parameters: &jsonschema.Schema{Type: "object"}, RequireApproval: true,
			Execute: func(ctx *sdk.ToolExecContext, in any) (any, error) { return "x", nil },
		}}),
		sdk.WithMaxSteps(5),
		sdk.WithApprovalHandler(func(_ context.Context, tc sdk.ToolCall) (sdk.ToolApprovalResult, error) {
			return sdk.ToolApprovalResult{Decision: sdk.ToolApprovalDecisionDeferred, ApprovalID: "a1"}, nil
		}),
	)
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	var pauseAtFinish *sdk.ToolApprovalPause
	for part := range sr.Stream {
		if p, ok := part.(*sdk.FinishPart); ok && p.FinishReason == sdk.FinishReasonPaused {
			pauseAtFinish = sr.Pause
		}
	}
	if pauseAtFinish == nil || len(pauseAtFinish.Pending) != 1 {
		t.Fatalf("sr.Pause must be populated when FinishPart(paused) is received: %#v", pauseAtFinish)
	}
}

// Regression: a deferral in a batch with duplicate or empty tool-call IDs
// would produce an unaddressable pause — it must fail at pause time instead.
func TestClient_GenerateTextResult_DeferredWithBadIDsFails(t *testing.T) {
	run := func(t *testing.T, calls []sdk.ToolCall, wantErr string) {
		mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
			return &sdk.GenerateResult{FinishReason: sdk.FinishReasonToolCalls, ToolCalls: calls}, nil
		}}
		executed := false
		_, err := sdk.GenerateTextResult(context.Background(),
			sdk.WithModel(mockModel(mp)),
			sdk.WithMessages([]sdk.Message{sdk.UserMessage("go")}),
			sdk.WithTools([]sdk.Tool{
				{Name: "safe", Parameters: &jsonschema.Schema{Type: "object"},
					Execute: func(ctx *sdk.ToolExecContext, in any) (any, error) { executed = true; return "ok", nil }},
				{Name: "gated", Parameters: &jsonschema.Schema{Type: "object"}, RequireApproval: true,
					Execute: func(ctx *sdk.ToolExecContext, in any) (any, error) { executed = true; return "ok", nil }},
			}),
			sdk.WithMaxSteps(5),
			sdk.WithApprovalHandler(func(_ context.Context, tc sdk.ToolCall) (sdk.ToolApprovalResult, error) {
				return sdk.ToolApprovalResult{Decision: sdk.ToolApprovalDecisionDeferred, ApprovalID: "a-" + tc.ToolCallID}, nil
			}),
		)
		if err == nil || !strings.Contains(err.Error(), wantErr) {
			t.Fatalf("error = %v, want containing %q", err, wantErr)
		}
		if executed {
			t.Fatal("no tool may execute when the batch fails ID validation")
		}
	}

	t.Run("empty deferred ID", func(t *testing.T) {
		run(t, []sdk.ToolCall{{ToolCallID: "", ToolName: "gated", Input: nil}}, "has no tool call ID")
	})
	t.Run("duplicate IDs with deferral", func(t *testing.T) {
		run(t, []sdk.ToolCall{
			{ToolCallID: "c1", ToolName: "safe", Input: nil},
			{ToolCallID: "c1", ToolName: "gated", Input: nil},
		}, "not unique")
	})
}

// A keyed batch request must be addressable before host code runs. Invalid
// provider IDs are known before the handler, so a transactional handler must
// never commit rows that the SDK will immediately reject.
func TestClient_GenerateTextResult_ApprovalBatchHandlerRejectsBadCallIDsBeforeInvocation(t *testing.T) {
	tests := []struct {
		name    string
		calls   []sdk.ToolCall
		wantErr string
	}{
		{
			name:    "empty gated ID",
			calls:   []sdk.ToolCall{{ToolCallID: "", ToolName: "gated_a", Input: nil}},
			wantErr: "has no tool call ID",
		},
		{
			name: "duplicate gated IDs",
			calls: []sdk.ToolCall{
				{ToolCallID: "c1", ToolName: "gated_a", Input: nil},
				{ToolCallID: "c1", ToolName: "gated_b", Input: nil},
			},
			wantErr: "not unique",
		},
		{
			name: "gated ID duplicates non-gated sibling",
			calls: []sdk.ToolCall{
				{ToolCallID: "c1", ToolName: "safe", Input: nil},
				{ToolCallID: "c1", ToolName: "gated_a", Input: nil},
			},
			wantErr: "not unique",
		},
		{
			name: "duplicate non-gated sibling IDs",
			calls: []sdk.ToolCall{
				{ToolCallID: "c1", ToolName: "safe", Input: nil},
				{ToolCallID: "c1", ToolName: "safe", Input: nil},
				{ToolCallID: "c2", ToolName: "gated_a", Input: nil},
			},
			wantErr: "not unique",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
				return &sdk.GenerateResult{FinishReason: sdk.FinishReasonToolCalls, ToolCalls: tc.calls}, nil
			}}
			handlerCalls := 0
			executed := false
			result, err := sdk.GenerateTextResult(context.Background(),
				sdk.WithModel(mockModel(mp)),
				sdk.WithMessages([]sdk.Message{sdk.UserMessage("go")}),
				sdk.WithTools([]sdk.Tool{
					{Name: "safe", Parameters: &jsonschema.Schema{Type: "object"},
						Execute: func(ctx *sdk.ToolExecContext, in any) (any, error) { executed = true; return "ok", nil }},
					{Name: "gated_a", Parameters: &jsonschema.Schema{Type: "object"}, RequireApproval: true,
						Execute: func(ctx *sdk.ToolExecContext, in any) (any, error) { executed = true; return "ok", nil }},
					{Name: "gated_b", Parameters: &jsonschema.Schema{Type: "object"}, RequireApproval: true,
						Execute: func(ctx *sdk.ToolExecContext, in any) (any, error) { executed = true; return "ok", nil }},
				}),
				sdk.WithMaxSteps(5),
				sdk.WithApprovalBatchHandler(func(_ context.Context, _ string, calls []sdk.ToolCall) ([]sdk.ToolApprovalBatchResult, error) {
					handlerCalls++
					answers := make([]sdk.ToolApprovalBatchResult, len(calls))
					for i, call := range calls {
						answers[i] = sdk.ToolApprovalBatchResult{
							ToolCallID: call.ToolCallID,
							Result:     sdk.ToolApprovalResult{Decision: sdk.ToolApprovalDecisionApproved},
						}
					}
					return answers, nil
				}),
			)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error = %v, want containing %q", err, tc.wantErr)
			}
			if result != nil {
				t.Fatalf("result = %#v, want nil", result)
			}
			if handlerCalls != 0 {
				t.Fatalf("batch handler calls: %d, want 0", handlerCalls)
			}
			if executed {
				t.Fatal("no tool may execute when batch request IDs are invalid")
			}
		})
	}
}

func TestClient_StreamText_ApprovalBatchHandlerRejectsBadCallIDsBeforeInvocation(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		return &sdk.GenerateResult{
			FinishReason: sdk.FinishReasonToolCalls,
			ToolCalls: []sdk.ToolCall{
				{ToolCallID: "c1", ToolName: "safe", Input: nil},
				{ToolCallID: "c1", ToolName: "gated", Input: nil},
			},
		}, nil
	}}
	handlerCalls := 0
	executed := false
	sr, err := sdk.StreamText(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("go")}),
		sdk.WithTools([]sdk.Tool{
			{Name: "safe", Parameters: &jsonschema.Schema{Type: "object"},
				Execute: func(ctx *sdk.ToolExecContext, in any) (any, error) { executed = true; return "ok", nil }},
			{Name: "gated", Parameters: &jsonschema.Schema{Type: "object"}, RequireApproval: true,
				Execute: func(ctx *sdk.ToolExecContext, in any) (any, error) { executed = true; return "ok", nil }},
		}),
		sdk.WithMaxSteps(5),
		sdk.WithApprovalBatchHandler(func(_ context.Context, _ string, calls []sdk.ToolCall) ([]sdk.ToolApprovalBatchResult, error) {
			handlerCalls++
			return nil, errors.New("must not be called")
		}),
	)
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	sawError := false
	for part := range sr.Stream {
		switch p := part.(type) {
		case *sdk.ToolApprovalRequestPart:
			t.Fatalf("approval event emitted for invalid call IDs: %#v", p)
		case *sdk.ErrorPart:
			sawError = true
			if !strings.Contains(p.Error.Error(), "not unique") {
				t.Fatalf("stream error = %v, want non-unique IDs", p.Error)
			}
		}
	}
	if !sawError {
		t.Fatal("expected ErrorPart")
	}
	if handlerCalls != 0 {
		t.Fatalf("batch handler calls: %d, want 0", handlerCalls)
	}
	if executed {
		t.Fatal("no tool may execute when batch request IDs are invalid")
	}
	if len(sr.Steps) != 0 || sr.Pause != nil {
		t.Fatalf("stream state after validation failure: steps=%d pause=%#v", len(sr.Steps), sr.Pause)
	}
}

// The batch approval handler answers a whole step's gated calls in one
// invocation: hosts create all pending records in one transaction, and the
// batch ID reappears on the pause for reconciliation.
func TestClient_GenerateTextResult_ApprovalBatchHandler(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		if call != 1 {
			t.Fatalf("unexpected provider call %d", call)
		}
		return &sdk.GenerateResult{
			FinishReason: sdk.FinishReasonToolCalls,
			ToolCalls: []sdk.ToolCall{
				{ToolCallID: "c1", ToolName: "safe", Input: nil},
				{ToolCallID: "c2", ToolName: "write", Input: nil},
				{ToolCallID: "c3", ToolName: "deploy", Input: nil},
			},
		}, nil
	}}

	executed := map[string]bool{}
	var mu sync.Mutex
	execute := func(name string) sdk.ToolExecuteFunc {
		return func(ctx *sdk.ToolExecContext, in any) (any, error) {
			mu.Lock()
			executed[name] = true
			mu.Unlock()
			return name + "-ok", nil
		}
	}

	var batchCalls int
	var seenBatchID string
	var seenNames []string
	result, err := sdk.GenerateTextResult(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("go")}),
		sdk.WithTools([]sdk.Tool{
			{Name: "safe", Parameters: &jsonschema.Schema{Type: "object"}, Execute: execute("safe")},
			{Name: "write", Parameters: &jsonschema.Schema{Type: "object"}, RequireApproval: true, Execute: execute("write")},
			{Name: "deploy", Parameters: &jsonschema.Schema{Type: "object"}, RequireApproval: true, Execute: execute("deploy")},
		}),
		sdk.WithMaxSteps(5),
		sdk.WithApprovalBatchHandler(func(_ context.Context, batchID string, calls []sdk.ToolCall) ([]sdk.ToolApprovalBatchResult, error) {
			batchCalls++
			seenBatchID = batchID
			for _, c := range calls {
				seenNames = append(seenNames, c.ToolName)
			}
			// One transaction's worth of answers, deliberately assembled out
			// of input order: association is by ToolCallID, not position.
			return []sdk.ToolApprovalBatchResult{
				{ToolCallID: "c3", Result: sdk.ToolApprovalResult{Decision: sdk.ToolApprovalDecisionDeferred, ApprovalID: "a-deploy"}},
				{ToolCallID: "c2", Result: sdk.ToolApprovalResult{Decision: sdk.ToolApprovalDecisionApproved}},
			}, nil
		}),
	)
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	if batchCalls != 1 {
		t.Fatalf("batch handler invocations: %d, want 1", batchCalls)
	}
	if len(seenNames) != 2 || seenNames[0] != "write" || seenNames[1] != "deploy" {
		t.Fatalf("batch handler saw %v, want [write deploy] (gated calls only)", seenNames)
	}
	if seenBatchID == "" {
		t.Fatal("batch handler must receive a batch ID")
	}
	if !executed["safe"] || !executed["write"] || executed["deploy"] {
		t.Fatalf("execution set: %v", executed)
	}
	if result.Pause == nil || result.Pause.BatchID != seenBatchID {
		t.Fatalf("pause must carry the same batch ID: pause=%#v seen=%q", result.Pause, seenBatchID)
	}
	if len(result.Pause.Pending) != 1 || result.Pause.Pending[0].ToolCall.ToolCallID != "c3" {
		t.Fatalf("pending: %#v", result.Pause.Pending)
	}
}

// A batch handler error fails the whole step before any emission or
// execution — the transactional rollback story requires zero SDK-side trace.
func TestClient_GenerateTextResult_ApprovalBatchHandlerError(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		return &sdk.GenerateResult{
			FinishReason: sdk.FinishReasonToolCalls,
			ToolCalls: []sdk.ToolCall{
				{ToolCallID: "c1", ToolName: "safe", Input: nil},
				{ToolCallID: "c2", ToolName: "gated", Input: nil},
			},
		}, nil
	}}
	executed := false
	_, err := sdk.GenerateTextResult(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("go")}),
		sdk.WithTools([]sdk.Tool{
			{Name: "safe", Parameters: &jsonschema.Schema{Type: "object"},
				Execute: func(ctx *sdk.ToolExecContext, in any) (any, error) { executed = true; return "ok", nil }},
			{Name: "gated", Parameters: &jsonschema.Schema{Type: "object"}, RequireApproval: true,
				Execute: func(ctx *sdk.ToolExecContext, in any) (any, error) { executed = true; return "ok", nil }},
		}),
		sdk.WithMaxSteps(5),
		sdk.WithApprovalBatchHandler(func(_ context.Context, _ string, calls []sdk.ToolCall) ([]sdk.ToolApprovalBatchResult, error) {
			return nil, errors.New("transaction rolled back")
		}),
	)
	if err == nil || !strings.Contains(err.Error(), "approval batch handler") {
		t.Fatalf("error = %v, want batch handler failure", err)
	}
	if executed {
		t.Fatal("no tool may execute when the batch handler fails")
	}
}

// The batch response is keyed by ToolCallID and verified complete: missing,
// duplicate, or unknown IDs fail loudly, and — unlike the per-call handler's
// historical convention — an empty Decision is an error here, never an
// implicit approval: batch responses come from data assembly where a missed
// assignment yields the zero value.
func TestClient_GenerateTextResult_ApprovalBatchHandlerValidation(t *testing.T) {
	cases := []struct {
		name    string
		answers []sdk.ToolApprovalBatchResult
		wantErr string
	}{
		{
			name:    "missing result",
			answers: []sdk.ToolApprovalBatchResult{},
			wantErr: "no result for tool call \"c1\"",
		},
		{
			name: "duplicate results",
			answers: []sdk.ToolApprovalBatchResult{
				{ToolCallID: "c1", Result: sdk.ToolApprovalResult{Decision: sdk.ToolApprovalDecisionApproved}},
				{ToolCallID: "c1", Result: sdk.ToolApprovalResult{Decision: sdk.ToolApprovalDecisionRejected}},
			},
			wantErr: "duplicate results for tool call \"c1\"",
		},
		{
			name: "unknown tool call",
			answers: []sdk.ToolApprovalBatchResult{
				{ToolCallID: "c1", Result: sdk.ToolApprovalResult{Decision: sdk.ToolApprovalDecisionApproved}},
				{ToolCallID: "ghost", Result: sdk.ToolApprovalResult{Decision: sdk.ToolApprovalDecisionApproved}},
			},
			wantErr: "unknown tool call \"ghost\"",
		},
		{
			name: "zero-value decision fails closed",
			answers: []sdk.ToolApprovalBatchResult{
				{ToolCallID: "c1"}, // Decision unset — must NOT approve
			},
			wantErr: "must be explicitly approved, rejected, or deferred",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
				return &sdk.GenerateResult{
					FinishReason: sdk.FinishReasonToolCalls,
					ToolCalls:    []sdk.ToolCall{{ToolCallID: "c1", ToolName: "gated", Input: nil}},
				}, nil
			}}
			executed := false
			_, err := sdk.GenerateTextResult(context.Background(),
				sdk.WithModel(mockModel(mp)),
				sdk.WithMessages([]sdk.Message{sdk.UserMessage("go")}),
				sdk.WithTools([]sdk.Tool{{Name: "gated", Parameters: &jsonschema.Schema{Type: "object"}, RequireApproval: true,
					Execute: func(ctx *sdk.ToolExecContext, in any) (any, error) { executed = true; return "ok", nil }}}),
				sdk.WithMaxSteps(5),
				sdk.WithApprovalBatchHandler(func(_ context.Context, _ string, calls []sdk.ToolCall) ([]sdk.ToolApprovalBatchResult, error) {
					return tc.answers, nil
				}),
			)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error = %v, want containing %q", err, tc.wantErr)
			}
			if executed {
				t.Fatal("no tool may execute when batch validation fails")
			}
		})
	}
}

// Configuring both handlers is ambiguous and rejected at config time.
func TestClient_ApprovalHandlersMutuallyExclusive(t *testing.T) {
	_, err := sdk.GenerateTextResult(context.Background(),
		sdk.WithModel(mockModel(&mockProvider{})),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("go")}),
		sdk.WithApprovalHandler(func(_ context.Context, tc sdk.ToolCall) (sdk.ToolApprovalResult, error) {
			return sdk.ToolApprovalResult{Decision: sdk.ToolApprovalDecisionApproved}, nil
		}),
		sdk.WithApprovalBatchHandler(func(_ context.Context, _ string, _ []sdk.ToolCall) ([]sdk.ToolApprovalBatchResult, error) {
			return nil, nil
		}),
	)
	if err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("error = %v, want mutual-exclusion error", err)
	}
}

// When the paused step's commit barrier fails, the pause must still reach
// the host: approval requests were announced and sibling side effects have
// happened — the pause is the only reconciliation handle.
func TestClient_GenerateTextResult_PauseSurvivesCommitBarrierFailure(t *testing.T) {
	mp := &mockProvider{handler: func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error) {
		return &sdk.GenerateResult{
			FinishReason: sdk.FinishReasonToolCalls,
			ToolCalls: []sdk.ToolCall{
				{ToolCallID: "c1", ToolName: "safe", Input: nil},
				{ToolCallID: "c2", ToolName: "gated", Input: nil},
			},
		}, nil
	}}
	sideEffect := false
	result, err := sdk.GenerateTextResult(context.Background(),
		sdk.WithModel(mockModel(mp)),
		sdk.WithMessages([]sdk.Message{sdk.UserMessage("go")}),
		sdk.WithTools([]sdk.Tool{
			{Name: "safe", Parameters: &jsonschema.Schema{Type: "object"},
				Execute: func(ctx *sdk.ToolExecContext, in any) (any, error) { sideEffect = true; return "done", nil }},
			{Name: "gated", Parameters: &jsonschema.Schema{Type: "object"}, RequireApproval: true,
				Execute: func(ctx *sdk.ToolExecContext, in any) (any, error) { return "x", nil }},
		}),
		sdk.WithMaxSteps(5),
		sdk.WithApprovalHandler(func(_ context.Context, tc sdk.ToolCall) (sdk.ToolApprovalResult, error) {
			return sdk.ToolApprovalResult{Decision: sdk.ToolApprovalDecisionDeferred, ApprovalID: "a-c2"}, nil
		}),
		sdk.WithOnStepCommitted(func(_ context.Context, _ int, step *sdk.StepResult) error {
			if len(step.DeferredToolApprovals) > 0 {
				return errors.New("db blip") // barrier fails exactly on the paused step
			}
			return nil
		}),
	)
	if err == nil || !strings.Contains(err.Error(), "db blip") {
		t.Fatalf("error = %v, want barrier failure", err)
	}
	if !sideEffect {
		t.Fatal("test setup: the sibling side effect should have run before the barrier")
	}
	if result == nil || result.Pause == nil {
		t.Fatal("the pause must survive a commit-barrier failure on the paused step")
	}
	if len(result.Pause.Pending) != 1 || result.Pause.Pending[0].ToolCall.ToolCallID != "c2" {
		t.Fatalf("pending: %#v", result.Pause.Pending)
	}
	// The pause's conversation must be complete even though the step never
	// committed: user input, the assistant message with both calls, and the
	// tool message carrying the resolved sibling's result.
	wantRoles := []sdk.MessageRole{sdk.MessageRoleUser, sdk.MessageRoleAssistant, sdk.MessageRoleTool}
	if len(result.Pause.Messages) != len(wantRoles) {
		t.Fatalf("pause messages: got %d, want %d: %#v", len(result.Pause.Messages), len(wantRoles), result.Pause.Messages)
	}
	for i, want := range wantRoles {
		if result.Pause.Messages[i].Role != want {
			t.Fatalf("pause message %d role: got %q, want %q", i, result.Pause.Messages[i].Role, want)
		}
	}
	trp, ok := result.Pause.Messages[2].Content[0].(sdk.ToolResultPart)
	if !ok || trp.ToolCallID != "c1" {
		t.Fatalf("resolved sibling's result must be in the pause: %#v", result.Pause.Messages[2].Content)
	}
	// No step was committed: the barrier rejected the only step.
	if len(result.Steps) != 0 {
		t.Fatalf("steps: %d, want 0 (barrier rejected the step)", len(result.Steps))
	}
}
