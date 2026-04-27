package sdk_test

import (
	"context"
	"fmt"
	"os"
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
	calls   int
	handler func(call int, params sdk.GenerateParams) (*sdk.GenerateResult, error)
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
	if result.DeferredToolApproval == nil || result.DeferredToolApproval.ApprovalID != "approval-1" {
		t.Fatalf("missing deferred approval: %#v", result.DeferredToolApproval)
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
	if sr.DeferredToolApproval == nil || sr.DeferredToolApproval.ApprovalID != "approval-1" {
		t.Fatalf("missing deferred approval: %#v", sr.DeferredToolApproval)
	}
	if len(sr.Messages) != 1 {
		t.Fatalf("expected assistant tool-call message, got %#v", sr.Messages)
	}
	if mp.calls != 1 {
		t.Fatalf("expected one provider call, got %d", mp.calls)
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
