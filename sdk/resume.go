package sdk

import (
	"context"
	"fmt"
	"sort"
	"strings"
)

// ToolDecisionType is a caller-submitted verdict for one pending tool call.
// Unlike ToolApprovalDecision (the ApprovalHandler's answer during a run,
// which may legitimately defer), a decision at resume time must be final:
// approved or rejected. The zero value is invalid and fails validation.
type ToolDecisionType string

const (
	ToolDecisionApproved ToolDecisionType = "approved"
	ToolDecisionRejected ToolDecisionType = "rejected"
)

// ToolDecision is the caller's verdict for one pending tool call, keyed by
// ToolCallID in the decisions map handed to ApplyToolDecisions / ResumeText.
type ToolDecision struct {
	Decision ToolDecisionType `json:"decision"`
	// Reason is included in the error tool result for rejected calls.
	Reason string `json:"reason,omitempty"`
}

// ToolApprovalResolution is the outcome of applying decisions to a pause:
// the executed (or rejection) results for every previously pending call, and
// the tool message that completes the paused conversation. After a
// resolution, pause.Messages + resolution.Messages is a protocol-complete
// conversation ready for an ordinary generation call.
type ToolApprovalResolution struct {
	// Results holds one entry per previously pending call, in the paused
	// assistant message's order. IsError distinguishes a failed execution or
	// a rejection from a successful result.
	Results []ToolResultPart `json:"results"`
	// Messages holds the tool message(s) completing the paused conversation.
	Messages []Message `json:"messages"`
}

// ApplyToolDecisions applies the caller's decisions to a paused run without
// making any model call: approved calls execute (in parallel, through the
// same path as a normal run), rejected calls produce an error tool result
// carrying the decision's Reason.
//
// Use this instead of ResumeText when tool side effects must not be
// repeatable: persist the completed conversation durably, then generate over
// it. If generation fails, retry generation alone — the approved tools do
// not run again.
//
//	resolution, err := sdk.ApplyToolDecisions(ctx, pause, decisions, tools)
//	completed := slices.Concat(pause.Messages, resolution.Messages)
//	// persist `completed` here
//	result, err := sdk.GenerateTextResult(ctx, sdk.WithMessages(completed), ...)
//
// Validation is strict and runs before any tool executes: every pending call
// needs a decision, decisions for unknown calls are rejected, a decision
// must be explicitly approved or rejected (the zero value is not approval),
// and every approved call's tool must be present and executable in tools.
// The pause itself is cross-checked: its Pending list must match the calls
// left unresolved by its Messages tail, so a hand-assembled pause that
// disagrees with its own conversation fails loudly instead of producing a
// protocol-invalid resume.
func ApplyToolDecisions(ctx context.Context, pause ToolApprovalPause, decisions map[string]ToolDecision, tools []Tool) (*ToolApprovalResolution, error) {
	return applyToolDecisions(ctx, pause, decisions, tools, nil)
}

// applyToolDecisions is ApplyToolDecisions with a progress sink: the
// streaming resume path threads its stream in so resumed executions emit the
// same tool parts a live run would.
func applyToolDecisions(ctx context.Context, pause ToolApprovalPause, decisions map[string]ToolDecision, tools []Tool, sendProgress func(StreamPart)) (*ToolApprovalResolution, error) {
	if err := validatePause(pause); err != nil {
		return nil, err
	}
	toolMap := buildToolMap(tools)
	if err := validateToolDecisions(pause.Pending, decisions, toolMap); err != nil {
		return nil, err
	}

	results := make([]ToolResultPart, len(pause.Pending))
	var toRun []pendingToolExec
	for i := range pause.Pending {
		tc := pause.Pending[i].ToolCall
		if decisions[tc.ToolCallID].Decision == ToolDecisionRejected {
			if sendProgress != nil {
				sendProgress(&ToolOutputDeniedPart{
					ToolCallID: tc.ToolCallID,
					ToolName:   tc.ToolName,
				})
			}
			results[i] = rejectedToolResult(tc, ToolApprovalResult{Reason: decisions[tc.ToolCallID].Reason})
			continue
		}
		// validateToolDecisions guarantees the tool exists and is executable
		// for every approved call.
		toRun = append(toRun, pendingToolExec{idx: i, tc: tc, tool: toolMap[tc.ToolName]})
	}
	runToolsParallel(ctx, toRun, results, sendProgress)

	// A canceled context makes ctx-honoring tools return their context error
	// as an ordinary tool failure. Persisting those as final outcomes would
	// record approved-but-never-ran calls as failures, so surface the
	// cancellation instead of a resolution.
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("twilightai: resume: context ended while applying decisions (side effects of completed tools have already happened): %w", err)
	}

	res := make([]ToolResultPart, len(results))
	copy(res, results)
	return &ToolApprovalResolution{
		Results:  res,
		Messages: []Message{ToolMessage(results...)},
	}, nil
}

// ResumeText resumes a run that paused on deferred tool approvals: it applies
// the decisions (executing approved calls), completes the conversation, and
// hands it to the ordinary generation path. Do not pass WithMessages — the
// conversation comes from the pause.
//
// Tool side effects happen before the model call; if the model call fails
// and the caller retries ResumeText with the same pause, approved tools run
// again. When that matters, use ApplyToolDecisions and persist the completed
// conversation before generating.
//
// The resolution is reported on Result.Resume. Steps and the step-indexed
// callbacks observe only real model steps, numbered exactly as in a fresh
// run. Keep the ApprovalHandler registered if the model may request further
// approval-gated calls: a resumed run can pause again, yielding a new Pause.
func ResumeText(ctx context.Context, pause ToolApprovalPause, decisions map[string]ToolDecision, options ...GenerateOption) (*GenerateResult, error) {
	return defaultClient.ResumeText(ctx, pause, decisions, options...)
}

// ResumeText is the method form of the package-level function.
func (c *Client) ResumeText(ctx context.Context, pause ToolApprovalPause, decisions map[string]ToolDecision, options ...GenerateOption) (*GenerateResult, error) {
	resolution, opts, err := c.prepareResume(ctx, pause, decisions, options, nil)
	if err != nil {
		return nil, err
	}
	result, err := c.GenerateTextResult(ctx, opts...)
	if err != nil {
		return nil, err
	}
	result.Resume = resolution
	return result, nil
}

// ResumeTextStream resumes a paused run in streaming mode. Validation and
// approved-tool execution happen synchronously before this function returns,
// so a validation error surfaces here — not as a mid-stream ErrorPart — and
// no stream is opened until the decisions are applied. The resolution is
// available on StreamResult.Resume immediately.
//
// The stream opens with the resume phase's tool parts (progress, results,
// denials for the calls the decisions covered — they happened before the
// first model call), followed by the provider's normal lifecycle starting at
// its StartPart.
func ResumeTextStream(ctx context.Context, pause ToolApprovalPause, decisions map[string]ToolDecision, options ...GenerateOption) (*StreamResult, error) {
	return defaultClient.ResumeTextStream(ctx, pause, decisions, options...)
}

// ResumeTextStream is the method form of the package-level function.
func (c *Client) ResumeTextStream(ctx context.Context, pause ToolApprovalPause, decisions map[string]ToolDecision, options ...GenerateOption) (*StreamResult, error) {
	// Buffer resume-phase tool parts and replay them as the stream's first
	// parts: resumed executions are visible to stream consumers exactly like
	// live executions, while validation still fails synchronously before any
	// stream exists.
	var resumeParts []StreamPart
	resolution, opts, err := c.prepareResume(ctx, pause, decisions, options, func(part StreamPart) {
		resumeParts = append(resumeParts, part)
	})
	if err != nil {
		return nil, err
	}
	sr, err := c.StreamText(ctx, opts...)
	if err != nil {
		return nil, err
	}
	sr.Resume = resolution
	if len(resumeParts) == 0 {
		return sr, nil
	}

	out := make(chan StreamPart, len(resumeParts)+16)
	inner := sr.Stream
	sr.Stream = out
	go func() {
		defer close(out)
		for _, p := range resumeParts {
			out <- p
		}
		for p := range inner {
			out <- p
		}
	}()
	return sr, nil
}

// prepareResume runs the shared resume front half: reject conflicting
// conversation sources, apply the decisions, and assemble the generation
// options over the completed conversation. The pause's System prompt is
// re-applied so the model keeps the original run's instructions.
func (c *Client) prepareResume(ctx context.Context, pause ToolApprovalPause, decisions map[string]ToolDecision, options []GenerateOption, sendProgress func(StreamPart)) (*ToolApprovalResolution, []GenerateOption, error) {
	cfg, _, err := buildConfig(options)
	if err != nil {
		return nil, nil, err
	}
	if len(cfg.Params.Messages) > 0 {
		return nil, nil, fmt.Errorf("twilightai: resume: WithMessages conflicts with the pause — the conversation comes from pause.Messages")
	}
	if cfg.Params.System != "" && cfg.Params.System != pause.System {
		return nil, nil, fmt.Errorf("twilightai: resume: WithSystem conflicts with the pause — the system prompt comes from pause.System")
	}

	resolution, err := applyToolDecisions(ctx, pause, decisions, cfg.Params.Tools, sendProgress)
	if err != nil {
		return nil, nil, err
	}

	completed := make([]Message, 0, len(pause.Messages)+len(resolution.Messages))
	completed = append(completed, pause.Messages...)
	completed = append(completed, resolution.Messages...)

	opts := make([]GenerateOption, 0, len(options)+3)
	opts = append(opts, options...)
	opts = append(opts, WithMessages(completed))
	if pause.System != "" {
		opts = append(opts, WithSystem(pause.System))
	}
	// The resolution must be visible to OnFinish observers, which fire inside
	// the generation call — before the entry point attaches Result.Resume.
	if userOnFinish := cfg.OnFinish; userOnFinish != nil {
		opts = append(opts, WithOnFinish(func(r *GenerateResult) {
			r.Resume = resolution
			userOnFinish(r)
		}))
	}
	return resolution, opts, nil
}

// validatePause cross-checks the pause against its own conversation: the
// Pending list must be exactly the calls the Messages tail leaves
// unresolved. The two describe one fact; a hand-assembled pause where they
// disagree would resume into a protocol-invalid conversation.
func validatePause(pause ToolApprovalPause) error {
	if len(pause.Pending) == 0 {
		return fmt.Errorf("twilightai: resume: pause has no pending tool calls")
	}
	derived := pendingToolCallsFromTail(pause.Messages)
	if len(derived) != len(pause.Pending) {
		return fmt.Errorf("twilightai: resume: pause.Pending lists %d calls but pause.Messages leaves %d unresolved", len(pause.Pending), len(derived))
	}
	for i := range derived {
		if derived[i].ToolCallID != pause.Pending[i].ToolCall.ToolCallID {
			return fmt.Errorf("twilightai: resume: pause.Pending[%d] is %q but pause.Messages leaves %q unresolved at that position",
				i, pause.Pending[i].ToolCall.ToolCallID, derived[i].ToolCallID)
		}
		// The ID alone does not prove the pause agrees with its conversation:
		// executing a call whose recorded name differs from the conversation's
		// would run a different tool than the model requested.
		if derived[i].ToolName != pause.Pending[i].ToolCall.ToolName {
			return fmt.Errorf("twilightai: resume: pause.Pending[%d] (%s) names tool %q but the conversation's call is %q",
				i, derived[i].ToolCallID, pause.Pending[i].ToolCall.ToolName, derived[i].ToolName)
		}
	}
	return nil
}

// validateToolDecisions enforces the decision contract before any side
// effect: every pending call has a decision, no decision targets an unknown
// call, every decision is an explicit approve or reject (the zero value
// fails closed), and every approved call's tool is present and executable.
func validateToolDecisions(pending []DeferredToolApproval, decisions map[string]ToolDecision, toolMap map[string]*Tool) error {
	pendingByID := make(map[string]ToolCall, len(pending))
	var missing, unaddressable []string
	for i := range pending {
		tc := pending[i].ToolCall
		// Decisions are keyed by ToolCallID; a pending call without a unique
		// non-empty ID cannot be addressed and must fail loudly rather than
		// collide in the maps below.
		if tc.ToolCallID == "" {
			unaddressable = append(unaddressable, tc.ToolName+" (empty tool call ID)")
			continue
		}
		if _, dup := pendingByID[tc.ToolCallID]; dup {
			unaddressable = append(unaddressable, tc.ToolCallID+" (duplicate tool call ID)")
			continue
		}
		pendingByID[tc.ToolCallID] = tc
		if _, ok := decisions[tc.ToolCallID]; !ok {
			missing = append(missing, tc.ToolCallID)
		}
	}
	if len(unaddressable) > 0 {
		return fmt.Errorf("twilightai: resume: pending tool calls cannot be addressed by decisions: %s", strings.Join(unaddressable, ", "))
	}
	if len(missing) > 0 {
		return fmt.Errorf("twilightai: resume: missing decisions for pending tool calls: %s", strings.Join(missing, ", "))
	}

	var unknown, invalid, unrunnable []string
	for id, d := range decisions {
		tc, isPending := pendingByID[id]
		if !isPending {
			unknown = append(unknown, id)
			continue
		}
		switch d.Decision {
		case ToolDecisionApproved:
			if tool, ok := toolMap[tc.ToolName]; !ok || tool.Execute == nil {
				unrunnable = append(unrunnable, fmt.Sprintf("%s (%s)", id, tc.ToolName))
			}
		case ToolDecisionRejected:
			// Nothing to check.
		default:
			// Includes the zero value: an absent explicit decision must never
			// execute an approval-gated tool.
			invalid = append(invalid, id)
		}
	}
	if len(unknown) > 0 {
		sort.Strings(unknown)
		return fmt.Errorf("twilightai: resume: decisions for unknown tool calls: %s", strings.Join(unknown, ", "))
	}
	if len(invalid) > 0 {
		sort.Strings(invalid)
		return fmt.Errorf("twilightai: resume: decisions must be explicitly approved or rejected: %s", strings.Join(invalid, ", "))
	}
	if len(unrunnable) > 0 {
		sort.Strings(unrunnable)
		return fmt.Errorf("twilightai: resume: approved tools missing from the tool set or not executable: %s", strings.Join(unrunnable, ", "))
	}
	return nil
}

// pendingToolCallsFromTail scans the tail of a conversation for the tool
// calls of the last assistant message that do not yet have a matching
// ToolResultPart in the tool messages that follow it. This is the state a
// run is left in after pausing on deferred approvals: the assistant message
// carries every call, trailing tool messages carry only the resolved
// results.
func pendingToolCallsFromTail(messages []Message) []ToolCall {
	resolved := make(map[string]bool)
	for i := len(messages) - 1; i >= 0; i-- {
		msg := messages[i]
		switch msg.Role {
		case MessageRoleTool:
			for _, part := range msg.Content {
				if trp, ok := part.(ToolResultPart); ok {
					resolved[trp.ToolCallID] = true
				}
			}
		case MessageRoleAssistant:
			var pending []ToolCall
			for _, part := range msg.Content {
				if tcp, ok := part.(ToolCallPart); ok && !resolved[tcp.ToolCallID] {
					pending = append(pending, ToolCall{
						ToolCallID:       tcp.ToolCallID,
						ToolName:         tcp.ToolName,
						Input:            tcp.Input,
						ProviderMetadata: tcp.ProviderMetadata,
					})
				}
			}
			return pending
		default:
			// A user/system/developer message means the conversation tail is
			// not a paused tool step.
			return nil
		}
	}
	return nil
}
