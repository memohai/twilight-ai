# Tool Calling

Twilight AI supports LLM tool calling (also known as function calling) with automatic multi-step execution. You define tools with execution handlers, and the SDK manages the call-execute-respond loop.

## Defining a Tool

There are three ways to define a tool's parameter schema.

### Using `NewTool[T]` (recommended)

The generic `NewTool` function infers the JSON Schema from a Go struct and provides type-safe input in the `Execute` handler:

```go
type WeatherParams struct {
    City string `json:"city" jsonschema:"City name, e.g. 'Tokyo'"`
}

weatherTool := sdk.NewTool("get_weather", "Get the current weather for a given city",
    func(ctx *sdk.ToolExecContext, input WeatherParams) (any, error) {
        return map[string]any{
            "city":    input.City,
            "temp":    "22°C",
            "weather": "sunny",
        }, nil
    },
)
```

### Passing a Go struct

You can pass a struct value directly to `Parameters`. The SDK infers the JSON Schema via reflection before sending to the provider:

```go
weatherTool := sdk.Tool{
    Name:        "get_weather",
    Description: "Get the current weather for a given city",
    Parameters:  WeatherParams{},
    Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
        args := input.(map[string]any)
        city := args["city"].(string)
        return map[string]any{"city": city, "temp": "22°C"}, nil
    },
}
```

### Using `*jsonschema.Schema` directly

For full control over the schema, construct a `*jsonschema.Schema` value:

```go
import "github.com/google/jsonschema-go/jsonschema"

weatherTool := sdk.Tool{
    Name:        "get_weather",
    Description: "Get the current weather for a given city",
    Parameters: &jsonschema.Schema{
        Type: "object",
        Properties: map[string]*jsonschema.Schema{
            "city": {Type: "string", Description: "City name, e.g. 'Tokyo'"},
        },
        Required: []string{"city"},
    },
    Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
        args := input.(map[string]any)
        city := args["city"].(string)
        return map[string]any{"city": city, "temp": "22°C"}, nil
    },
}
```

### Tool Fields

| Field | Type | Description |
|-------|------|-------------|
| `Name` | `string` | Unique tool name passed to the LLM |
| `Description` | `string` | Human-readable description for the LLM |
| `Parameters` | `any` | Go struct (auto-inferred) or `*jsonschema.Schema` |
| `Execute` | `ToolExecuteFunc` | Go function that runs when the LLM calls this tool |
| `RequireApproval` | `bool` | If true, requires approval before execution |

## Using MCP Tools

Twilight AI can load remote tools from an MCP server and expose them as normal `sdk.Tool` values.

This is useful when:

- the tool already exists behind an MCP server
- you want to share the same tool inventory across multiple apps
- you want the model to call remote tools without writing a local `Execute` handler

### Create an MCP client

Use `CreateMCPClient` with HTTP, SSE, or a custom transport:

```go
import (
    "context"

    "github.com/memohai/twilight-ai/sdk"
)

mcpClient, err := sdk.CreateMCPClient(context.Background(), &sdk.MCPClientConfig{
    Type: sdk.MCPTransportHTTP, // default; may be omitted
    URL:  "https://example.com/mcp",
    Headers: map[string]string{
        "Authorization": "Bearer <token>",
    },
})
if err != nil {
    log.Fatal(err)
}
defer mcpClient.Close()
```

### Supported transport patterns

| Pattern | How to configure |
|--------|------------------|
| Streamable HTTP | `Type: sdk.MCPTransportHTTP`, `URL: "https://.../mcp"` |
| SSE | `Type: sdk.MCPTransportSSE`, `URL: "https://.../sse"` |
| Stdio / custom | Create `mcp.Transport` yourself and pass `Transport: ...` |

For stdio, Twilight AI intentionally does not create the transport for you. Build it using the official MCP Go SDK:

```go
import (
    "context"
    "os/exec"

    "github.com/memohai/twilight-ai/sdk"
    "github.com/modelcontextprotocol/go-sdk/mcp"
)

transport := &mcp.CommandTransport{
    Command: exec.Command("my-mcp-server"),
}

mcpClient, err := sdk.CreateMCPClient(context.Background(), &sdk.MCPClientConfig{
    Transport: transport,
})
if err != nil {
    log.Fatal(err)
}
defer mcpClient.Close()
```

### Convert MCP tools into Twilight tools

```go
tools, err := mcpClient.Tools(ctx)
if err != nil {
    log.Fatal(err)
}

result, err := sdk.GenerateTextResult(ctx,
    sdk.WithModel(model),
    sdk.WithMessages([]sdk.Message{
        sdk.UserMessage("Use the available MCP tools to answer this request."),
    }),
    sdk.WithTools(tools),
    sdk.WithMaxSteps(5),
)
```

### What gets converted automatically

When you call `mcpClient.Tools(ctx)`, Twilight AI:

1. calls `tools/list` on the MCP server
2. converts each `mcp.Tool.InputSchema` into `*jsonschema.Schema`
3. creates an `sdk.Tool.Execute` wrapper that calls `tools/call`
4. returns MCP text content as the tool output seen by the model

MCP tools behave like normal Twilight AI tools once loaded, so they work with:

- `WithTools(...)`
- `WithMaxSteps(...)`
- `GenerateTextResult(...)`
- `StreamText(...)`

### ToolExecContext

The execution function receives a `*ToolExecContext` that embeds `context.Context` and provides additional metadata:

```go
type ToolExecContext struct {
    context.Context
    ToolCallID   string           // unique ID for this call
    ToolName     string           // name of the tool being called
    SendProgress func(content any) // send progress updates (nil when not streaming)
}
```

## Single-Step Tool Calling

With `MaxSteps` at its default (`0`), the SDK returns the tool call without executing it:

```go
result, err := sdk.GenerateTextResult(ctx,
    sdk.WithModel(model),
    sdk.WithMessages([]sdk.Message{
        sdk.UserMessage("What's the weather in Tokyo?"),
    }),
    sdk.WithTools([]sdk.Tool{weatherTool}),
)

// result.ToolCalls contains the LLM's tool call request
// result.Text may be empty — the LLM chose to call a tool instead
for _, tc := range result.ToolCalls {
    fmt.Printf("Tool: %s, Input: %v\n", tc.ToolName, tc.Input)
}
```

## Multi-Step Execution

Set `WithMaxSteps` to enable automatic tool execution. The SDK will:

1. Send messages to the LLM
2. If the LLM returns tool calls, execute them
3. Append tool results to the conversation
4. Send updated messages back to the LLM
5. Repeat until the LLM stops calling tools or the step limit is reached

```go
result, err := sdk.GenerateTextResult(ctx,
    sdk.WithModel(model),
    sdk.WithMessages([]sdk.Message{
        sdk.UserMessage("What's the weather in Tokyo and Paris?"),
    }),
    sdk.WithTools([]sdk.Tool{weatherTool}),
    sdk.WithMaxSteps(10),
)

// result.Text contains the final response after all tool calls
// result.Steps contains each step's details
fmt.Println(result.Text)
fmt.Printf("Completed in %d steps\n", len(result.Steps))
```

### MaxSteps Values

| Value | Behavior |
|-------|----------|
| `0` (default) | Single LLM call, no tool auto-execution |
| `N` (N > 0) | Up to N LLM calls in the loop |
| `-1` | Unlimited — loops until the LLM stops requesting tools |

## Tool Choice

Control how the LLM decides whether to use tools:

```go
sdk.WithToolChoice("auto")     // LLM decides (default)
sdk.WithToolChoice("none")     // never call tools
sdk.WithToolChoice("required") // must call at least one tool
```

## Approval Flow

For sensitive operations, mark tools with `RequireApproval` and provide an approval handler:

```go
dangerousTool := sdk.Tool{
    Name:            "delete_file",
    Description:     "Delete a file from the filesystem",
    Parameters:      fileSchema,
    RequireApproval: true,
    Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
        // This only runs if approved
        path := input.(map[string]any)["path"].(string)
        return os.Remove(path), nil
    },
}

result, err := sdk.GenerateTextResult(ctx,
    sdk.WithModel(model),
    sdk.WithMessages(msgs),
    sdk.WithTools([]sdk.Tool{dangerousTool}),
    sdk.WithMaxSteps(5),
    sdk.WithApprovalHandler(func(ctx context.Context, call sdk.ToolCall) (bool, error) {
        fmt.Printf("Allow %s with input %v? [y/n] ", call.ToolName, call.Input)
        var answer string
        fmt.Scanln(&answer)
        return answer == "y", nil
    }),
)
```

When a tool call is denied, a `ToolOutputDeniedPart` is sent in streaming mode, and the tool result is marked as an error.

### Deferred Approvals

When the decision cannot be made synchronously (for example, it needs user input in a UI), use the full `WithApprovalHandler` form and return `ToolApprovalDecisionDeferred`. The run pauses at that step instead of executing the deferred calls.

Every call in the step still gets its approval check: calls that need no approval or were approved execute as usual and their results are recorded, while all deferred calls are surfaced together. The approval phase itself is side-effect-free — every handler is consulted before anything executes, so a handler error fails the batch cleanly without orphaning already-announced approvals.

```go
result, err := sdk.GenerateTextResult(ctx,
    sdk.WithModel(model),
    sdk.WithMessages(msgs),
    sdk.WithTools(tools),
    sdk.WithMaxSteps(5),
    sdk.WithApprovalHandler(func(ctx context.Context, call sdk.ToolCall) (sdk.ToolApprovalResult, error) {
        id := approvalStore.CreatePending(call) // persist for the UI
        return sdk.ToolApprovalResult{
            Decision:   sdk.ToolApprovalDecisionDeferred,
            ApprovalID: id,
        }, nil
    }),
)

if result.FinishReason == sdk.FinishReasonPaused {
    pause := result.Pause // portable resume state: full conversation + pending calls
    for _, d := range pause.Pending {
        // d.ToolCall is the pending call; d.Approval.ApprovalID identifies the decision.
    }
}
```

A paused run reports `FinishReasonPaused` on the overall result (and on the stream's `FinishPart`), so it is always distinguishable from a normal `tool-calls` finish. `Result.Pause` is a `ToolApprovalPause` — plain data carrying the full conversation and the pending calls. Persist it (it round-trips through JSON) and hand it back once decisions arrive. Individual steps keep the provider's finish reason. In streaming mode, one `ToolApprovalRequestPart` is emitted per deferred call.

The paused step's `Messages` include the assistant message with all tool calls plus a tool message covering the already-resolved calls. Migration note for pre-0.5 integrations: the paused step used to record no tool results at all — code that appended a complete tool message covering every call on resume must now supply results only for the deferred calls, or providers will reject the duplicate results. (Persisted JSON from older versions used the singular `deferredToolApproval` key, which this version no longer reads.)

### Resuming After Decisions

Once the decisions arrive, hand the pause back with one explicit `ToolDecision` per pending call, keyed by `ToolCallID`. `ResumeText` (blocking) and `ResumeTextStream` (streaming) apply the decisions and continue generation. Do not pass `WithMessages` — the conversation comes from the pause:

```go
decisions := map[string]sdk.ToolDecision{}
for _, d := range pause.Pending {
    decisions[d.ToolCall.ToolCallID] = decideFromUI(d) // approved or rejected
}

resumed, err := sdk.ResumeText(ctx, *pause, decisions,
    sdk.WithModel(model),
    sdk.WithTools(tools),               // same tool set as the original run
    sdk.WithMaxSteps(5),
    sdk.WithApprovalHandler(handler),   // keep it: the model may pause again
)
```

Before the first model call, the SDK validates and applies the decisions: approved calls execute through the normal tool path, rejected calls get an error result carrying the decision's `Reason`. Validation fails fast — before any tool execution or model call — on a missing decision, a decision for an unknown call, a decision that is not explicitly approved or rejected (the zero value is not approval), or an approved call whose tool is missing from `WithTools`. The pause is also cross-checked against its own conversation: `Pending` must match the calls the `Messages` tail leaves unresolved, so a hand-assembled pause that disagrees with itself fails loudly.

The applied decisions are reported on `result.Resume` (a `ToolApprovalResolution` with the results and the completing tool message) rather than as a synthetic step: `Steps`, `PrepareStep`, and `OnStepCommitted` see only real model steps, numbered exactly as in a fresh run. If the model requests another gated call, the resumed run pauses again with a fresh `result.Pause`. In streaming mode the decisions are applied before the stream opens, so the stream carries a normal provider lifecycle and `StreamResult.Resume` is available immediately.

When tool side effects must not run twice (deployments, payments), split the phases with `ApplyToolDecisions`: it applies the decisions and returns the completing tool message without any model call — no model configuration needed — so you can persist the completed conversation first and retry generation alone on failure:

```go
resolution, err := sdk.ApplyToolDecisions(ctx, *pause, decisions, tools)
if err != nil {
    return err
}
// slices.Concat allocates a fresh slice: appending to pause.Messages could
// write into the pause's backing array after a JSON reload.
completed := slices.Concat(pause.Messages, resolution.Messages)
persist(completed) // durable before the model call
result, err := sdk.GenerateTextResult(ctx,
    sdk.WithModel(model), sdk.WithMessages(completed),
    sdk.WithTools(tools), sdk.WithMaxSteps(5)) // safe to retry
```

## Streaming with Tools

Tool calling works seamlessly with `StreamText`. Progress updates from tool execution are delivered through the stream:

```go
sr, err := sdk.StreamText(ctx,
    sdk.WithModel(model),
    sdk.WithMessages([]sdk.Message{
        sdk.UserMessage("What's the weather in Tokyo?"),
    }),
    sdk.WithTools([]sdk.Tool{weatherTool}),
    sdk.WithMaxSteps(5),
)

for part := range sr.Stream {
    switch p := part.(type) {
    case *sdk.TextDeltaPart:
        fmt.Print(p.Text)
    case *sdk.StreamToolCallPart:
        fmt.Printf("\n[Calling tool: %s]\n", p.ToolName)
    case *sdk.StreamToolResultPart:
        fmt.Printf("[Tool result: %v]\n", p.Output)
    case *sdk.ToolProgressPart:
        fmt.Printf("[Progress: %v]\n", p.Content)
    case *sdk.ErrorPart:
        log.Fatal(p.Error)
    }
}
```

### Sending Progress from Tools

During streaming, tools can send progress updates via `SendProgress`:

```go
Execute: func(ctx *sdk.ToolExecContext, input any) (any, error) {
    if ctx.SendProgress != nil {
        ctx.SendProgress("Fetching data...")
    }
    // do work...
    if ctx.SendProgress != nil {
        ctx.SendProgress("Processing results...")
    }
    return result, nil
},
```

## Inspecting Steps

After a multi-step execution, inspect individual steps:

```go
for i, step := range result.Steps {
    fmt.Printf("Step %d: finish=%s, tokens=%d\n",
        i+1, step.FinishReason, step.Usage.TotalTokens)

    for _, tc := range step.ToolCalls {
        fmt.Printf("  Called: %s(%v)\n", tc.ToolName, tc.Input)
    }
    for _, tr := range step.ToolResults {
        fmt.Printf("  Result: %v\n", tr.Output)
    }
}
```

## Callbacks

### OnStep

Called after each step completes. Can override params for the next step:

```go
sdk.WithOnStep(func(step *sdk.StepResult) *sdk.GenerateParams {
    fmt.Printf("Step finished: %s\n", step.FinishReason)
    return nil // return non-nil to override next step's params
}),
```

### OnStepCommitted

Use a synchronous durability barrier after a complete step is assembled and
before the next model call begins. Returning an error stops generation and
leaves that step out of the accumulated result:

```go
sdk.WithOnStepCommitted(func(ctx context.Context, stepIndex int, step *sdk.StepResult) error {
    return persistStep(ctx, stepIndex, step)
}),
```

### PrepareStep

Called before each step (starting from step 2). Allows modifying params:

```go
sdk.WithPrepareStep(func(params *sdk.GenerateParams) *sdk.GenerateParams {
    // Reduce temperature after first step
    t := 0.3
    params.Temperature = &t
    return params
}),
```

### OnFinish

Called once when all steps are complete:

```go
sdk.WithOnFinish(func(result *sdk.GenerateResult) {
    fmt.Printf("Done! Total tokens: %d\n", result.Usage.TotalTokens)
}),
```

## Next Steps

- [Streaming](streaming.md) — deep dive into StreamPart types
- [API Reference](api-reference.md) — complete type and function reference
