package sdk

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// startTestServer creates an in-memory MCP server with a single tool and
// returns the client-side transport. The server runs in the background until
// the returned cancel function is called.
func startTestServer(t *testing.T) mcp.Transport {
	t.Helper()

	server := mcp.NewServer(&mcp.Implementation{
		Name:    "test-server",
		Version: "v1.0.0",
	}, nil)

	type EchoInput struct {
		Message string `json:"message" jsonschema:"The message to echo back"`
	}

	mcp.AddTool(server, &mcp.Tool{
		Name:        "echo",
		Description: "Echoes the input message",
	}, func(_ context.Context, _ *mcp.CallToolRequest, input EchoInput) (*mcp.CallToolResult, any, error) {
		return &mcp.CallToolResult{
			Content: []mcp.Content{
				&mcp.TextContent{Text: "echo: " + input.Message},
			},
		}, nil, nil
	})

	type AddInput struct {
		A float64 `json:"a" jsonschema:"First number"`
		B float64 `json:"b" jsonschema:"Second number"`
	}
	type AddOutput struct {
		Sum float64 `json:"sum"`
	}

	mcp.AddTool(server, &mcp.Tool{
		Name:        "add",
		Description: "Adds two numbers",
	}, func(_ context.Context, _ *mcp.CallToolRequest, input AddInput) (*mcp.CallToolResult, AddOutput, error) {
		return nil, AddOutput{Sum: input.A + input.B}, nil
	})

	serverTransport, clientTransport := mcp.NewInMemoryTransports()

	go func() {
		_ = server.Run(context.Background(), serverTransport)
	}()

	return clientTransport
}

func TestCreateMCPClient(t *testing.T) {
	ctx := context.Background()
	transport := startTestServer(t)

	mc, err := CreateMCPClient(ctx, &MCPClientConfig{
		Transport: transport,
	})
	if err != nil {
		t.Fatalf("CreateMCPClient: %v", err)
	}
	defer mc.Close()

	if mc.session == nil {
		t.Fatal("session is nil")
	}
}

func TestMCPClientTools(t *testing.T) {
	ctx := context.Background()
	transport := startTestServer(t)

	mc, err := CreateMCPClient(ctx, &MCPClientConfig{Transport: transport})
	if err != nil {
		t.Fatalf("CreateMCPClient: %v", err)
	}
	defer mc.Close()

	tools, err := mc.Tools(ctx)
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}

	if len(tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(tools))
	}

	toolMap := make(map[string]Tool)
	for _, tool := range tools {
		toolMap[tool.Name] = tool
	}

	echo, ok := toolMap["echo"]
	if !ok {
		t.Fatal("missing echo tool")
	}
	if echo.Description != "Echoes the input message" {
		t.Errorf("echo description = %q", echo.Description)
	}
	if echo.Parameters == nil {
		t.Fatal("echo.Parameters is nil")
	}
	if echo.Execute == nil {
		t.Fatal("echo.Execute is nil")
	}

	add, ok := toolMap["add"]
	if !ok {
		t.Fatal("missing add tool")
	}
	if add.Description != "Adds two numbers" {
		t.Errorf("add description = %q", add.Description)
	}
}

func TestMCPToolExecute(t *testing.T) {
	ctx := context.Background()
	transport := startTestServer(t)

	mc, err := CreateMCPClient(ctx, &MCPClientConfig{Transport: transport})
	if err != nil {
		t.Fatalf("CreateMCPClient: %v", err)
	}
	defer mc.Close()

	tools, err := mc.Tools(ctx)
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}

	toolMap := make(map[string]Tool)
	for _, tool := range tools {
		toolMap[tool.Name] = tool
	}

	t.Run("echo", func(t *testing.T) {
		echoTool := toolMap["echo"]
		execCtx := &ToolExecContext{
			Context:    ctx,
			ToolCallID: "call-1",
			ToolName:   "echo",
		}
		result, err := echoTool.Execute(execCtx, map[string]any{"message": "hello"})
		if err != nil {
			t.Fatalf("Execute: %v", err)
		}
		text, ok := result.(string)
		if !ok {
			t.Fatalf("expected string result, got %T", result)
		}
		if text != "echo: hello" {
			t.Errorf("result = %q, want %q", text, "echo: hello")
		}
	})

	t.Run("add", func(t *testing.T) {
		addTool := toolMap["add"]
		execCtx := &ToolExecContext{
			Context:    ctx,
			ToolCallID: "call-2",
			ToolName:   "add",
		}
		result, err := addTool.Execute(execCtx, map[string]any{"a": 3.0, "b": 4.0})
		if err != nil {
			t.Fatalf("Execute: %v", err)
		}
		text, ok := result.(string)
		if !ok {
			t.Fatalf("expected string result, got %T", result)
		}
		if text == "" {
			t.Fatal("empty result")
		}
	})
}

func TestConvertInputSchema(t *testing.T) {
	t.Run("nil", func(t *testing.T) {
		s, err := convertInputSchema(nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if s != nil {
			t.Fatalf("expected nil, got %v", s)
		}
	})

	t.Run("already schema", func(t *testing.T) {
		original := &jsonschema.Schema{Type: "object"}
		s, err := convertInputSchema(original)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if s != original {
			t.Fatal("expected same pointer")
		}
	})

	t.Run("map to schema", func(t *testing.T) {
		m := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name": map[string]any{
					"type":        "string",
					"description": "A name",
				},
			},
			"required": []any{"name"},
		}
		s, err := convertInputSchema(m)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if s == nil {
			t.Fatal("expected non-nil schema")
		} else if s.Type != "object" {
			t.Errorf("type = %q, want %q", s.Type, "object")
		}
	})
}

func TestToJSONObject(t *testing.T) {
	t.Run("map passthrough", func(t *testing.T) {
		m := map[string]any{"a": 1}
		out, err := toJSONObject(m)
		if err != nil {
			t.Fatal(err)
		}
		if out["a"] != 1 {
			t.Errorf("unexpected value: %v", out)
		}
	})

	t.Run("struct conversion", func(t *testing.T) {
		type S struct {
			X int `json:"x"`
		}
		out, err := toJSONObject(S{X: 42})
		if err != nil {
			t.Fatal(err)
		}
		if out["x"] != float64(42) {
			t.Errorf("unexpected value: %v", out)
		}
	})
}

func TestExtractText(t *testing.T) {
	r := &mcp.CallToolResult{
		Content: []mcp.Content{
			&mcp.TextContent{Text: "line1"},
			&mcp.TextContent{Text: "line2"},
		},
	}
	text := extractText(r)
	if text != "line1\nline2" {
		t.Errorf("extractText = %q", text)
	}
}

func TestResolveTransport(t *testing.T) {
	t.Run("custom transport", func(t *testing.T) {
		st, ct := mcp.NewInMemoryTransports()
		_ = st
		tr, err := resolveTransport(&MCPClientConfig{Transport: ct})
		if err != nil {
			t.Fatal(err)
		}
		if tr != ct {
			t.Fatal("expected same transport")
		}
	})

	t.Run("missing URL", func(t *testing.T) {
		_, err := resolveTransport(&MCPClientConfig{})
		if err == nil {
			t.Fatal("expected error for missing URL")
		}
	})

	t.Run("unsupported type", func(t *testing.T) {
		_, err := resolveTransport(&MCPClientConfig{
			Type: "grpc",
			URL:  "http://localhost",
		})
		if err == nil {
			t.Fatal("expected error for unsupported type")
		}
	})

	t.Run("http default", func(t *testing.T) {
		tr, err := resolveTransport(&MCPClientConfig{
			URL: "http://localhost:8080",
		})
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := tr.(*mcp.StreamableClientTransport); !ok {
			t.Fatalf("expected StreamableClientTransport, got %T", tr)
		}
	})

	t.Run("sse", func(t *testing.T) {
		tr, err := resolveTransport(&MCPClientConfig{
			Type: MCPTransportSSE,
			URL:  "http://localhost:8080",
		})
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := tr.(*mcp.SSEClientTransport); !ok {
			t.Fatalf("expected SSEClientTransport, got %T", tr)
		}
	})
}

func TestMCPClientConfigDefaults(t *testing.T) {
	ctx := context.Background()
	transport := startTestServer(t)

	mc, err := CreateMCPClient(ctx, &MCPClientConfig{
		Transport: transport,
	})
	if err != nil {
		t.Fatalf("CreateMCPClient: %v", err)
	}
	defer mc.Close()

	// Verify defaults are applied (no panic, connection works)
	tools, err := mc.Tools(ctx)
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}
	if len(tools) == 0 {
		t.Fatal("expected tools")
	}
}

func TestConvertMCPToolsEmpty(t *testing.T) {
	tools, err := convertMCPTools(nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(tools) != 0 {
		t.Fatalf("expected 0 tools, got %d", len(tools))
	}
}

func TestMCPToolSchemaProperties(t *testing.T) {
	ctx := context.Background()
	transport := startTestServer(t)

	mc, err := CreateMCPClient(ctx, &MCPClientConfig{Transport: transport})
	if err != nil {
		t.Fatalf("CreateMCPClient: %v", err)
	}
	defer mc.Close()

	tools, err := mc.Tools(ctx)
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}

	toolMap := make(map[string]Tool)
	for _, tool := range tools {
		toolMap[tool.Name] = tool
	}

	// The echo tool's schema should have a "message" property.
	echo := toolMap["echo"]
	schemaJSON, err := json.Marshal(echo.Parameters)
	if err != nil {
		t.Fatal(err)
	}

	var raw map[string]any
	if err := json.Unmarshal(schemaJSON, &raw); err != nil {
		t.Fatal(err)
	}

	props, ok := raw["properties"].(map[string]any)
	if !ok {
		t.Fatalf("expected properties object, got %T", raw["properties"])
	}
	if _, ok := props["message"]; !ok {
		t.Fatal("expected 'message' in echo tool properties")
	}
}
