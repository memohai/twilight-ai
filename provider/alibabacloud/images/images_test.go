package images

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/memohai/twilight-ai/sdk"
)

func TestGenerateImageCreatesWanAsyncTaskAndPollsResult(t *testing.T) {
	t.Parallel()

	var polls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v1/services/aigc/image-generation/generation":
			if got := r.Method; got != http.MethodPost {
				t.Fatalf("method = %s, want POST", got)
			}
			if got := r.Header.Get("Authorization"); got != "Bearer dashscope-key" {
				t.Fatalf("authorization = %q, want bearer key", got)
			}
			if got := r.Header.Get("X-DashScope-Async"); got != "enable" {
				t.Fatalf("X-DashScope-Async = %q, want enable", got)
			}
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatalf("decode request body: %v", err)
			}
			if got := body["model"]; got != "wan2.7-image-pro" {
				t.Fatalf("model = %v, want wan2.7-image-pro", got)
			}
			input := body["input"].(map[string]any)
			messages := input["messages"].([]any)
			message := messages[0].(map[string]any)
			content := message["content"].([]any)
			textPart := content[0].(map[string]any)
			if got := message["role"]; got != "user" {
				t.Fatalf("role = %v, want user", got)
			}
			if got := textPart["text"]; got != "a red cube" {
				t.Fatalf("text = %v, want a red cube", got)
			}
			parameters := body["parameters"].(map[string]any)
			if got := parameters["size"]; got != "1024*1024" {
				t.Fatalf("size = %v, want 1024*1024", got)
			}
			if got := parameters["n"]; got != float64(1) {
				t.Fatalf("n = %v, want 1", got)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"output":{"task_id":"task-1","task_status":"PENDING"},"request_id":"req-1"}`))
		case "/api/v1/tasks/task-1":
			if got := r.Method; got != http.MethodGet {
				t.Fatalf("method = %s, want GET", got)
			}
			if got := r.Header.Get("Authorization"); got != "Bearer dashscope-key" {
				t.Fatalf("authorization = %q, want bearer key", got)
			}
			polls++
			w.Header().Set("Content-Type", "application/json")
			if polls == 1 {
				_, _ = w.Write([]byte(`{"output":{"task_id":"task-1","task_status":"RUNNING"}}`))
				return
			}
			_, _ = w.Write([]byte(`{
				"output": {
					"task_id": "task-1",
					"task_status": "SUCCEEDED",
					"choices": [{
						"finish_reason": "stop",
						"message": {
							"role": "assistant",
							"content": [{"image":"https://example.com/result.png","type":"image"}]
						}
					}]
				},
				"usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
			}`))
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	t.Cleanup(server.Close)

	provider := New(
		WithAPIKey("dashscope-key"),
		WithBaseURL(server.URL+"/compatible-mode/v1"),
		WithHTTPClient(server.Client()),
		WithPollInterval(time.Millisecond),
		WithPollTimeout(time.Second),
	)
	result, err := sdk.GenerateImage(context.Background(),
		sdk.WithImageGenerationModel(provider.GenerationModel("wan2.7-image-pro")),
		sdk.WithImagePrompt("a red cube"),
		sdk.WithImageSize("1024x1024"),
		sdk.WithImageN(1),
	)
	if err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	if len(result.Data) != 1 || result.Data[0].URL != "https://example.com/result.png" {
		t.Fatalf("result data = %+v, want result image URL", result.Data)
	}
	if result.Usage.TotalTokens != 5 || result.Usage.InputTokens != 2 || result.Usage.OutputTokens != 3 {
		t.Fatalf("usage = %+v, want token usage", result.Usage)
	}
	if polls != 2 {
		t.Fatalf("polls = %d, want 2", polls)
	}
}

func TestGenerateQwenLegacyImageCreatesText2ImageTaskAndParsesResults(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v1/services/aigc/text2image/image-synthesis":
			if got := r.Method; got != http.MethodPost {
				t.Fatalf("method = %s, want POST", got)
			}
			if got := r.Header.Get("Authorization"); got != "Bearer dashscope-key" {
				t.Fatalf("authorization = %q, want bearer key", got)
			}
			if got := r.Header.Get("X-DashScope-Async"); got != "enable" {
				t.Fatalf("X-DashScope-Async = %q, want enable", got)
			}
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatalf("decode request body: %v", err)
			}
			if got := body["model"]; got != "qwen-image" {
				t.Fatalf("model = %v, want qwen-image", got)
			}
			input := body["input"].(map[string]any)
			if got := input["prompt"]; got != "a red cube" {
				t.Fatalf("prompt = %v, want a red cube", got)
			}
			parameters := body["parameters"].(map[string]any)
			if got := parameters["size"]; got != "1024*1024" {
				t.Fatalf("size = %v, want 1024*1024", got)
			}
			if got := parameters["n"]; got != float64(1) {
				t.Fatalf("n = %v, want 1", got)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"output":{"task_id":"qwen-task","task_status":"PENDING"},"request_id":"req-qwen"}`))
		case "/api/v1/tasks/qwen-task":
			if got := r.Method; got != http.MethodGet {
				t.Fatalf("method = %s, want GET", got)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"output": {
					"task_id": "qwen-task",
					"task_status": "SUCCEEDED",
					"results": [{"url":"https://example.com/qwen.png"}]
				},
				"usage": {"image_count": 1}
			}`))
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	t.Cleanup(server.Close)

	provider := New(
		WithAPIKey("dashscope-key"),
		WithBaseURL(server.URL+"/compatible-mode/v1"),
		WithHTTPClient(server.Client()),
		WithPollInterval(time.Millisecond),
		WithPollTimeout(time.Second),
	)
	result, err := sdk.GenerateImage(context.Background(),
		sdk.WithImageGenerationModel(provider.GenerationModel("qwen-image")),
		sdk.WithImagePrompt("a red cube"),
		sdk.WithImageSize("1024x1024"),
		sdk.WithImageN(1),
	)
	if err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	if len(result.Data) != 1 || result.Data[0].URL != "https://example.com/qwen.png" {
		t.Fatalf("result data = %+v, want qwen image URL", result.Data)
	}
}

func TestGenerateLegacyWanImageCreatesText2ImageTask(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v1/services/aigc/text2image/image-synthesis":
			if got := r.Method; got != http.MethodPost {
				t.Fatalf("method = %s, want POST", got)
			}
			if got := r.Header.Get("X-DashScope-Async"); got != "enable" {
				t.Fatalf("X-DashScope-Async = %q, want enable", got)
			}
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatalf("decode request body: %v", err)
			}
			if got := body["model"]; got != "wanx-v1" {
				t.Fatalf("model = %v, want wanx-v1", got)
			}
			input := body["input"].(map[string]any)
			if got := input["prompt"]; got != "a red cube" {
				t.Fatalf("prompt = %v, want a red cube", got)
			}
			if _, ok := input["messages"]; ok {
				t.Fatalf("messages input was sent for legacy Wan model: %v", input["messages"])
			}
			parameters := body["parameters"].(map[string]any)
			if got := parameters["size"]; got != "1024*1024" {
				t.Fatalf("size = %v, want 1024*1024", got)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"output":{"task_id":"wanx-task","task_status":"PENDING"}}`))
		case "/api/v1/tasks/wanx-task":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"output": {
					"task_id": "wanx-task",
					"task_status": "SUCCEEDED",
					"results": [{"url":"https://example.com/wanx.png"}]
				}
			}`))
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	t.Cleanup(server.Close)

	provider := New(
		WithAPIKey("dashscope-key"),
		WithBaseURL(server.URL+"/api/v1"),
		WithHTTPClient(server.Client()),
		WithPollInterval(time.Millisecond),
		WithPollTimeout(time.Second),
	)
	result, err := sdk.GenerateImage(context.Background(),
		sdk.WithImageGenerationModel(provider.GenerationModel("wanx-v1")),
		sdk.WithImagePrompt("a red cube"),
		sdk.WithImageSize("1024x1024"),
	)
	if err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	if len(result.Data) != 1 || result.Data[0].URL != "https://example.com/wanx.png" {
		t.Fatalf("result data = %+v, want legacy Wan image URL", result.Data)
	}
}

func TestGenerateQwenMultimodalUsesSynchronousEndpoint(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v1/services/aigc/multimodal-generation/generation":
			if got := r.Method; got != http.MethodPost {
				t.Fatalf("method = %s, want POST", got)
			}
			if got := r.Header.Get("Authorization"); got != "Bearer dashscope-key" {
				t.Fatalf("authorization = %q, want bearer key", got)
			}
			if got := r.Header.Get("X-DashScope-Async"); got != "" {
				t.Fatalf("X-DashScope-Async = %q, want empty", got)
			}
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatalf("decode request body: %v", err)
			}
			if got := body["model"]; got != "qwen-image-edit" {
				t.Fatalf("model = %v, want qwen-image-edit", got)
			}
			input := body["input"].(map[string]any)
			messages := input["messages"].([]any)
			message := messages[0].(map[string]any)
			content := message["content"].([]any)
			textPart := content[0].(map[string]any)
			if got := textPart["text"]; got != "a red cube" {
				t.Fatalf("text = %v, want a red cube", got)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"output": {
					"choices": [{
						"finish_reason": "stop",
						"message": {
							"role": "assistant",
							"content": [{"image":"https://example.com/qwen-edit.png","type":"image"}]
						}
					}]
				},
				"usage": {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10}
			}`))
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	t.Cleanup(server.Close)

	provider := New(
		WithAPIKey("dashscope-key"),
		WithBaseURL(server.URL+"/api/v1"),
		WithHTTPClient(server.Client()),
	)
	result, err := sdk.GenerateImage(context.Background(),
		sdk.WithImageGenerationModel(provider.GenerationModel("qwen-image-edit")),
		sdk.WithImagePrompt("a red cube"),
	)
	if err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	if len(result.Data) != 1 || result.Data[0].URL != "https://example.com/qwen-edit.png" {
		t.Fatalf("result data = %+v, want qwen multimodal image URL", result.Data)
	}
	if result.Usage.TotalTokens != 10 || result.Usage.InputTokens != 4 || result.Usage.OutputTokens != 6 {
		t.Fatalf("usage = %+v, want token usage", result.Usage)
	}
}

func TestGenerateDatedQwenPlusUsesSynchronousEndpoint(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v1/services/aigc/multimodal-generation/generation":
			if got := r.Method; got != http.MethodPost {
				t.Fatalf("method = %s, want POST", got)
			}
			if got := r.Header.Get("X-DashScope-Async"); got != "" {
				t.Fatalf("X-DashScope-Async = %q, want empty", got)
			}
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatalf("decode request body: %v", err)
			}
			if got := body["model"]; got != "qwen-image-plus-2026-01-09" {
				t.Fatalf("model = %v, want qwen-image-plus-2026-01-09", got)
			}
			input := body["input"].(map[string]any)
			messages := input["messages"].([]any)
			message := messages[0].(map[string]any)
			content := message["content"].([]any)
			textPart := content[0].(map[string]any)
			if got := textPart["text"]; got != "a red cube" {
				t.Fatalf("text = %v, want a red cube", got)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"output": {
					"choices": [{
						"finish_reason": "stop",
						"message": {
							"role": "assistant",
							"content": [{"image":"https://example.com/qwen-plus-dated.png","type":"image"}]
						}
					}]
				}
			}`))
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	t.Cleanup(server.Close)

	provider := New(
		WithAPIKey("dashscope-key"),
		WithBaseURL(server.URL+"/api/v1"),
		WithHTTPClient(server.Client()),
	)
	result, err := sdk.GenerateImage(context.Background(),
		sdk.WithImageGenerationModel(provider.GenerationModel("qwen-image-plus-2026-01-09")),
		sdk.WithImagePrompt("a red cube"),
	)
	if err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	if len(result.Data) != 1 || result.Data[0].URL != "https://example.com/qwen-plus-dated.png" {
		t.Fatalf("result data = %+v, want dated qwen plus image URL", result.Data)
	}
}

func TestGenerateImageReturnsTaskFailure(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v1/services/aigc/text2image/image-synthesis":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"output":{"task_id":"task-2","task_status":"PENDING"}}`))
		case "/api/v1/tasks/task-2":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"output":{"task_id":"task-2","task_status":"FAILED","message":"content rejected"}}`))
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	t.Cleanup(server.Close)

	provider := New(
		WithAPIKey("dashscope-key"),
		WithBaseURL(server.URL+"/api/v1"),
		WithHTTPClient(server.Client()),
		WithPollInterval(time.Millisecond),
		WithPollTimeout(time.Second),
	)
	_, err := sdk.GenerateImage(context.Background(),
		sdk.WithImageGenerationModel(provider.GenerationModel("qwen-image")),
		sdk.WithImagePrompt("blocked prompt"),
	)
	if err == nil || !strings.Contains(err.Error(), "FAILED") || !strings.Contains(err.Error(), "content rejected") {
		t.Fatalf("GenerateImage() error = %v, want task failure", err)
	}
}

func TestNormalizeBaseURL(t *testing.T) {
	t.Parallel()

	tests := []struct {
		input string
		want  string
	}{
		{"", defaultBaseURL},
		{"https://dashscope.aliyuncs.com", "https://dashscope.aliyuncs.com/api/v1"},
		{"https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", "https://workspace.cn-beijing.maas.aliyuncs.com/api/v1"},
		{"https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions", "https://workspace.cn-beijing.maas.aliyuncs.com/api/v1"},
		{"https://workspace.cn-beijing.maas.aliyuncs.com/api/v1/services/aigc/image-generation/generation", "https://workspace.cn-beijing.maas.aliyuncs.com/api/v1"},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			if got := normalizeBaseURL(tt.input); got != tt.want {
				t.Fatalf("normalizeBaseURL(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

// ---------- integration tests (real API, skipped unless explicitly enabled) ----------

func envOrSkip(t *testing.T, key string) string {
	t.Helper()
	v := os.Getenv(key)
	if v == "" {
		t.Skipf("skipping: %s not set", key)
	}
	return v
}

func imageLiveOrSkip(t *testing.T) {
	t.Helper()
	if os.Getenv("DASHSCOPE_IMAGE_LIVE") != "1" {
		t.Skip("skipping: set DASHSCOPE_IMAGE_LIVE=1 to run real image generation tests")
	}
}

func newIntegrationProvider(t *testing.T) *Provider {
	t.Helper()
	imageLiveOrSkip(t)
	apiKey := envOrSkip(t, "DASHSCOPE_API_KEY")
	opts := []Option{WithAPIKey(apiKey)}
	if base := os.Getenv("DASHSCOPE_BASE_URL"); base != "" {
		opts = append(opts, WithBaseURL(base))
	}
	return New(opts...)
}

func integrationEnv(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func TestIntegration_GenerateQwenImage(t *testing.T) {
	provider := newIntegrationProvider(t)
	modelID := integrationEnv("DASHSCOPE_QWEN_IMAGE_MODEL", "qwen-image")

	ctx, cancel := context.WithTimeout(context.Background(), defaultPollTimeout+time.Minute)
	defer cancel()

	result, err := sdk.GenerateImage(ctx,
		sdk.WithImageGenerationModel(provider.GenerationModel(modelID)),
		sdk.WithImagePrompt("A single red cube on a clean white background, no text."),
		sdk.WithImageN(1),
	)
	if err != nil {
		t.Fatalf("GenerateImage(%s): %v", modelID, err)
	}
	assertIntegrationImageURL(t, result)
}

func TestIntegration_GenerateWanImage(t *testing.T) {
	provider := newIntegrationProvider(t)
	modelID := integrationEnv("DASHSCOPE_WAN_IMAGE_MODEL", "wan2.6-t2i")

	ctx, cancel := context.WithTimeout(context.Background(), defaultPollTimeout+time.Minute)
	defer cancel()

	result, err := sdk.GenerateImage(ctx,
		sdk.WithImageGenerationModel(provider.GenerationModel(modelID)),
		sdk.WithImagePrompt("A simple watercolor landscape with one tree beside a lake, no text."),
		sdk.WithImageSize("1024x1024"),
		sdk.WithImageN(1),
	)
	if err != nil {
		t.Fatalf("GenerateImage(%s): %v", modelID, err)
	}
	assertIntegrationImageURL(t, result)
}

func assertIntegrationImageURL(t *testing.T, result *sdk.ImageResult) {
	t.Helper()
	if result == nil {
		t.Fatal("GenerateImage returned nil result")
	}
	if len(result.Data) == 0 {
		t.Fatal("GenerateImage returned no image data")
	}
	if strings.TrimSpace(result.Data[0].URL) == "" && strings.TrimSpace(result.Data[0].B64JSON) == "" {
		t.Fatalf("first image has no URL or b64_json: %+v", result.Data[0])
	}
	t.Logf("image generated: url_set=%t b64_set=%t", strings.TrimSpace(result.Data[0].URL) != "", strings.TrimSpace(result.Data[0].B64JSON) != "")
}
