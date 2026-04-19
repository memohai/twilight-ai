package embedding_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/memohai/twilight-ai/provider/openai/embedding"
	"github.com/memohai/twilight-ai/sdk"
)

func newTestServer(t *testing.T, handler http.HandlerFunc) (*httptest.Server, *embedding.Provider) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	p := embedding.New(
		embedding.WithAPIKey("test-key"),
		embedding.WithBaseURL(srv.URL),
	)
	return srv, p
}

func TestDoEmbed(t *testing.T) {
	_, p := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("unexpected auth header: %s", r.Header.Get("Authorization"))
		}

		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)

		if body["model"] != "text-embedding-3-small" {
			t.Errorf("unexpected model: %v", body["model"])
		}
		if body["encoding_format"] != "float" {
			t.Errorf("unexpected encoding_format: %v", body["encoding_format"])
		}

		input, ok := body["input"].([]any)
		if !ok || len(input) != 2 {
			t.Errorf("expected 2 input values, got %v", body["input"])
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"data": []map[string]any{
				{"object": "embedding", "index": 0, "embedding": []float64{0.1, 0.2, 0.3}},
				{"object": "embedding", "index": 1, "embedding": []float64{0.4, 0.5, 0.6}},
			},
			"model": "text-embedding-3-small",
			"usage": map[string]any{
				"prompt_tokens": 12,
				"total_tokens":  12,
			},
		})
	})

	model := p.EmbeddingModel("text-embedding-3-small")
	result, err := p.DoEmbed(context.Background(), sdk.EmbedParams{
		Model:  model,
		Values: []string{"sunny day at the beach", "rainy day in the city"},
	})
	if err != nil {
		t.Fatalf("DoEmbed failed: %v", err)
	}

	if len(result.Embeddings) != 2 {
		t.Fatalf("expected 2 embeddings, got %d", len(result.Embeddings))
	}
	if len(result.Embeddings[0]) != 3 {
		t.Errorf("expected 3 dimensions, got %d", len(result.Embeddings[0]))
	}
	if result.Embeddings[0][0] != 0.1 {
		t.Errorf("expected first value 0.1, got %f", result.Embeddings[0][0])
	}
	if result.Embeddings[1][0] != 0.4 {
		t.Errorf("expected first value of second embedding 0.4, got %f", result.Embeddings[1][0])
	}
	if result.Usage.Tokens != 12 {
		t.Errorf("expected 12 tokens, got %d", result.Usage.Tokens)
	}
}

func TestDoEmbed_WithBedrockCredentials(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got == "" || got[:16] != "AWS4-HMAC-SHA256" {
			t.Fatalf("expected SigV4 auth header, got %q", got)
		}
		if got := r.Header.Get("X-Amz-Date"); got == "" {
			t.Fatal("expected X-Amz-Date header")
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"data": []map[string]any{
				{"object": "embedding", "index": 0, "embedding": []float64{0.1, 0.2}},
			},
			"model": "amazon.titan-embed-text-v2:0",
			"usage": map[string]any{
				"prompt_tokens": 4,
				"total_tokens":  4,
			},
		})
	}))
	defer srv.Close()

	p := embedding.New(
		embedding.WithBaseURL(srv.URL),
		embedding.WithBedrockCredentials("us-east-1", "AKIDEXAMPLE", "secret", ""),
	)

	result, err := p.DoEmbed(context.Background(), sdk.EmbedParams{
		Model:  p.EmbeddingModel("amazon.titan-embed-text-v2:0"),
		Values: []string{"hello"},
	})
	if err != nil {
		t.Fatalf("DoEmbed failed: %v", err)
	}
	if len(result.Embeddings) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(result.Embeddings))
	}
}

func TestDoEmbed_WithDimensions(t *testing.T) {
	var capturedBody map[string]any

	_, p := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&capturedBody)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"data": []map[string]any{
				{"object": "embedding", "index": 0, "embedding": []float64{0.1, 0.2}},
			},
			"model": "text-embedding-3-small",
			"usage": map[string]any{"prompt_tokens": 5, "total_tokens": 5},
		})
	})

	model := p.EmbeddingModel("text-embedding-3-small")
	dims := 64
	_, err := p.DoEmbed(context.Background(), sdk.EmbedParams{
		Model:      model,
		Values:     []string{"hello"},
		Dimensions: &dims,
	})
	if err != nil {
		t.Fatalf("DoEmbed failed: %v", err)
	}

	if capturedBody["dimensions"] != float64(64) {
		t.Errorf("expected dimensions 64, got %v", capturedBody["dimensions"])
	}
}

func TestDoEmbed_NilModel(t *testing.T) {
	p := embedding.New(embedding.WithAPIKey("test-key"))

	_, err := p.DoEmbed(context.Background(), sdk.EmbedParams{
		Values: []string{"hello"},
	})
	if err == nil {
		t.Fatal("expected error for nil model")
	}
}

func TestDoEmbed_APIError(t *testing.T) {
	_, p := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"message": "Incorrect API key provided",
				"type":    "invalid_request_error",
			},
		})
	})

	model := p.EmbeddingModel("text-embedding-3-small")
	_, err := p.DoEmbed(context.Background(), sdk.EmbedParams{
		Model:  model,
		Values: []string{"hello"},
	})
	if err == nil {
		t.Fatal("expected error for 401 response")
	}
}

func TestEmbeddingModel(t *testing.T) {
	p := embedding.New(embedding.WithAPIKey("test-key"))
	model := p.EmbeddingModel("text-embedding-3-large")

	if model.ID != "text-embedding-3-large" {
		t.Errorf("expected model ID 'text-embedding-3-large', got %q", model.ID)
	}
	if model.MaxEmbeddingsPerCall != 2048 {
		t.Errorf("expected MaxEmbeddingsPerCall 2048, got %d", model.MaxEmbeddingsPerCall)
	}
	if model.Provider == nil {
		t.Error("expected non-nil provider")
	}
}

func TestClientEmbed(t *testing.T) {
	_, p := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"data": []map[string]any{
				{"object": "embedding", "index": 0, "embedding": []float64{0.7, 0.8, 0.9}},
			},
			"model": "text-embedding-3-small",
			"usage": map[string]any{"prompt_tokens": 4, "total_tokens": 4},
		})
	})

	model := p.EmbeddingModel("text-embedding-3-small")

	vec, err := sdk.Embed(context.Background(), "test",
		sdk.WithEmbeddingModel(model),
	)
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}
	if len(vec) != 3 {
		t.Errorf("expected 3 dimensions, got %d", len(vec))
	}
	if vec[0] != 0.7 {
		t.Errorf("expected 0.7, got %f", vec[0])
	}
}

func TestClientEmbedMany(t *testing.T) {
	_, p := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"data": []map[string]any{
				{"object": "embedding", "index": 0, "embedding": []float64{0.1, 0.2}},
				{"object": "embedding", "index": 1, "embedding": []float64{0.3, 0.4}},
			},
			"model": "text-embedding-3-small",
			"usage": map[string]any{"prompt_tokens": 8, "total_tokens": 8},
		})
	})

	model := p.EmbeddingModel("text-embedding-3-small")

	result, err := sdk.EmbedMany(context.Background(), []string{"a", "b"},
		sdk.WithEmbeddingModel(model),
	)
	if err != nil {
		t.Fatalf("EmbedMany failed: %v", err)
	}
	if len(result.Embeddings) != 2 {
		t.Fatalf("expected 2 embeddings, got %d", len(result.Embeddings))
	}
	if result.Usage.Tokens != 8 {
		t.Errorf("expected 8 tokens, got %d", result.Usage.Tokens)
	}
}
