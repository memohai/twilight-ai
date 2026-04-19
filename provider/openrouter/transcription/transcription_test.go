package transcription

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	sdk "github.com/memohai/twilight-ai/sdk"
)

func TestProvider_ListModels(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"data":[{"id":"openai/gpt-4o-mini-transcribe","architecture":{"input_modalities":["audio"],"output_modalities":["text"]}},{"id":"openai/gpt-4o-mini","architecture":{"input_modalities":["text"],"output_modalities":["text"]}}]}`))
	}))
	defer srv.Close()

	p := New(WithAPIKey("key"), WithBaseURL(srv.URL))
	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}
	if len(models) != 1 || models[0].ID != "openai/gpt-4o-mini-transcribe" {
		t.Fatalf("unexpected models: %+v", models)
	}
}

func TestProvider_ListModels_ArrayResponse(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`[{"id":"openai/gpt-4o-mini-transcribe","architecture":{"input_modalities":["audio"],"output_modalities":["text"]}},{"id":"openai/gpt-4o-mini","architecture":{"input_modalities":["text"],"output_modalities":["text"]}}]`))
	}))
	defer srv.Close()

	p := New(WithAPIKey("key"), WithBaseURL(srv.URL))
	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}
	if len(models) != 1 || models[0].ID != "openai/gpt-4o-mini-transcribe" {
		t.Fatalf("unexpected models: %+v", models)
	}
}

func TestProvider_DoTranscribe(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("path = %s", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"hello from openrouter"}}]}`))
	}))
	defer srv.Close()

	p := New(WithAPIKey("key"), WithBaseURL(srv.URL))
	result, err := p.DoTranscribe(context.Background(), sdk.TranscriptionParams{
		Model:       p.TranscriptionModel("openai/gpt-4o-mini-transcribe"),
		Audio:       []byte("audio"),
		Filename:    "test.wav",
		ContentType: "audio/wav",
	})
	if err != nil {
		t.Fatalf("DoTranscribe: %v", err)
	}
	if result.Text != "hello from openrouter" {
		t.Fatalf("text = %q", result.Text)
	}
}
