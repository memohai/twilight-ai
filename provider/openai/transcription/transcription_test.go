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
		if r.URL.Path != "/models" {
			t.Fatalf("path = %s", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"data":[{"id":"gpt-4o-mini-transcribe"},{"id":"whisper-1"},{"id":"gpt-4o-mini"}]}`))
	}))
	defer srv.Close()

	p := New(WithAPIKey("key"), WithBaseURL(srv.URL))
	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("len(models) = %d, want 2", len(models))
	}
}

func TestProvider_DoTranscribe(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/audio/transcriptions" {
			t.Fatalf("path = %s", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"text":"hello world","language":"en"}`))
	}))
	defer srv.Close()

	p := New(WithAPIKey("key"), WithBaseURL(srv.URL))
	result, err := p.DoTranscribe(context.Background(), sdk.TranscriptionParams{
		Model:    p.TranscriptionModel("gpt-4o-mini-transcribe"),
		Audio:    []byte("audio"),
		Filename: "test.wav",
	})
	if err != nil {
		t.Fatalf("DoTranscribe: %v", err)
	}
	if result.Text != "hello world" {
		t.Fatalf("text = %q", result.Text)
	}
}
