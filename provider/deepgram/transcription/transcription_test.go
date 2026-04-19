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
	p := New()
	models, err := p.ListModels(context.Background())
	if err == nil {
		t.Fatal("expected unsupported error")
	}
	if len(models) != 0 {
		t.Fatalf("len(models) = %d, want 0", len(models))
	}
}

func TestProvider_DoTranscribe(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/listen" {
			t.Fatalf("path = %s", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"metadata":{"duration":1.5},"results":{"channels":[{"detected_language":"en","alternatives":[{"transcript":"hello from deepgram","words":[{"word":"hello","start":0,"end":0.3,"speaker":0}]}]}]}}`))
	}))
	defer srv.Close()

	p := New(WithAPIKey("key"), WithBaseURL(srv.URL))
	result, err := p.DoTranscribe(context.Background(), sdk.TranscriptionParams{
		Model:       p.TranscriptionModel("nova-3"),
		Audio:       []byte("audio"),
		Filename:    "test.wav",
		ContentType: "audio/wav",
	})
	if err != nil {
		t.Fatalf("DoTranscribe: %v", err)
	}
	if result.Text != "hello from deepgram" {
		t.Fatalf("text = %q", result.Text)
	}
}
