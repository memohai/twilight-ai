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
		if r.Method != http.MethodGet {
			t.Errorf("method = %s, want GET", r.Method)
		}
		if r.URL.Path != "/v1/models" {
			t.Errorf("path = %s, want /v1/models", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Token key" {
			t.Errorf("Authorization = %q", r.Header.Get("Authorization"))
		}
		_, _ = w.Write([]byte(`{"stt":[{"canonical_name":"nova-3-general"},{"canonical_name":"flux-general-en"},{"uuid":"not-a-model-id"}],"tts":[{"canonical_name":"aura-2-asteria-en"}]}`))
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
	if models[0].ID != "nova-3-general" || models[1].ID != "flux-general-en" {
		t.Fatalf("unexpected models: %+v", models)
	}
}

func TestProvider_ListModels_HTTPError(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
	}))
	defer srv.Close()

	p := New(WithAPIKey("bad"), WithBaseURL(srv.URL))
	models, err := p.ListModels(context.Background())
	if err == nil {
		t.Fatal("expected error for 401")
	}
	if models != nil {
		t.Fatalf("models = %+v, want nil", models)
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
