package transcription

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	sdk "github.com/memohai/twilight-ai/sdk"
)

func TestProvider_ListModels(t *testing.T) {
	t.Parallel()

	p := New()
	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}
	if len(models) != 1 {
		t.Fatalf("len(models) = %d, want 1", len(models))
	}
	if models[0].ID != defaultModelID {
		t.Fatalf("models[0].ID = %q, want %q", models[0].ID, defaultModelID)
	}
}

func TestProvider_DoTranscribe(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("path = %s, want /v1/chat/completions", r.URL.Path)
		}
		if got := r.Header.Get("api-key"); got != "test-key" {
			t.Fatalf("api-key = %q", got)
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read request body: %v", err)
		}
		reqBody := string(body)
		if !strings.Contains(reqBody, `"language":"zh"`) {
			t.Fatalf("language missing from request: %s", reqBody)
		}
		if !strings.Contains(reqBody, `data:audio/wav;base64,`) {
			t.Fatalf("audio data URL missing from request: %s", reqBody)
		}

		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"content": "hello world",
					},
				},
			},
		})
	}))
	defer srv.Close()

	p := New(
		WithAPIKey("test-key"),
		WithBaseURL(srv.URL+"/v1"),
		WithHTTPClient(srv.Client()),
	)
	result, err := p.DoTranscribe(context.Background(), sdk.TranscriptionParams{
		Model:       p.TranscriptionModel(""),
		Audio:       []byte("audio-bytes"),
		Filename:    "sample.wav",
		ContentType: "audio/wav",
		Config: map[string]any{
			"language": "zh",
		},
	})
	if err != nil {
		t.Fatalf("DoTranscribe: %v", err)
	}
	if result.Text != "hello world" {
		t.Fatalf("Text = %q, want %q", result.Text, "hello world")
	}
}

func TestDecodeResponse_UsesAudioTranscriptFallback(t *testing.T) {
	t.Parallel()

	result, err := decodeResponse([]byte(`{"choices":[{"message":{"audio":{"transcript":"transcript from audio"}}}]}`))
	if err != nil {
		t.Fatalf("decodeResponse: %v", err)
	}
	if result.Text != "transcript from audio" {
		t.Fatalf("Text = %q", result.Text)
	}
}

func TestBuildAudioDataURL_DefaultsToWaveByExtension(t *testing.T) {
	t.Parallel()

	dataURL := buildAudioDataURL([]byte("abc"), "", "clip.wav")
	if !strings.HasPrefix(dataURL, "data:audio/wav;base64,") {
		t.Fatalf("dataURL = %q", dataURL)
	}
}
