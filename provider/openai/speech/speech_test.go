package speech

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	sdk "github.com/memohai/twilight-ai/sdk"
)

// mockOpenAITTSHandler returns an HTTP handler that simulates the /audio/speech endpoint.
func mockOpenAITTSHandler(t *testing.T) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/audio/speech" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		if r.Header.Get("Authorization") == "" {
			http.Error(w, "missing auth", http.StatusUnauthorized)
			return
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "audio/mpeg")
		_, _ = w.Write([]byte("fake-mp3-audio-data"))
	})
}

func TestProvider_DoSynthesize(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(mockOpenAITTSHandler(t))
	defer srv.Close()

	p := New(WithAPIKey("test-key"), WithBaseURL(srv.URL))

	result, err := p.DoSynthesize(context.Background(), sdk.SpeechParams{
		Text: "Hello world",
		Config: map[string]any{
			"voice":           "alloy",
			"response_format": "mp3",
		},
	})
	if err != nil {
		t.Fatalf("DoSynthesize: %v", err)
	}
	if !bytes.Equal(result.Audio, []byte("fake-mp3-audio-data")) {
		t.Errorf("audio = %q", string(result.Audio))
	}
	if result.ContentType != "audio/mpeg" {
		t.Errorf("content type = %q, want audio/mpeg", result.ContentType)
	}
}

func TestProvider_DoStream(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(mockOpenAITTSHandler(t))
	defer srv.Close()

	p := New(WithAPIKey("test-key"), WithBaseURL(srv.URL))

	result, err := p.DoStream(context.Background(), sdk.SpeechParams{
		Text: "Hi",
		Config: map[string]any{
			"voice": "nova",
		},
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}

	audio, err := result.Bytes()
	if err != nil {
		t.Fatalf("Bytes: %v", err)
	}
	if !bytes.Equal(audio, []byte("fake-mp3-audio-data")) {
		t.Errorf("audio = %q", string(audio))
	}
}

func TestProvider_DoSynthesize_HTTPError(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"invalid_api_key"}`, http.StatusUnauthorized)
	}))
	defer srv.Close()

	p := New(WithAPIKey("bad-key"), WithBaseURL(srv.URL))
	_, err := p.DoSynthesize(context.Background(), sdk.SpeechParams{Text: "test"})
	if err == nil {
		t.Fatal("expected error for 401")
	}
}

func TestProvider_DoSynthesize_ConnectionFailure(t *testing.T) {
	t.Parallel()
	p := New(WithAPIKey("key"), WithBaseURL("http://127.0.0.1:0"))
	_, err := p.DoSynthesize(context.Background(), sdk.SpeechParams{Text: "x"})
	if err == nil {
		t.Fatal("expected error when connection fails")
	}
}

func TestProvider_SpeechModel(t *testing.T) {
	t.Parallel()
	p := New()

	m := p.SpeechModel("tts-1-hd")
	if m.ID != "tts-1-hd" {
		t.Errorf("ID = %q, want tts-1-hd", m.ID)
	}
	if m.Provider != p {
		t.Error("provider mismatch")
	}

	m2 := p.SpeechModel("")
	if m2.ID != defaultModelID {
		t.Errorf("default ID = %q, want %q", m2.ID, defaultModelID)
	}
}

func TestProvider_ListModels(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Errorf("method = %s, want GET", r.Method)
		}
		if r.URL.Path != "/models" {
			t.Errorf("path = %s, want /models", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"id":"gpt-4o-mini-tts"},{"id":"tts-1"},{"id":"gpt-4.1"}]}`))
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
	if models[0].ID != "gpt-4o-mini-tts" || models[1].ID != "tts-1" {
		t.Fatalf("unexpected models: %q, %q", models[0].ID, models[1].ID)
	}
}

func TestProvider_ListModels_ArrayResponse(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[{"id":"gpt-4o-mini-tts"},{"id":"tts-1"},{"id":"gpt-4.1"}]`))
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

func TestParseConfig(t *testing.T) {
	t.Parallel()
	cfg := parseConfig(map[string]any{
		"voice":           "shimmer",
		"response_format": "opus",
		"speed":           1.5,
		"instructions":    "speak slowly",
	})
	if cfg.Voice != "shimmer" {
		t.Errorf("voice = %q", cfg.Voice)
	}
	if cfg.ResponseFormat != "opus" {
		t.Errorf("format = %q", cfg.ResponseFormat)
	}
	if cfg.Speed != 1.5 {
		t.Errorf("speed = %v", cfg.Speed)
	}
	if cfg.Instructions != "speak slowly" {
		t.Errorf("instructions = %q", cfg.Instructions)
	}
}

func TestParseConfig_Defaults(t *testing.T) {
	t.Parallel()
	cfg := parseConfig(nil)
	if cfg.Voice != defaultVoice {
		t.Errorf("default voice = %q, want %q", cfg.Voice, defaultVoice)
	}
	if cfg.ResponseFormat != defaultFormat {
		t.Errorf("default format = %q, want %q", cfg.ResponseFormat, defaultFormat)
	}
}

func TestContentTypeForFormat(t *testing.T) {
	t.Parallel()
	// Supported formats: mp3, opus, pcm, wav (aac/flac are not part of the OpenAI TTS API).
	cases := []struct{ format, want string }{
		{"mp3", "audio/mpeg"},
		{"opus", "audio/opus"},
		{"wav", "audio/wav"},
		{"pcm", "audio/pcm"},
		{"", "audio/mpeg"},
		{"unknown", "audio/mpeg"},
	}
	for _, tc := range cases {
		got := contentTypeForFormat(tc.format)
		if got != tc.want {
			t.Errorf("contentTypeForFormat(%q) = %q, want %q", tc.format, got, tc.want)
		}
	}
}

func TestProvider_DoSynthesize_WithModel(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &req)
		if req["model"] != "tts-1-hd" {
			t.Errorf("model = %v, want tts-1-hd", req["model"])
		}
		w.Header().Set("Content-Type", "audio/mpeg")
		_, _ = w.Write([]byte("audio"))
	}))
	defer srv.Close()

	p := New(WithAPIKey("key"), WithBaseURL(srv.URL))
	model := p.SpeechModel("tts-1-hd")

	_, err := p.DoSynthesize(context.Background(), sdk.SpeechParams{
		Model: model,
		Text:  "test",
	})
	if err != nil {
		t.Fatalf("DoSynthesize: %v", err)
	}
}
