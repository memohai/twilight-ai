package speech

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	sdk "github.com/memohai/twilight-ai/sdk"
)

// mockElevenLabsHandler returns an HTTP handler simulating the ElevenLabs TTS API.
// When streaming is true, the handler asserts that the path ends with "/stream".
func mockElevenLabsHandler(t *testing.T, streaming bool) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.Header.Get("xi-api-key") == "" {
			http.Error(w, "missing api key", http.StatusUnauthorized)
			return
		}
		if !strings.HasPrefix(r.URL.Path, "/v1/text-to-speech/") {
			t.Errorf("unexpected path: %s", r.URL.Path)
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		if streaming && !strings.HasSuffix(r.URL.Path, "/stream") {
			t.Errorf("DoStream must use /stream endpoint, got %s", r.URL.Path)
		}
		if !streaming && strings.HasSuffix(r.URL.Path, "/stream") {
			t.Errorf("DoSynthesize must not use /stream endpoint, got %s", r.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "audio/mpeg")
		_, _ = w.Write([]byte("fake-elevenlabs-audio"))
	})
}

func TestProvider_DoSynthesize(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(mockElevenLabsHandler(t, false))
	defer srv.Close()

	p := New(WithAPIKey("test-key"), WithBaseURL(srv.URL))

	result, err := p.DoSynthesize(context.Background(), sdk.SpeechParams{
		Text: "Hello",
		Config: map[string]any{
			"voice_id": "21m00Tcm4TlvDq8ikWAM",
		},
	})
	if err != nil {
		t.Fatalf("DoSynthesize: %v", err)
	}
	if !bytes.Equal(result.Audio, []byte("fake-elevenlabs-audio")) {
		t.Errorf("audio = %q", string(result.Audio))
	}
	if result.ContentType != "audio/mpeg" {
		t.Errorf("content type = %q, want audio/mpeg", result.ContentType)
	}
}

func TestProvider_DoSynthesize_MissingVoiceID(t *testing.T) {
	t.Parallel()
	p := New(WithAPIKey("key"), WithBaseURL("http://localhost"))

	_, err := p.DoSynthesize(context.Background(), sdk.SpeechParams{Text: "test"})
	if err == nil {
		t.Fatal("expected error when voice_id is missing")
	}
}

func TestProvider_DoStream(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(mockElevenLabsHandler(t, true))
	defer srv.Close()

	p := New(WithAPIKey("test-key"), WithBaseURL(srv.URL))

	result, err := p.DoStream(context.Background(), sdk.SpeechParams{
		Text: "Hi",
		Config: map[string]any{
			"voice_id": "21m00Tcm4TlvDq8ikWAM",
		},
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}

	audio, err := result.Bytes()
	if err != nil {
		t.Fatalf("Bytes: %v", err)
	}
	if !bytes.Equal(audio, []byte("fake-elevenlabs-audio")) {
		t.Errorf("audio = %q", string(audio))
	}
}

func TestProvider_DoSynthesize_HTTPError(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"detail":"invalid api key"}`, http.StatusUnauthorized)
	}))
	defer srv.Close()

	p := New(WithAPIKey("bad"), WithBaseURL(srv.URL))
	_, err := p.DoSynthesize(context.Background(), sdk.SpeechParams{
		Text:   "test",
		Config: map[string]any{"voice_id": "somevoice"},
	})
	if err == nil {
		t.Fatal("expected error for 401")
	}
}

func TestProvider_SpeechModel(t *testing.T) {
	t.Parallel()
	p := New()
	m := p.SpeechModel("eleven_turbo_v2_5")
	if m.ID != "eleven_turbo_v2_5" {
		t.Errorf("ID = %q", m.ID)
	}
	m2 := p.SpeechModel("")
	if m2.ID != defaultModelLLM {
		t.Errorf("default ID = %q, want %q", m2.ID, defaultModelLLM)
	}
}

func TestProvider_ListModels(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Errorf("method = %s, want GET", r.Method)
		}
		if r.URL.Path != "/v1/models" {
			t.Errorf("path = %s, want /v1/models", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"models":[{"model_id":"eleven_multilingual_v2","can_do_text_to_speech":true},{"model_id":"eleven_v3","can_do_text_to_speech":true},{"model_id":"scribe_v1","can_do_text_to_speech":false}]}`))
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
	if models[0].ID != "eleven_multilingual_v2" || models[1].ID != "eleven_v3" {
		t.Fatalf("unexpected models: %q, %q", models[0].ID, models[1].ID)
	}
}

func TestParseConfig(t *testing.T) {
	t.Parallel()
	cfg := parseConfig(map[string]any{
		"voice_id":          "abc",
		"model_id":          "eleven_turbo_v2_5",
		"stability":         0.3,
		"similarity_boost":  0.9,
		"style":             0.4,
		"use_speaker_boost": false,
		"output_format":     "pcm_16000",
		"speed":             1.2,
	})
	if cfg.VoiceID != "abc" {
		t.Errorf("voice_id = %q", cfg.VoiceID)
	}
	if cfg.ModelID != "eleven_turbo_v2_5" {
		t.Errorf("model_id = %q", cfg.ModelID)
	}
	if cfg.Stability != 0.3 {
		t.Errorf("stability = %v", cfg.Stability)
	}
	if cfg.SimilarityBoost != 0.9 {
		t.Errorf("similarity_boost = %v", cfg.SimilarityBoost)
	}
	if cfg.Style != 0.4 {
		t.Errorf("style = %v", cfg.Style)
	}
	if cfg.UseSpeakerBoost {
		t.Error("use_speaker_boost = true, want false")
	}
	if cfg.OutputFormat != "pcm_16000" {
		t.Errorf("output_format = %q", cfg.OutputFormat)
	}
	if cfg.Speed != 1.2 {
		t.Errorf("speed = %v", cfg.Speed)
	}
}

func TestParseConfig_Defaults(t *testing.T) {
	t.Parallel()
	cfg := parseConfig(nil)
	if cfg.ModelID != defaultModelLLM {
		t.Errorf("model_id = %q, want %q", cfg.ModelID, defaultModelLLM)
	}
	if cfg.Stability != 0.5 {
		t.Errorf("stability = %v", cfg.Stability)
	}
	if cfg.SimilarityBoost != 0.75 {
		t.Errorf("similarity_boost = %v", cfg.SimilarityBoost)
	}
	if cfg.Style != 0.0 {
		t.Errorf("style default = %v, want 0.0", cfg.Style)
	}
	if !cfg.UseSpeakerBoost {
		t.Error("use_speaker_boost default = false, want true")
	}
}
