package speech

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	sdk "github.com/memohai/twilight-ai/sdk"
)

// fakeAudio is the test audio payload; the mock server hex-encodes it in the JSON response.
var fakeAudio = []byte("fake-minimax-audio")

// mockMinimaxHandler returns an HTTP handler simulating the MiniMax /v1/t2a_v2 endpoint.
func mockMinimaxHandler(t *testing.T) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/v1/t2a_v2" {
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
		resp := map[string]any{
			"data": map[string]any{
				"audio": hex.EncodeToString(fakeAudio),
			},
			"base_resp": map[string]any{
				"status_code": 0,
				"status_msg":  "success",
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})
}

func TestProvider_DoSynthesize(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(mockMinimaxHandler(t))
	defer srv.Close()

	p := New(WithAPIKey("test-key"), WithBaseURL(srv.URL))

	result, err := p.DoSynthesize(context.Background(), sdk.SpeechParams{
		Text: "Hello world",
		Config: map[string]any{
			"voice_id": "English_expressive_narrator",
		},
	})
	if err != nil {
		t.Fatalf("DoSynthesize: %v", err)
	}
	if !bytes.Equal(result.Audio, fakeAudio) {
		t.Errorf("audio = %q, want %q", string(result.Audio), string(fakeAudio))
	}
	if result.ContentType != "audio/mpeg" {
		t.Errorf("content type = %q, want audio/mpeg", result.ContentType)
	}
}

func TestProvider_DoStream(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(mockMinimaxHandler(t))
	defer srv.Close()

	p := New(WithAPIKey("test-key"), WithBaseURL(srv.URL))

	result, err := p.DoStream(context.Background(), sdk.SpeechParams{
		Text: "Hi",
		Config: map[string]any{
			"voice_id": "English_expressive_narrator",
		},
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}

	audio, err := result.Bytes()
	if err != nil {
		t.Fatalf("Bytes: %v", err)
	}
	if !bytes.Equal(audio, fakeAudio) {
		t.Errorf("audio = %q, want %q", string(audio), string(fakeAudio))
	}
}

func TestProvider_DoSynthesize_APIError(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]any{
			"data": map[string]any{"audio": ""},
			"base_resp": map[string]any{
				"status_code": 1000,
				"status_msg":  "invalid api key",
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	p := New(WithAPIKey("bad"), WithBaseURL(srv.URL))
	_, err := p.DoSynthesize(context.Background(), sdk.SpeechParams{Text: "test"})
	if err == nil {
		t.Fatal("expected error for API error")
	}
}

func TestProvider_DoSynthesize_HTTPError(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
	}))
	defer srv.Close()

	p := New(WithAPIKey("bad"), WithBaseURL(srv.URL))
	_, err := p.DoSynthesize(context.Background(), sdk.SpeechParams{Text: "test"})
	if err == nil {
		t.Fatal("expected error for 401")
	}
}

func TestProvider_SpeechModel(t *testing.T) {
	t.Parallel()
	p := New()
	m := p.SpeechModel("speech-2.8")
	if m.ID != "speech-2.8" {
		t.Errorf("ID = %q", m.ID)
	}
	m2 := p.SpeechModel("")
	if m2.ID != defaultModel {
		t.Errorf("default ID = %q, want %q", m2.ID, defaultModel)
	}
}

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

func TestParseConfig(t *testing.T) {
	t.Parallel()
	cfg := parseConfig(map[string]any{
		"voice_id":      "Chinese_narrator",
		"model":         "speech-2.8",
		"speed":         1.2,
		"vol":           0.8,
		"pitch":         2,
		"output_format": "wav",
		"sample_rate":   16000,
	})
	if cfg.VoiceID != "Chinese_narrator" {
		t.Errorf("voice_id = %q", cfg.VoiceID)
	}
	if cfg.Model != "speech-2.8" {
		t.Errorf("model = %q", cfg.Model)
	}
	if cfg.Speed != 1.2 {
		t.Errorf("speed = %v", cfg.Speed)
	}
	if cfg.Pitch != 2 {
		t.Errorf("pitch = %d", cfg.Pitch)
	}
	if cfg.OutputFormat != "wav" {
		t.Errorf("output_format = %q", cfg.OutputFormat)
	}
	if cfg.SampleRate != 16000 {
		t.Errorf("sample_rate = %d", cfg.SampleRate)
	}
}

func TestParseConfig_Defaults(t *testing.T) {
	t.Parallel()
	cfg := parseConfig(nil)
	if cfg.VoiceID != defaultVoiceID {
		t.Errorf("default voice_id = %q, want %q", cfg.VoiceID, defaultVoiceID)
	}
	if cfg.Model != defaultModel {
		t.Errorf("default model = %q, want %q", cfg.Model, defaultModel)
	}
	if cfg.SampleRate != 32000 {
		t.Errorf("default sample_rate = %d, want 32000", cfg.SampleRate)
	}
}
