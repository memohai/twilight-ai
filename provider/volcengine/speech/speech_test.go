package speech

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	sdk "github.com/memohai/twilight-ai/sdk"
)

var fakeAudio = []byte("fake-volcengine-audio")

// mockSAMIHandler returns an HTTP handler simulating the SAMI /api/v1/invoke endpoint.
// It bypasses token validation since the test uses WithToken to inject a static token.
func mockSAMIHandler(t *testing.T) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/api/v1/invoke" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		if r.URL.Query().Get("namespace") != "TTS" {
			t.Errorf("expected namespace=TTS, got %q", r.URL.Query().Get("namespace"))
		}
		if r.URL.Query().Get("token") == "" {
			http.Error(w, "missing token", http.StatusUnauthorized)
			return
		}

		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}

		resp := map[string]any{
			"status_code": 20000000,
			"status_text": "ok",
			"task_id":     "test-task",
			"namespace":   "TTS",
			// JSON marshaling will base64-encode the []byte, matching the real API format
			"data": fakeAudio,
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})
}

func TestProvider_DoSynthesize(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(mockSAMIHandler(t))
	defer srv.Close()

	p := New(
		WithBaseURL(srv.URL),
		WithToken("test-token"),
		WithAppKey("test-appkey"),
	)

	result, err := p.DoSynthesize(context.Background(), sdk.SpeechParams{
		Text: "Hello",
		Config: map[string]any{
			"speaker": "zh_female_qingxin",
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
	srv := httptest.NewServer(mockSAMIHandler(t))
	defer srv.Close()

	p := New(
		WithBaseURL(srv.URL),
		WithToken("test-token"),
		WithAppKey("test-appkey"),
	)

	result, err := p.DoStream(context.Background(), sdk.SpeechParams{
		Text: "Hi",
		Config: map[string]any{
			"speaker": "zh_female_qingxin",
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

func TestProvider_DoSynthesize_HTTPError(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL), WithToken("bad"), WithAppKey("key"))
	_, err := p.DoSynthesize(context.Background(), sdk.SpeechParams{Text: "test"})
	if err == nil {
		t.Fatal("expected error for 401")
	}
}

func TestProvider_DoSynthesize_APIError(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]any{
			"status_code": 40400001,
			"status_text": "invalid speaker",
			"task_id":     "t1",
			"namespace":   "TTS",
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL), WithToken("ok"), WithAppKey("key"))
	_, err := p.DoSynthesize(context.Background(), sdk.SpeechParams{Text: "test"})
	if err == nil {
		t.Fatal("expected error for API error")
	}
}

func TestProvider_SpeechModel(t *testing.T) {
	t.Parallel()
	p := New()
	m := p.SpeechModel("sami-tts")
	if m.ID != "sami-tts" {
		t.Errorf("ID = %q", m.ID)
	}
	m2 := p.SpeechModel("")
	if m2.ID != defaultModelID {
		t.Errorf("default ID = %q, want %q", m2.ID, defaultModelID)
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
		"speaker":     "zh_female_qingxin",
		"encoding":    "wav",
		"sample_rate": 16000,
		"speech_rate": 50,
		"pitch_rate":  5,
	})
	if cfg.Speaker != "zh_female_qingxin" {
		t.Errorf("speaker = %q", cfg.Speaker)
	}
	if cfg.Encoding != "wav" {
		t.Errorf("encoding = %q", cfg.Encoding)
	}
	if cfg.SampleRate != 16000 {
		t.Errorf("sample_rate = %d", cfg.SampleRate)
	}
	if cfg.SpeechRate != 50 {
		t.Errorf("speech_rate = %d", cfg.SpeechRate)
	}
	if cfg.PitchRate != 5 {
		t.Errorf("pitch_rate = %d", cfg.PitchRate)
	}
}

func TestParseConfig_Defaults(t *testing.T) {
	t.Parallel()
	cfg := parseConfig(nil)
	if cfg.Encoding != defaultEncoding {
		t.Errorf("default encoding = %q, want %q", cfg.Encoding, defaultEncoding)
	}
	if cfg.SampleRate != defaultSampleRate {
		t.Errorf("default sample_rate = %d, want %d", cfg.SampleRate, defaultSampleRate)
	}
}

func TestContentTypeForEncoding(t *testing.T) {
	t.Parallel()
	cases := []struct{ enc, want string }{
		{"mp3", "audio/mpeg"},
		{"wav", "audio/wav"},
		{"aac", "audio/aac"},
		{"", "audio/mpeg"},
	}
	for _, tc := range cases {
		got := contentTypeForEncoding(tc.enc)
		if got != tc.want {
			t.Errorf("contentTypeForEncoding(%q) = %q, want %q", tc.enc, got, tc.want)
		}
	}
}
