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

// mockDeepgramHandler returns an HTTP handler simulating the Deepgram /v1/speak endpoint.
func mockDeepgramHandler(t *testing.T) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/v1/speak" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		auth := r.Header.Get("Authorization")
		if auth == "" || len(auth) < 6 || auth[:6] != "Token " {
			http.Error(w, "missing token auth", http.StatusUnauthorized)
			return
		}
		var body map[string]string
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body["text"] == "" {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "audio/mpeg")
		_, _ = w.Write([]byte("fake-deepgram-audio"))
	})
}

func TestProvider_DoSynthesize(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(mockDeepgramHandler(t))
	defer srv.Close()

	p := New(WithAPIKey("test-key"), WithBaseURL(srv.URL))

	result, err := p.DoSynthesize(context.Background(), sdk.SpeechParams{
		Text: "Hello world",
		Config: map[string]any{
			"model": "aura-2-asteria-en",
		},
	})
	if err != nil {
		t.Fatalf("DoSynthesize: %v", err)
	}
	if !bytes.Equal(result.Audio, []byte("fake-deepgram-audio")) {
		t.Errorf("audio = %q", string(result.Audio))
	}
}

func TestProvider_DoStream(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(mockDeepgramHandler(t))
	defer srv.Close()

	p := New(WithAPIKey("test-key"), WithBaseURL(srv.URL))

	result, err := p.DoStream(context.Background(), sdk.SpeechParams{
		Text: "Hi",
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}

	audio, err := result.Bytes()
	if err != nil {
		t.Fatalf("Bytes: %v", err)
	}
	if !bytes.Equal(audio, []byte("fake-deepgram-audio")) {
		t.Errorf("audio = %q", string(audio))
	}
}

func TestProvider_DoSynthesize_HTTPError(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"err_code":"INVALID_AUTH"}`, http.StatusUnauthorized)
	}))
	defer srv.Close()

	p := New(WithAPIKey("bad"), WithBaseURL(srv.URL))
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
	m := p.SpeechModel("aura-2-orpheus-en")
	if m.ID != "aura-2-orpheus-en" {
		t.Errorf("ID = %q", m.ID)
	}
	m2 := p.SpeechModel("")
	if m2.ID != defaultVoiceModel {
		t.Errorf("default ID = %q, want %q", m2.ID, defaultVoiceModel)
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
		"model":       "aura-orpheus-en",
		"encoding":    "linear16",
		"sample_rate": 16000,
		"container":   "wav",
	})
	if cfg.Model != "aura-orpheus-en" {
		t.Errorf("model = %q", cfg.Model)
	}
	if cfg.Encoding != "linear16" {
		t.Errorf("encoding = %q", cfg.Encoding)
	}
	if cfg.SampleRate != 16000 {
		t.Errorf("sample_rate = %d", cfg.SampleRate)
	}
	if cfg.Container != "wav" {
		t.Errorf("container = %q", cfg.Container)
	}
}

func TestParseConfig_Defaults(t *testing.T) {
	t.Parallel()
	cfg := parseConfig(nil)
	if cfg.Model != defaultVoiceModel {
		t.Errorf("default model = %q, want %q", cfg.Model, defaultVoiceModel)
	}
}

func TestContentTypeForEncoding(t *testing.T) {
	t.Parallel()
	cases := []struct{ enc, container, want string }{
		{"", "wav", "audio/wav"},
		{"linear16", "", "audio/l16"},
		{"mulaw", "", "audio/basic"},
		{"", "", "audio/mpeg"},
	}
	for _, tc := range cases {
		got := contentTypeForEncoding(tc.enc, tc.container)
		if got != tc.want {
			t.Errorf("contentTypeForEncoding(%q, %q) = %q, want %q", tc.enc, tc.container, got, tc.want)
		}
	}
}
