package speech

import (
	"context"
	"encoding/base64"
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

func TestProvider_DoSynthesize(t *testing.T) {
	t.Parallel()

	audioPayload := base64.StdEncoding.EncodeToString([]byte("wav-bytes"))
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
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Fatalf("decode request body: %v", err)
		}
		if got := payload["model"]; got != defaultModelID {
			t.Fatalf("model = %#v, want %q", got, defaultModelID)
		}

		messages, ok := payload["messages"].([]any)
		if !ok || len(messages) != 2 {
			t.Fatalf("messages = %#v", payload["messages"])
		}
		audioCfg, ok := payload["audio"].(map[string]any)
		if !ok {
			t.Fatalf("audio config = %#v", payload["audio"])
		}
		if audioCfg["voice"] != "Chloe" || audioCfg["format"] != "wav" {
			t.Fatalf("audio config = %#v", audioCfg)
		}

		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"audio": map[string]any{"data": audioPayload},
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
	result, err := p.DoSynthesize(context.Background(), sdk.SpeechParams{
		Model: p.SpeechModel(""),
		Text:  "hello",
		Config: map[string]any{
			"style_prompt": "bright voice",
			"voice":        "Chloe",
			"format":       "wav",
		},
	})
	if err != nil {
		t.Fatalf("DoSynthesize: %v", err)
	}
	if result.ContentType != speechContentType {
		t.Fatalf("ContentType = %q, want %q", result.ContentType, speechContentType)
	}
	if string(result.Audio) != "wav-bytes" {
		t.Fatalf("audio = %q", string(result.Audio))
	}
}

func TestProvider_DoStream(t *testing.T) {
	t.Parallel()

	pcmChunk := base64.StdEncoding.EncodeToString([]byte{0x01, 0x02, 0x03, 0x04})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read request body: %v", err)
		}
		reqBody := string(body)
		if !strings.Contains(reqBody, `"stream":true`) {
			t.Fatalf("stream flag missing: %s", reqBody)
		}
		if !strings.Contains(reqBody, `"format":"pcm16"`) {
			t.Fatalf("pcm16 format missing: %s", reqBody)
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"audio\":{\"data\":\""+pcmChunk+"\"}}}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	p := New(
		WithAPIKey("test-key"),
		WithBaseURL(srv.URL+"/v1"),
		WithHTTPClient(srv.Client()),
	)
	result, err := p.DoStream(context.Background(), sdk.SpeechParams{
		Model: p.SpeechModel(""),
		Text:  "hello",
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}
	if result.ContentType != speechContentType {
		t.Fatalf("ContentType = %q, want %q", result.ContentType, speechContentType)
	}
	audio, err := result.Bytes()
	if err != nil {
		t.Fatalf("Bytes: %v", err)
	}
	if len(audio) < 4 || string(audio[:4]) != "RIFF" {
		t.Fatalf("expected WAV header, got %q", string(audio))
	}
}

func TestParseConfig(t *testing.T) {
	t.Parallel()

	cfg := parseConfig(map[string]any{
		"voice":        "Clara",
		"format":       "pcm16",
		"style_prompt": "warm tone",
	})
	if cfg.Voice != "Clara" {
		t.Fatalf("Voice = %q", cfg.Voice)
	}
	if cfg.Format != "pcm16" {
		t.Fatalf("Format = %q", cfg.Format)
	}
	if cfg.StylePrompt != "warm tone" {
		t.Fatalf("StylePrompt = %q", cfg.StylePrompt)
	}
}
