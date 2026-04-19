package speech

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gorilla/websocket"
	sdk "github.com/memohai/twilight-ai/sdk"
)

var wsUpgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

// mockDashScopeHandler simulates the DashScope CosyVoice WebSocket server.
//
// Protocol:
//  1. Client connects with Authorization header
//  2. Client sends run-task → server replies task-started
//  3. Client sends continue-task + finish-task → server sends binary audio + task-finished
func mockDashScopeHandler(t *testing.T) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth := r.Header.Get("Authorization")
		if auth == "" {
			http.Error(w, "missing auth", http.StatusUnauthorized)
			return
		}

		conn, err := wsUpgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("upgrade: %v", err)
			return
		}
		defer conn.Close()

		taskID := ""
		for {
			mt, data, err := conn.ReadMessage()
			if err != nil {
				return
			}
			if mt != websocket.TextMessage {
				continue
			}

			var msg struct {
				Header struct {
					Action string `json:"action"`
					TaskID string `json:"task_id"`
				} `json:"header"`
			}
			if err := json.Unmarshal(data, &msg); err != nil {
				continue
			}

			switch msg.Header.Action {
			case "run-task":
				taskID = msg.Header.TaskID
				reply, _ := json.Marshal(map[string]any{
					"header": map[string]any{
						"event":   "task-started",
						"task_id": taskID,
					},
					"payload": map[string]any{},
				})
				_ = conn.WriteMessage(websocket.TextMessage, reply)

			case "finish-task":
				// Send fake audio binary frame
				_ = conn.WriteMessage(websocket.BinaryMessage, []byte("fake-alibabacloud-audio"))
				// Send task-finished
				reply, _ := json.Marshal(map[string]any{
					"header": map[string]any{
						"event":   "task-finished",
						"task_id": taskID,
					},
					"payload": map[string]any{},
				})
				_ = conn.WriteMessage(websocket.TextMessage, reply)
				return
			}
		}
	})
}

func TestProvider_DoSynthesize(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(mockDashScopeHandler(t))
	defer srv.Close()

	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")
	p := New(WithAPIKey("test-key"), WithBaseURL(wsURL))

	result, err := p.DoSynthesize(context.Background(), sdk.SpeechParams{
		Text: "Hello",
		Config: map[string]any{
			"voice": "longanyang",
		},
	})
	if err != nil {
		t.Fatalf("DoSynthesize: %v", err)
	}
	if !bytes.Equal(result.Audio, []byte("fake-alibabacloud-audio")) {
		t.Errorf("audio = %q", string(result.Audio))
	}
	if result.ContentType != "audio/mpeg" {
		t.Errorf("content type = %q, want audio/mpeg", result.ContentType)
	}
}

func TestProvider_DoStream(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(mockDashScopeHandler(t))
	defer srv.Close()

	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")
	p := New(WithAPIKey("test-key"), WithBaseURL(wsURL))

	result, err := p.DoStream(context.Background(), sdk.SpeechParams{
		Text: "Hi",
		Config: map[string]any{
			"voice": "longanyang",
		},
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}

	audio, err := result.Bytes()
	if err != nil {
		t.Fatalf("Bytes: %v", err)
	}
	if !bytes.Equal(audio, []byte("fake-alibabacloud-audio")) {
		t.Errorf("audio = %q", string(audio))
	}
}

func TestProvider_DoSynthesize_ConnectionFailure(t *testing.T) {
	t.Parallel()
	p := New(WithAPIKey("key"), WithBaseURL("ws://127.0.0.1:0"))
	_, err := p.DoSynthesize(context.Background(), sdk.SpeechParams{Text: "x"})
	if err == nil {
		t.Fatal("expected error when connection fails")
	}
}

func TestProvider_SpeechModel(t *testing.T) {
	t.Parallel()
	p := New()
	m := p.SpeechModel("cosyvoice-v2")
	if m.ID != "cosyvoice-v2" {
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
		"model":       "cosyvoice-v2",
		"voice":       "longanyang",
		"format":      "wav",
		"sample_rate": 16000,
		"volume":      80,
		"rate":        1.5,
		"pitch":       1.2,
	})
	if cfg.Model != "cosyvoice-v2" {
		t.Errorf("model = %q", cfg.Model)
	}
	if cfg.Voice != "longanyang" {
		t.Errorf("voice = %q", cfg.Voice)
	}
	if cfg.Format != "wav" {
		t.Errorf("format = %q", cfg.Format)
	}
	if cfg.SampleRate != 16000 {
		t.Errorf("sample_rate = %d", cfg.SampleRate)
	}
	if cfg.Volume != 80 {
		t.Errorf("volume = %d", cfg.Volume)
	}
	if cfg.Rate != 1.5 {
		t.Errorf("rate = %v", cfg.Rate)
	}
	if cfg.Pitch != 1.2 {
		t.Errorf("pitch = %v", cfg.Pitch)
	}
}

func TestParseConfig_Defaults(t *testing.T) {
	t.Parallel()
	cfg := parseConfig(nil)
	if cfg.Model != defaultModel {
		t.Errorf("default model = %q, want %q", cfg.Model, defaultModel)
	}
	if cfg.Format != defaultFormat {
		t.Errorf("default format = %q, want %q", cfg.Format, defaultFormat)
	}
	if cfg.SampleRate != defaultSampleRate {
		t.Errorf("default sample_rate = %d, want %d", cfg.SampleRate, defaultSampleRate)
	}
}
