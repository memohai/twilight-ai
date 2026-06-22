package videos

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/memohai/twilight-ai/sdk"
)

func TestDoCreateBuildsArkContent(t *testing.T) {
	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/contents/generations/tasks" {
			t.Fatalf("path = %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer ark-key" {
			t.Fatalf("missing auth header")
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"id": "task-1", "status": "queued"})
	}))
	defer server.Close()

	prov := New(WithAPIKey("ark-key"), WithBaseURL(server.URL))
	_, err := prov.DoCreate(context.Background(), sdk.VideoParams{
		Model:           prov.VideoModel("doubao-seedance-2-0-260128"),
		Prompt:          "city at night",
		DurationSeconds: intPtr(8),
		Resolution:      "720p",
		AspectRatio:     "16:9",
		GenerateAudio:   boolPtr(true),
		Seed:            int64Ptr(9),
		CallbackURL:     "https://example.com/hook",
		InputImage:      &sdk.MediaInput{URL: "https://example.com/first.png"},
		InputVideo:      &sdk.MediaInput{URL: "https://example.com/source.mp4"},
		ReferenceImages: []sdk.MediaInput{{URL: "https://example.com/ref.png"}},
		ReferenceVideos: []sdk.MediaInput{{URL: "https://example.com/ref.mp4"}},
		ReferenceAudio:  []sdk.MediaInput{{URL: "https://example.com/ref.mp3"}},
		Config:          map[string]any{"watermark": false},
	})
	if err != nil {
		t.Fatalf("DoCreate returned error: %v", err)
	}

	if got["model"] != "doubao-seedance-2-0-260128" || got["duration"].(float64) != 8 {
		t.Fatalf("unexpected top-level body: %#v", got)
	}
	if got["resolution"] != "720p" || got["ratio"] != "16:9" || got["callback_url"] != "https://example.com/hook" {
		t.Fatalf("unexpected mapped fields: %#v", got)
	}
	if got["watermark"] != false {
		t.Fatalf("config passthrough missing: %#v", got)
	}
	content := got["content"].([]any)
	if len(content) != 6 {
		t.Fatalf("content len = %d, want 6: %#v", len(content), content)
	}
	text := content[0].(map[string]any)
	if text["type"] != "text" || text["text"] != "city at night" {
		t.Fatalf("unexpected text item: %#v", text)
	}
	image := content[1].(map[string]any)
	if image["type"] != "image_url" || image["role"] != "first_frame" {
		t.Fatalf("unexpected image item: %#v", image)
	}
}

func TestDoGetExtractsVideoURL(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/contents/generations/tasks/task-1" {
			t.Fatalf("path = %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":     "task-1",
			"model":  "doubao-seedance-2-0-260128",
			"status": "succeeded",
			"content": map[string]any{
				"video_url": "https://example.com/out.mp4",
			},
		})
	}))
	defer server.Close()

	prov := New(WithAPIKey("ark-key"), WithBaseURL(server.URL))
	job, err := prov.DoGet(context.Background(), prov.VideoModel("doubao-seedance-2-0-260128"), "task-1")
	if err != nil {
		t.Fatalf("DoGet returned error: %v", err)
	}
	if job.Status != sdk.VideoJobSucceeded {
		t.Fatalf("status = %s", job.Status)
	}
	if len(job.Outputs) != 1 || job.Outputs[0].URL != "https://example.com/out.mp4" {
		t.Fatalf("unexpected outputs: %#v", job.Outputs)
	}
}

func TestDoCancelUsesDeleteTaskEndpoint(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		if r.Method != http.MethodDelete {
			t.Fatalf("method = %s", r.Method)
		}
		if r.URL.Path != "/contents/generations/tasks/task-1" {
			t.Fatalf("path = %s", r.URL.Path)
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	prov := New(WithAPIKey("ark-key"), WithBaseURL(server.URL))
	if err := prov.DoCancel(context.Background(), prov.VideoModel("doubao-seedance-2-0-260128"), "task-1"); err != nil {
		t.Fatalf("DoCancel returned error: %v", err)
	}
	if !called {
		t.Fatalf("server was not called")
	}
}

func intPtr(v int) *int       { return &v }
func int64Ptr(v int64) *int64 { return &v }
func boolPtr(v bool) *bool    { return &v }
