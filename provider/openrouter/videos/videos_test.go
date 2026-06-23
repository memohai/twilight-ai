package videos

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/memohai/twilight-ai/sdk"
)

func TestDoCreateMapsVideoRequest(t *testing.T) {
	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/videos" {
			t.Fatalf("path = %s, want /v1/videos", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Fatalf("missing auth header")
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		_ = json.NewEncoder(w).Encode(videoResponse{ID: "job-1", Status: "pending", GenerationID: "gen-1"})
	}))
	defer server.Close()

	prov := New(WithAPIKey("test-key"), WithBaseURL(server.URL))
	model := prov.VideoModel("google/veo-3.1")
	_, err := prov.DoCreate(context.Background(), sdk.VideoParams{
		Model:           model,
		Prompt:          "cinematic ocean",
		DurationSeconds: intPtr(8),
		Resolution:      "720p",
		AspectRatio:     "16:9",
		Size:            "1280x720",
		Seed:            int64Ptr(42),
		GenerateAudio:   boolPtr(true),
		CallbackURL:     "https://example.com/hook",
		InputImage:      &sdk.MediaInput{URL: "https://example.com/first.png"},
		ReferenceImages: []sdk.MediaInput{{URL: "https://example.com/ref.png"}},
		ReferenceAudio:  []sdk.MediaInput{{URL: "https://example.com/ref.mp3"}},
		ReferenceVideos: []sdk.MediaInput{{URL: "https://example.com/ref.mp4"}},
		Config:          map[string]any{"provider": map[string]any{"sort": "price"}},
	})
	if err != nil {
		t.Fatalf("DoCreate returned error: %v", err)
	}

	if got["model"] != "google/veo-3.1" || got["prompt"] != "cinematic ocean" {
		t.Fatalf("unexpected core fields: %#v", got)
	}
	if got["duration"].(float64) != 8 || got["resolution"] != "720p" || got["aspect_ratio"] != "16:9" {
		t.Fatalf("unexpected video fields: %#v", got)
	}
	frameImages := got["frame_images"].([]any)
	frame := frameImages[0].(map[string]any)
	if frame["frame_type"] != "first_frame" {
		t.Fatalf("frame_type = %#v", frame["frame_type"])
	}
	refs := got["input_references"].([]any)
	if len(refs) != 3 {
		t.Fatalf("input refs len = %d, want 3", len(refs))
	}
	if got["provider"].(map[string]any)["sort"] != "price" {
		t.Fatalf("provider passthrough missing: %#v", got["provider"])
	}
}

func TestDoGetMapsCompletedResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/videos/job-1" {
			t.Fatalf("path = %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(videoResponse{
			ID:           "job-1",
			Status:       "completed",
			GenerationID: "gen-1",
			UnsignedURLs: []string{"https://storage.example.com/out.mp4"},
			Usage:        map[string]any{"cost": 0.5},
		})
	}))
	defer server.Close()

	prov := New(WithAPIKey("test-key"), WithBaseURL(server.URL))
	job, err := prov.DoGet(context.Background(), prov.VideoModel("google/veo-3.1"), "job-1")
	if err != nil {
		t.Fatalf("DoGet returned error: %v", err)
	}
	if job.Status != sdk.VideoJobSucceeded {
		t.Fatalf("status = %s", job.Status)
	}
	if len(job.Outputs) != 1 || job.Outputs[0].URL != "https://storage.example.com/out.mp4" {
		t.Fatalf("unexpected outputs: %#v", job.Outputs)
	}
}

func TestListModelsMapsCapabilities(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/videos/models" {
			t.Fatalf("path = %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(listModelsResponse{Data: []modelResponse{{
			ID:                    "google/veo-3.1",
			Name:                  "Veo 3.1",
			SupportedDurations:    []int{5, 8},
			SupportedResolutions:  []string{"720p"},
			SupportedAspectRatios: []string{"16:9"},
		}}})
	}))
	defer server.Close()

	prov := New(WithAPIKey("test-key"), WithBaseURL(server.URL))
	models, err := prov.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}
	if len(models) != 1 || models[0].ID != "google/veo-3.1" {
		t.Fatalf("unexpected models: %#v", models)
	}
	if models[0].ProviderMetadata["name"] != "Veo 3.1" {
		t.Fatalf("metadata missing: %#v", models[0].ProviderMetadata)
	}
}

func intPtr(v int) *int       { return &v }
func int64Ptr(v int64) *int64 { return &v }
func boolPtr(v bool) *bool    { return &v }
