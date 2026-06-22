package videos

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/memohai/twilight-ai/sdk"
)

func TestDoCreateJSONMapsInputReference(t *testing.T) {
	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/videos" {
			t.Fatalf("path = %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer openai-key" {
			t.Fatalf("missing auth header")
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		_ = json.NewEncoder(w).Encode(videoResponse{ID: "video-1", Status: "queued", Model: "sora-2"})
	}))
	defer server.Close()

	prov := New(WithAPIKey("openai-key"), WithBaseURL(server.URL))
	_, err := prov.DoCreate(context.Background(), sdk.VideoParams{
		Model:           prov.VideoModel("sora-2"),
		Prompt:          "sparkling letters",
		Size:            "1280x720",
		DurationSeconds: intPtr(8),
		InputImage:      &sdk.MediaInput{URL: "https://example.com/frame.png"},
	})
	if err != nil {
		t.Fatalf("DoCreate returned error: %v", err)
	}
	if got["model"] != "sora-2" || got["prompt"] != "sparkling letters" {
		t.Fatalf("unexpected body: %#v", got)
	}
	if got["seconds"].(float64) != 8 || got["size"] != "1280x720" {
		t.Fatalf("unexpected size/duration: %#v", got)
	}
	ref := got["input_reference"].(map[string]any)
	if ref["image_url"] != "https://example.com/frame.png" {
		t.Fatalf("unexpected input_reference: %#v", ref)
	}
}

func TestDoCreateMultipartUploadsInputReference(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.Header.Get("Content-Type"), "multipart/form-data") {
			t.Fatalf("content type = %s", r.Header.Get("Content-Type"))
		}
		if err := r.ParseMultipartForm(1024); err != nil {
			t.Fatalf("parse multipart: %v", err)
		}
		if r.FormValue("model") != "sora-2-pro" || r.FormValue("seconds") != "4" {
			t.Fatalf("unexpected form values: %#v", r.MultipartForm.Value)
		}
		file, header, err := r.FormFile("input_reference")
		if err != nil {
			t.Fatalf("missing input_reference file: %v", err)
		}
		defer file.Close()
		data, _ := io.ReadAll(file)
		if string(data) != "png-bytes" || header.Filename != "frame.png" {
			t.Fatalf("unexpected file: %q %s", data, header.Filename)
		}
		_ = json.NewEncoder(w).Encode(videoResponse{ID: "video-1", Status: "completed", Model: "sora-2-pro"})
	}))
	defer server.Close()

	prov := New(WithAPIKey("openai-key"), WithBaseURL(server.URL))
	job, err := prov.DoCreate(context.Background(), sdk.VideoParams{
		Model:           prov.VideoModel("sora-2-pro"),
		Prompt:          "turn around",
		DurationSeconds: intPtr(4),
		InputImage:      &sdk.MediaInput{Data: []byte("png-bytes"), Filename: "frame.png", ContentType: "image/png"},
	})
	if err != nil {
		t.Fatalf("DoCreate returned error: %v", err)
	}
	if job.Status != sdk.VideoJobSucceeded || len(job.Outputs) != 1 {
		t.Fatalf("unexpected job: %#v", job)
	}
}

func TestDoGetAndDownloadContentVariant(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/videos/video-1":
			_ = json.NewEncoder(w).Encode(videoResponse{ID: "video-1", Status: "completed", Model: "sora-2"})
		case "/videos/video-1/content":
			if r.URL.Query().Get("variant") != "thumbnail" {
				t.Fatalf("variant = %q", r.URL.Query().Get("variant"))
			}
			w.Header().Set("Content-Type", "image/webp")
			_, _ = w.Write([]byte("thumb"))
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer server.Close()

	prov := New(WithAPIKey("openai-key"), WithBaseURL(server.URL))
	job, err := prov.DoGet(context.Background(), prov.VideoModel("sora-2"), "video-1")
	if err != nil {
		t.Fatalf("DoGet returned error: %v", err)
	}
	if job.Status != sdk.VideoJobSucceeded || len(job.Outputs) != 1 {
		t.Fatalf("unexpected job: %#v", job)
	}
	job.Outputs[0].ProviderMetadata["variant"] = "thumbnail"
	data, contentType, err := prov.DoDownload(context.Background(), prov.VideoModel("sora-2"), job.Outputs[0])
	if err != nil {
		t.Fatalf("DoDownload returned error: %v", err)
	}
	if string(data) != "thumb" || contentType != "image/webp" {
		t.Fatalf("unexpected download: %q %s", data, contentType)
	}
}

func intPtr(v int) *int { return &v }
