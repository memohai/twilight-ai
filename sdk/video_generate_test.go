package sdk

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestCreateVideoValidation(t *testing.T) {
	_, err := CreateVideo(context.Background(), WithVideoPrompt("hello"))
	if err == nil || !strings.Contains(err.Error(), "video model is required") {
		t.Fatalf("expected missing model error, got %v", err)
	}

	_, err = CreateVideo(context.Background(), WithVideoModel(&VideoModel{ID: "m"}), WithVideoPrompt("hello"))
	if err == nil || !strings.Contains(err.Error(), "has no provider") {
		t.Fatalf("expected missing provider error, got %v", err)
	}

	_, err = CreateVideo(context.Background(), WithVideoModel(testVideoModel(&fakeVideoProvider{})))
	if err == nil || !strings.Contains(err.Error(), "prompt is required") {
		t.Fatalf("expected missing prompt error, got %v", err)
	}
}

func TestGenerateVideoPollsUntilSucceeded(t *testing.T) {
	prov := &fakeVideoProvider{
		createJob: &VideoJob{ID: "job-1", Status: VideoJobQueued},
		getJobs: []*VideoJob{
			{ID: "job-1", Status: VideoJobRunning},
			{ID: "job-1", Status: VideoJobSucceeded, Outputs: []VideoOutput{{URL: "https://example.com/out.mp4"}}},
		},
		downloadData: []byte("video"),
	}

	result, err := GenerateVideo(context.Background(),
		WithVideoModel(testVideoModel(prov)),
		WithVideoPrompt("make a clip"),
		WithVideoPollInterval(time.Millisecond),
		WithVideoPollTimeout(time.Second),
		WithVideoDownload(true),
	)
	if err != nil {
		t.Fatalf("GenerateVideo returned error: %v", err)
	}
	if result.Job.Status != VideoJobSucceeded {
		t.Fatalf("status = %s, want succeeded", result.Job.Status)
	}
	if prov.getCalls != 2 {
		t.Fatalf("get calls = %d, want 2", prov.getCalls)
	}
	if string(result.Data) != "video" || result.ContentType != "video/mp4" {
		t.Fatalf("unexpected download result: %q %q", result.Data, result.ContentType)
	}
}

func TestGenerateVideoTimeout(t *testing.T) {
	prov := &fakeVideoProvider{
		createJob: &VideoJob{ID: "job-1", Status: VideoJobQueued},
		getJobs:   []*VideoJob{{ID: "job-1", Status: VideoJobRunning}},
	}

	_, err := GenerateVideo(context.Background(),
		WithVideoModel(testVideoModel(prov)),
		WithVideoPrompt("make a clip"),
		WithVideoPollInterval(time.Millisecond),
		WithVideoPollTimeout(3*time.Millisecond),
	)
	if err == nil || !strings.Contains(err.Error(), "timed out") {
		t.Fatalf("expected timeout error, got %v", err)
	}
}

func TestGenerateVideoFailedStatus(t *testing.T) {
	prov := &fakeVideoProvider{
		createJob: &VideoJob{ID: "job-1", Status: VideoJobQueued},
		getJobs: []*VideoJob{
			{ID: "job-1", Status: VideoJobFailed, Error: &VideoError{Message: "blocked"}},
		},
	}

	result, err := GenerateVideo(context.Background(),
		WithVideoModel(testVideoModel(prov)),
		WithVideoPrompt("make a clip"),
		WithVideoPollInterval(time.Millisecond),
		WithVideoPollTimeout(time.Second),
	)
	if err == nil || !strings.Contains(err.Error(), "blocked") {
		t.Fatalf("expected failed status error, got %v", err)
	}
	if result == nil || result.Job.Status != VideoJobFailed {
		t.Fatalf("expected failed result, got %#v", result)
	}
}

func testVideoModel(prov VideoProvider) *VideoModel {
	return &VideoModel{ID: "model-1", Provider: prov}
}

type fakeVideoProvider struct {
	createJob    *VideoJob
	getJobs      []*VideoJob
	getCalls     int
	downloadData []byte
}

func (p *fakeVideoProvider) ListModels(context.Context) ([]*VideoModel, error) {
	return nil, nil
}

func (p *fakeVideoProvider) DoCreate(context.Context, VideoParams) (*VideoJob, error) {
	return p.createJob, nil
}

func (p *fakeVideoProvider) DoGet(context.Context, *VideoModel, string) (*VideoJob, error) {
	if len(p.getJobs) == 0 {
		return nil, errors.New("no jobs configured")
	}
	idx := p.getCalls
	if idx >= len(p.getJobs) {
		idx = len(p.getJobs) - 1
	}
	p.getCalls++
	return p.getJobs[idx], nil
}

func (p *fakeVideoProvider) DoCancel(context.Context, *VideoModel, string) error {
	return nil
}

func (p *fakeVideoProvider) DoDownload(context.Context, *VideoModel, VideoOutput) ([]byte, string, error) {
	return p.downloadData, "video/mp4", nil
}
