//go:build integration

package speech

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	sdk "github.com/memohai/twilight-ai/sdk"
)

// Real Edge TTS integration tests. Not compiled by default (requires -tags=integration).
// Requires network access to speech.platform.bing.com.
//
// Run:
//
//	go test -tags=integration ./provider/edge/speech/... -run TestRealEdgeTTS -v

func TestRealEdgeTTS_Synthesize(t *testing.T) {
	p := New()
	model := p.SpeechModel("edge-read-aloud")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := sdk.GenerateSpeech(ctx,
		sdk.WithSpeechModel(model),
		sdk.WithText("Hello, this is a real Edge TTS test."),
		sdk.WithSpeechConfig(map[string]any{
			"voice":    "en-US-JennyNeural",
			"language": "en-US",
			"speed":    1.0,
		}),
	)
	if err != nil {
		t.Fatalf("GenerateSpeech: %v", err)
	}
	if len(result.Audio) == 0 {
		t.Fatal("expected non-empty audio from real Edge TTS")
	}
	t.Logf("got %d bytes of audio (content-type: %s)", len(result.Audio), result.ContentType)
}

func TestRealEdgeTTS_Stream(t *testing.T) {
	p := New()
	model := p.SpeechModel("edge-read-aloud")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	sr, err := sdk.StreamSpeech(ctx,
		sdk.WithSpeechModel(model),
		sdk.WithText("你好，这是流式测试。"),
		sdk.WithSpeechConfig(map[string]any{
			"voice":    "zh-CN-XiaoxiaoNeural",
			"language": "zh-CN",
		}),
	)
	if err != nil {
		t.Fatalf("StreamSpeech: %v", err)
	}
	var total int
	for b := range sr.Stream {
		total += len(b)
	}
	if total == 0 {
		t.Fatal("expected non-empty audio stream")
	}
	t.Logf("streamed %d bytes total", total)
}

func TestRealEdgeTTS_Formats(t *testing.T) {
	formats := []string{
		"audio-24khz-48kbitrate-mono-mp3",
		"audio-24khz-96kbitrate-mono-mp3",
		"webm-24khz-16bit-mono-opus",
	}

	for _, fmt := range formats {
		t.Run(fmt, func(t *testing.T) {
			p := New()
			model := p.SpeechModel("edge-read-aloud")

			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()

			result, err := sdk.GenerateSpeech(ctx,
				sdk.WithSpeechModel(model),
				sdk.WithText("Hello, format test."),
				sdk.WithSpeechConfig(map[string]any{
					"voice":    "en-US-JennyNeural",
					"language": "en-US",
					"format":   fmt,
					"speed":    1.0,
				}),
			)
			if err != nil {
				t.Errorf("UNSUPPORTED format %q: %v", fmt, err)
				return
			}
			t.Logf("OK format %q -> %d bytes", fmt, len(result.Audio))
		})
	}
}

func TestRealEdgeTTS_SaveAudio(t *testing.T) {
	p := New()
	model := p.SpeechModel("edge-read-aloud")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	cases := []struct {
		name  string
		text  string
		voice string
		lang  string
		file  string
	}{
		{"en", "Hello, this is an Edge TTS audio save test.", "en-US-JennyNeural", "en-US", "test_en.mp3"},
		{"zh", "你好，这是一段中文语音合成测试。", "zh-CN-XiaoxiaoNeural", "zh-CN", "test_zh.mp3"},
	}

	outDir := filepath.Join(os.TempDir(), "edge_tts_test")
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", outDir, err)
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := sdk.GenerateSpeech(ctx,
				sdk.WithSpeechModel(model),
				sdk.WithText(tc.text),
				sdk.WithSpeechConfig(map[string]any{
					"voice":    tc.voice,
					"language": tc.lang,
					"speed":    1.0,
					"pitch":    float64(-10),
				}),
			)
			if err != nil {
				t.Fatalf("GenerateSpeech: %v", err)
			}
			if len(result.Audio) == 0 {
				t.Fatal("expected non-empty audio")
			}
			outPath := filepath.Join(outDir, tc.file)
			if err := os.WriteFile(outPath, result.Audio, 0o644); err != nil {
				t.Fatalf("write file: %v", err)
			}
			t.Logf("saved %d bytes -> %s", len(result.Audio), outPath)
		})
	}
}
