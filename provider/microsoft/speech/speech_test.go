package speech

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	sdk "github.com/memohai/twilight-ai/sdk"
)

// mockAzureTTSHandler returns an HTTP handler simulating the Azure TTS endpoint.
func mockAzureTTSHandler(t *testing.T) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != ttsPath {
			t.Errorf("unexpected path: %s, want %s", r.URL.Path, ttsPath)
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		if r.Header.Get("Ocp-Apim-Subscription-Key") == "" {
			http.Error(w, "missing subscription key", http.StatusUnauthorized)
			return
		}
		ct := r.Header.Get("Content-Type")
		if !strings.HasPrefix(ct, "application/ssml+xml") {
			t.Errorf("unexpected Content-Type: %s", ct)
		}
		if r.Header.Get("X-Microsoft-OutputFormat") == "" {
			t.Error("X-Microsoft-OutputFormat header missing")
		}
		w.Header().Set("Content-Type", "audio/mpeg")
		_, _ = w.Write([]byte("fake-azure-audio"))
	})
}

func TestProvider_DoSynthesize(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(mockAzureTTSHandler(t))
	defer srv.Close()

	p := New(WithAPIKey("test-key"), WithBaseURL(srv.URL))

	result, err := p.DoSynthesize(context.Background(), sdk.SpeechParams{
		Text: "Hello Azure",
		Config: map[string]any{
			"voice": "en-US-JennyNeural",
		},
	})
	if err != nil {
		t.Fatalf("DoSynthesize: %v", err)
	}
	if !bytes.Equal(result.Audio, []byte("fake-azure-audio")) {
		t.Errorf("audio = %q, want %q", string(result.Audio), "fake-azure-audio")
	}
	if result.ContentType != "audio/mpeg" {
		t.Errorf("content type = %q, want audio/mpeg", result.ContentType)
	}
}

func TestProvider_DoStream(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(mockAzureTTSHandler(t))
	defer srv.Close()

	p := New(WithAPIKey("test-key"), WithBaseURL(srv.URL))

	result, err := p.DoStream(context.Background(), sdk.SpeechParams{
		Text: "Streaming Azure",
		Config: map[string]any{
			"voice": "en-US-JennyNeural",
		},
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}

	audio, err := result.Bytes()
	if err != nil {
		t.Fatalf("Bytes: %v", err)
	}
	if !bytes.Equal(audio, []byte("fake-azure-audio")) {
		t.Errorf("audio = %q", string(audio))
	}
}

func TestProvider_MissingRegion(t *testing.T) {
	t.Parallel()
	p := New(WithAPIKey("key"))

	_, err := p.DoSynthesize(context.Background(), sdk.SpeechParams{Text: "test"})
	if err == nil {
		t.Fatal("expected error when region is missing")
	}
	if !strings.Contains(err.Error(), "region") {
		t.Errorf("error should mention region, got: %v", err)
	}
}

func TestProvider_DoSynthesize_HTTPError(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
	}))
	defer srv.Close()

	p := New(WithAPIKey("bad-key"), WithBaseURL(srv.URL))
	_, err := p.DoSynthesize(context.Background(), sdk.SpeechParams{
		Text:   "test",
		Config: map[string]any{"voice": "en-US-JennyNeural"},
	})
	if err == nil {
		t.Fatal("expected error for 401")
	}
}

func TestProvider_SpeechModel(t *testing.T) {
	t.Parallel()
	p := New()
	m := p.SpeechModel("microsoft-tts")
	if m.ID != "microsoft-tts" {
		t.Errorf("ID = %q", m.ID)
	}
	m2 := p.SpeechModel("")
	if m2.ID != defaultModelID {
		t.Errorf("default ID = %q, want %q", m2.ID, defaultModelID)
	}
}

func TestBuildSSML(t *testing.T) {
	t.Parallel()

	t.Run("simple", func(t *testing.T) {
		t.Parallel()
		ssml := buildSSML("Hello", &audioConfig{
			Voice:    "en-US-JennyNeural",
			Language: "en-US",
		})
		if !strings.Contains(ssml, `name="en-US-JennyNeural"`) {
			t.Errorf("SSML missing voice name: %s", ssml)
		}
		if !strings.Contains(ssml, "Hello") {
			t.Errorf("SSML missing text: %s", ssml)
		}
	})

	t.Run("xml_escape", func(t *testing.T) {
		t.Parallel()
		ssml := buildSSML("a & b < c > d", &audioConfig{Voice: "en-US-JennyNeural"})
		if !strings.Contains(ssml, "a &amp; b &lt; c &gt; d") {
			t.Errorf("SSML XML escaping failed: %s", ssml)
		}
	})

	t.Run("with_style", func(t *testing.T) {
		t.Parallel()
		ssml := buildSSML("Hi", &audioConfig{Voice: "zh-CN-XiaoxiaoNeural", Style: "cheerful"})
		if !strings.Contains(ssml, `style="cheerful"`) {
			t.Errorf("SSML missing style: %s", ssml)
		}
	})

	t.Run("with_rate_pitch", func(t *testing.T) {
		t.Parallel()
		ssml := buildSSML("Test", &audioConfig{Voice: "en-US-JennyNeural", Rate: "+10%", Pitch: "+5Hz"})
		if !strings.Contains(ssml, `rate="+10%"`) {
			t.Errorf("SSML missing rate: %s", ssml)
		}
		if !strings.Contains(ssml, `pitch="+5Hz"`) {
			t.Errorf("SSML missing pitch: %s", ssml)
		}
	})
}

func TestLanguageFor(t *testing.T) {
	t.Parallel()
	cases := []struct {
		cfg  audioConfig
		want string
	}{
		{audioConfig{Voice: "en-US-JennyNeural"}, "en-US"},
		{audioConfig{Voice: "zh-CN-XiaoxiaoNeural"}, "zh-CN"},
		{audioConfig{Voice: "ja-JP-NanamiNeural"}, "ja-JP"},
		{audioConfig{Voice: "en-US-JennyNeural", Language: "fr-FR"}, "fr-FR"},
	}
	for _, tc := range cases {
		got := languageFor(&tc.cfg)
		if got != tc.want {
			t.Errorf("languageFor(%v) = %q, want %q", tc.cfg.Voice, got, tc.want)
		}
	}
}

func TestContentTypeForFormat(t *testing.T) {
	t.Parallel()
	cases := []struct{ format, want string }{
		{"audio-16khz-128kbitrate-mono-mp3", "audio/mpeg"},
		{"audio-24khz-160kbitrate-mono-mp3", "audio/mpeg"},
		{"riff-16khz-16bit-mono-pcm", "audio/wav"},
		{"riff-24khz-16bit-mono-pcm", "audio/wav"},
		{"ogg-16khz-16bit-mono-opus", "audio/ogg"},
		{"webm-16khz-16bit-mono-opus", "audio/webm"},
		{"unknown", "audio/mpeg"},
	}
	for _, tc := range cases {
		got := contentTypeForFormat(tc.format)
		if got != tc.want {
			t.Errorf("contentTypeForFormat(%q) = %q, want %q", tc.format, got, tc.want)
		}
	}
}

func TestParseConfig_Defaults(t *testing.T) {
	t.Parallel()
	cfg := parseConfig(nil)
	if cfg.Voice != defaultVoice {
		t.Errorf("default voice = %q, want %q", cfg.Voice, defaultVoice)
	}
	if cfg.OutputFormat != defaultOutputFormat {
		t.Errorf("default output_format = %q, want %q", cfg.OutputFormat, defaultOutputFormat)
	}
}

func TestParseConfig(t *testing.T) {
	t.Parallel()
	cfg := parseConfig(map[string]any{
		"region":        "eastasia",
		"voice":         "zh-CN-XiaoxiaoNeural",
		"language":      "zh-CN",
		"output_format": "riff-16khz-16bit-mono-pcm",
		"style":         "cheerful",
		"rate":          "+10%",
		"pitch":         "+5Hz",
	})
	if cfg.Region != "eastasia" {
		t.Errorf("region = %q", cfg.Region)
	}
	if cfg.Voice != "zh-CN-XiaoxiaoNeural" {
		t.Errorf("voice = %q", cfg.Voice)
	}
	if cfg.Style != "cheerful" {
		t.Errorf("style = %q", cfg.Style)
	}
}
