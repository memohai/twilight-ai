package speech

import (
	"fmt"
	"strings"
)

// audioConfig holds Microsoft Azure Cognitive Services TTS options extracted
// from SpeechParams.Config.
//
// Supported keys:
//   - "region"        (string):  Azure region, e.g. "eastus", "eastasia" (required
//                                unless WithBaseURL is used)
//   - "voice"         (string):  voice name, e.g. "en-US-JennyNeural" (default)
//   - "language"      (string):  BCP-47 tag for the <speak> element; inferred from
//                                voice name when not set
//   - "output_format" (string):  X-Microsoft-OutputFormat value; default
//                                "audio-16khz-128kbitrate-mono-mp3"
//   - "style"         (string):  speaking style, e.g. "cheerful", "sad" (optional)
//   - "rate"          (string):  speaking rate, e.g. "+10%", "0.5" (optional)
//   - "pitch"         (string):  pitch, e.g. "+5Hz", "+10%" (optional)
type audioConfig struct {
	Region       string
	Voice        string
	Language     string
	OutputFormat string
	Style        string
	Rate         string
	Pitch        string
}

func parseConfig(cfg map[string]any) audioConfig {
	ac := audioConfig{
		Voice:        defaultVoice,
		OutputFormat: defaultOutputFormat,
	}
	if cfg == nil {
		return ac
	}
	if v, ok := cfg["region"].(string); ok && v != "" {
		ac.Region = v
	}
	if v, ok := cfg["voice"].(string); ok && v != "" {
		ac.Voice = v
	}
	if v, ok := cfg["language"].(string); ok && v != "" {
		ac.Language = v
	}
	if v, ok := cfg["output_format"].(string); ok && v != "" {
		ac.OutputFormat = v
	}
	if v, ok := cfg["style"].(string); ok {
		ac.Style = v
	}
	if v, ok := cfg["rate"].(string); ok {
		ac.Rate = v
	}
	if v, ok := cfg["pitch"].(string); ok {
		ac.Pitch = v
	}
	return ac
}

// languageFor returns the BCP-47 language tag to use in the SSML <speak> element.
// If cfg.Language is set it is used directly; otherwise the first two dash-separated
// segments of the voice name are used (e.g. "en-US" from "en-US-JennyNeural").
func languageFor(cfg *audioConfig) string {
	if cfg.Language != "" {
		return cfg.Language
	}
	parts := strings.SplitN(cfg.Voice, "-", 3)
	if len(parts) >= 2 {
		return parts[0] + "-" + parts[1]
	}
	return "en-US"
}

// buildSSML builds the SSML payload sent to Azure TTS.
func buildSSML(text string, cfg *audioConfig) string {
	lang := languageFor(cfg)

	// Build the inner prosody/voice content.
	inner := xmlEscape(text)
	if cfg.Rate != "" || cfg.Pitch != "" {
		var attrs string
		if cfg.Rate != "" {
			attrs += fmt.Sprintf(` rate=%q`, cfg.Rate)
		}
		if cfg.Pitch != "" {
			attrs += fmt.Sprintf(` pitch=%q`, cfg.Pitch)
		}
		inner = fmt.Sprintf(`<prosody%s>%s</prosody>`, attrs, inner)
	}

	// Wrap in <mstts:express-as> when style is requested.
	if cfg.Style != "" {
		inner = fmt.Sprintf(
			`<mstts:express-as style=%q>%s</mstts:express-as>`,
			cfg.Style, inner,
		)
	}

	return fmt.Sprintf(
		`<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"`+
			` xmlns:mstts="http://www.w3.org/2001/mstts"`+
			` xml:lang="%s"><voice name="%s">%s</voice></speak>`,
		lang, cfg.Voice, inner,
	)
}

// xmlEscape replaces the five XML-special characters so that arbitrary text is
// safe inside an SSML element.
func xmlEscape(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	s = strings.ReplaceAll(s, `"`, "&quot;")
	s = strings.ReplaceAll(s, "'", "&apos;")
	return s
}

// contentTypeForFormat maps X-Microsoft-OutputFormat values to MIME types.
func contentTypeForFormat(format string) string {
	lower := strings.ToLower(format)
	switch {
	case strings.Contains(lower, "mp3"):
		return "audio/mpeg"
	case strings.Contains(lower, "wav") || strings.Contains(lower, "pcm"):
		return "audio/wav"
	case strings.Contains(lower, "webm"):
		return "audio/webm"
	case strings.Contains(lower, "opus"):
		return "audio/ogg"
	default:
		return "audio/mpeg"
	}
}
