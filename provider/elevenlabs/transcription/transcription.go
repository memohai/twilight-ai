package transcription

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strconv"
	"strings"

	"github.com/memohai/twilight-ai/internal/utils"
	sdk "github.com/memohai/twilight-ai/sdk"
)

const (
	defaultModelID = "scribe_v2"
	defaultBaseURL = "https://api.elevenlabs.io"
)

type Option func(*Provider)

func WithAPIKey(key string) Option { return func(p *Provider) { p.apiKey = key } }
func WithBaseURL(url string) Option {
	return func(p *Provider) { p.baseURL = strings.TrimRight(url, "/") }
}
func WithHTTPClient(hc *http.Client) Option { return func(p *Provider) { p.httpClient = hc } }

type Provider struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

func New(opts ...Option) *Provider {
	p := &Provider{baseURL: defaultBaseURL, httpClient: &http.Client{}}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

func (p *Provider) TranscriptionModel(id string) *sdk.TranscriptionModel {
	if id == "" {
		id = defaultModelID
	}
	return &sdk.TranscriptionModel{ID: id, Provider: p}
}

func (p *Provider) ListModels(ctx context.Context) ([]*sdk.TranscriptionModel, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.baseURL+"/v1/models", http.NoBody)
	if err != nil {
		return nil, fmt.Errorf("elevenlabs transcription: build list models request: %w", err)
	}
	req.Header.Set("xi-api-key", p.apiKey)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("elevenlabs transcription: list models request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("elevenlabs transcription: unexpected status %d: %s", resp.StatusCode, string(body))
	}

	rawModels, err := decodeTranscriptionModelsResponse(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("elevenlabs transcription: decode response: %w", err)
	}

	models := make([]*sdk.TranscriptionModel, 0, len(rawModels))
	for _, raw := range rawModels {
		id, _ := raw["model_id"].(string)
		if id == "" {
			continue
		}
		if canDo, ok := raw["can_do_speech_to_text"].(bool); ok {
			if canDo {
				models = append(models, p.TranscriptionModel(id))
			}
			continue
		}
		if strings.Contains(strings.ToLower(id), "scribe") {
			models = append(models, p.TranscriptionModel(id))
		}
	}
	if len(models) == 0 {
		return nil, errors.New("elevenlabs transcription: no transcription models returned by provider")
	}
	return models, nil
}

func decodeTranscriptionModelsResponse(r io.Reader) ([]map[string]any, error) {
	body, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}

	var wrapped struct {
		Models []map[string]any `json:"models"`
	}
	if err := json.Unmarshal(body, &wrapped); err == nil && len(wrapped.Models) > 0 {
		return wrapped.Models, nil
	}

	var direct []map[string]any
	if err := json.Unmarshal(body, &direct); err != nil {
		return nil, err
	}
	return direct, nil
}

type audioConfig struct {
	LanguageCode          string
	TagAudioEvents        *bool
	Diarize               *bool
	NumSpeakers           *int
	TimestampsGranularity string
}

func parseConfig(cfg map[string]any) audioConfig {
	ac := audioConfig{}
	if cfg == nil {
		return ac
	}
	if v, ok := cfg["language_code"].(string); ok && v != "" {
		ac.LanguageCode = v
	}
	if v, ok := cfg["tag_audio_events"].(bool); ok {
		ac.TagAudioEvents = &v
	}
	if v, ok := cfg["diarize"].(bool); ok {
		ac.Diarize = &v
	}
	if v, ok := utils.ToInt(cfg["num_speakers"]); ok {
		ac.NumSpeakers = &v
	}
	if v, ok := cfg["timestamps_granularity"].(string); ok && v != "" {
		ac.TimestampsGranularity = v
	}
	return ac
}

func (p *Provider) DoTranscribe(ctx context.Context, params sdk.TranscriptionParams) (*sdk.TranscriptionResult, error) {
	cfg := parseConfig(params.Config)
	modelID := defaultModelID
	if params.Model != nil && params.Model.ID != "" {
		modelID = params.Model.ID
	}

	body, contentType, err := buildMultipartBody(&params, modelID, cfg)
	if err != nil {
		return nil, fmt.Errorf("elevenlabs transcription: build request body: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/v1/speech-to-text", body)
	if err != nil {
		return nil, fmt.Errorf("elevenlabs transcription: build request: %w", err)
	}
	req.Header.Set("Content-Type", contentType)
	req.Header.Set("xi-api-key", p.apiKey)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("elevenlabs transcription: request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("elevenlabs transcription: unexpected status %d: %s", resp.StatusCode, string(b))
	}

	var payload struct {
		Text         string `json:"text"`
		LanguageCode string `json:"language_code"`
		Words        []struct {
			Text      string  `json:"text"`
			Start     float64 `json:"start"`
			End       float64 `json:"end"`
			SpeakerID string  `json:"speaker_id"`
			Type      string  `json:"type"`
		} `json:"words"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("elevenlabs transcription: decode response: %w", err)
	}
	out := &sdk.TranscriptionResult{Text: payload.Text, Language: payload.LanguageCode}
	if len(payload.Words) > 0 {
		out.Words = make([]sdk.TranscriptionWord, 0, len(payload.Words))
		for _, w := range payload.Words {
			if w.Type != "" && w.Type != "word" && strings.TrimSpace(w.Text) == "" {
				continue
			}
			out.Words = append(out.Words, sdk.TranscriptionWord{
				Text:      w.Text,
				Start:     w.Start,
				End:       w.End,
				SpeakerID: w.SpeakerID,
			})
		}
	}
	return out, nil
}

func buildMultipartBody(params *sdk.TranscriptionParams, modelID string, cfg audioConfig) (*bytes.Buffer, string, error) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	if err := w.WriteField("model_id", modelID); err != nil {
		return nil, "", err
	}
	if cfg.LanguageCode != "" {
		if err := w.WriteField("language_code", cfg.LanguageCode); err != nil {
			return nil, "", err
		}
	}
	if cfg.TagAudioEvents != nil {
		if err := w.WriteField("tag_audio_events", strconv.FormatBool(*cfg.TagAudioEvents)); err != nil {
			return nil, "", err
		}
	}
	if cfg.Diarize != nil {
		if err := w.WriteField("diarize", strconv.FormatBool(*cfg.Diarize)); err != nil {
			return nil, "", err
		}
	}
	if cfg.NumSpeakers != nil {
		if err := w.WriteField("num_speakers", strconv.Itoa(*cfg.NumSpeakers)); err != nil {
			return nil, "", err
		}
	}
	if cfg.TimestampsGranularity != "" {
		if err := w.WriteField("timestamps_granularity", cfg.TimestampsGranularity); err != nil {
			return nil, "", err
		}
	}

	part, err := w.CreateFormFile("file", params.Filename)
	if err != nil {
		return nil, "", err
	}
	if _, err := part.Write(params.Audio); err != nil {
		return nil, "", err
	}
	if err := w.Close(); err != nil {
		return nil, "", err
	}
	return &buf, w.FormDataContentType(), nil
}
