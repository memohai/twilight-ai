package videos

type createRequest struct {
	Model           string           `json:"model"`
	Prompt          string           `json:"prompt"`
	AspectRatio     string           `json:"aspect_ratio,omitempty"`
	CallbackURL     string           `json:"callback_url,omitempty"`
	Duration        *int             `json:"duration,omitempty"`
	FrameImages     []frameImage     `json:"frame_images,omitempty"`
	GenerateAudio   *bool            `json:"generate_audio,omitempty"`
	InputReferences []inputReference `json:"input_references,omitempty"`
	Provider        any              `json:"provider,omitempty"`
	Resolution      string           `json:"resolution,omitempty"`
	Seed            *int64           `json:"seed,omitempty"`
	Size            string           `json:"size,omitempty"`
}

type mediaURLObject struct {
	URL string `json:"url"`
}

type frameImage struct {
	ImageURL  mediaURLObject `json:"image_url"`
	Type      string         `json:"type"`
	FrameType string         `json:"frame_type"`
}

type inputReference struct {
	Type     string          `json:"type"`
	AudioURL *mediaURLObject `json:"audio_url,omitempty"`
	ImageURL *mediaURLObject `json:"image_url,omitempty"`
	VideoURL *mediaURLObject `json:"video_url,omitempty"`
}

type videoResponse struct {
	ID           string         `json:"id"`
	PollingURL   string         `json:"polling_url,omitempty"`
	Status       string         `json:"status"`
	Error        string         `json:"error,omitempty"`
	GenerationID string         `json:"generation_id,omitempty"`
	UnsignedURLs []string       `json:"unsigned_urls,omitempty"`
	Usage        map[string]any `json:"usage,omitempty"`
}

type listModelsResponse struct {
	Data []modelResponse `json:"data"`
}

type modelResponse struct {
	AllowedPassthroughParameters []string       `json:"allowed_passthrough_parameters"`
	CanonicalSlug                string         `json:"canonical_slug"`
	Created                      int64          `json:"created"`
	Description                  string         `json:"description"`
	GenerateAudio                *bool          `json:"generate_audio"`
	HuggingFaceID                *string        `json:"hugging_face_id"`
	ID                           string         `json:"id"`
	Name                         string         `json:"name"`
	PricingSKUs                  map[string]any `json:"pricing_skus"`
	Seed                         *bool          `json:"seed"`
	SupportedAspectRatios        []string       `json:"supported_aspect_ratios"`
	SupportedDurations           []int          `json:"supported_durations"`
	SupportedFrameImages         []string       `json:"supported_frame_images"`
	SupportedResolutions         []string       `json:"supported_resolutions"`
	SupportedSizes               []string       `json:"supported_sizes"`
}
