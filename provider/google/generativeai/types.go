package generativeai

// --- Request types ---

type generateRequest struct {
	Contents          []content         `json:"contents"`
	SystemInstruction *content          `json:"systemInstruction,omitempty"`
	GenerationConfig  *generationConfig `json:"generationConfig,omitempty"`
	Tools             []toolGroup       `json:"tools,omitempty"`
	ToolConfig        *toolConfig       `json:"toolConfig,omitempty"`
	SafetySettings    []safetySetting   `json:"safetySettings,omitempty"`
}

type content struct {
	Role  string        `json:"role,omitempty"`
	Parts []contentPart `json:"parts"`
}

type contentPart struct {
	Text             string            `json:"text,omitempty"`
	InlineData       *inlineData       `json:"inlineData,omitempty"`
	FileData         *fileData         `json:"fileData,omitempty"`
	FunctionCall     *functionCall     `json:"functionCall,omitempty"`
	FunctionResponse *functionResponse `json:"functionResponse,omitempty"`
	Thought          *bool             `json:"thought,omitempty"`
	ThoughtSignature string            `json:"thoughtSignature,omitempty"`
}

type inlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type fileData struct {
	MimeType string `json:"mimeType"`
	FileURI  string `json:"fileUri"`
}

type functionCall struct {
	Name string `json:"name"`
	Args any    `json:"args"`
}

type functionResponse struct {
	Name     string              `json:"name"`
	Response functionResponseVal `json:"response"`
}

type functionResponseVal struct {
	Name    string `json:"name"`
	Content any    `json:"content"`
}

type generationConfig struct {
	MaxOutputTokens  *int     `json:"maxOutputTokens,omitempty"`
	Temperature      *float64 `json:"temperature,omitempty"`
	TopP             *float64 `json:"topP,omitempty"`
	TopK             *float64 `json:"topK,omitempty"`
	FrequencyPenalty *float64 `json:"frequencyPenalty,omitempty"`
	PresencePenalty  *float64 `json:"presencePenalty,omitempty"`
	StopSequences    []string `json:"stopSequences,omitempty"`
	Seed             *int     `json:"seed,omitempty"`
	ResponseMimeType string   `json:"responseMimeType,omitempty"`
	ResponseSchema   any      `json:"responseSchema,omitempty"`
}

type toolGroup struct {
	FunctionDeclarations []functionDeclaration `json:"functionDeclarations,omitempty"`
}

type functionDeclaration struct {
	Name                 string `json:"name"`
	Description          string `json:"description"`
	ParametersJSONSchema any    `json:"parametersJsonSchema,omitempty"`
}

type toolConfig struct {
	FunctionCallingConfig *functionCallingConfig `json:"functionCallingConfig,omitempty"`
}

type functionCallingConfig struct {
	Mode                 string   `json:"mode"`
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

type safetySetting struct {
	Category  string `json:"category"`
	Threshold string `json:"threshold"`
}

// --- Response types ---

type generateResponse struct {
	Candidates     []candidate     `json:"candidates"`
	UsageMetadata  *usageMetadata  `json:"usageMetadata,omitempty"`
	PromptFeedback *promptFeedback `json:"promptFeedback,omitempty"`
}

type candidate struct {
	Content       *content       `json:"content,omitempty"`
	FinishReason  string         `json:"finishReason,omitempty"`
	SafetyRatings []safetyRating `json:"safetyRatings,omitempty"`
}

type usageMetadata struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	TotalTokenCount         int `json:"totalTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount,omitempty"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount,omitempty"`
}

type promptFeedback struct {
	BlockReason   string         `json:"blockReason,omitempty"`
	SafetyRatings []safetyRating `json:"safetyRatings,omitempty"`
}

type safetyRating struct {
	Category    string `json:"category,omitempty"`
	Probability string `json:"probability,omitempty"`
	Blocked     bool   `json:"blocked,omitempty"`
}

// --- Models API response types ---

type googleModelsListResponse struct {
	Models        []googleModelObject `json:"models"`
	NextPageToken string              `json:"nextPageToken,omitempty"`
}

type googleModelObject struct {
	Name                       string   `json:"name"`
	DisplayName                string   `json:"displayName"`
	Description                string   `json:"description"`
	InputTokenLimit            int      `json:"inputTokenLimit"`
	OutputTokenLimit           int      `json:"outputTokenLimit"`
	SupportedGenerationMethods []string `json:"supportedGenerationMethods"`
}
