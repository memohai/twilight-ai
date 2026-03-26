package speech

import (
	_ "embed"
	"encoding/json"
)

const (
	edgeSpeechURL        = "wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1"
	edgeAPIToken         = "6A5AA1D4EAFF4E9FB37E23D68491D6F4" //nolint:gosec // Well-known public token shared by all Edge TTS clients, not a secret.
	chromiumFullVersion  = "143.0.3650.75"
	chromiumMajorVersion = "143"

	wssOrigin = "chrome-extension://jdiccldimpdaibmpdkjnbmckianbfold"

	DefaultVoice = "en-US-EmmaMultilingualNeural"

	winEpochOffset = 11644473600
	sToNS          = 1000000000
)

//go:embed voices.json
var voicesJSON []byte

// EdgeTTSVoices maps language tags to available voice IDs, loaded from voices.json.
var EdgeTTSVoices map[string][]string

var voiceToLang map[string]string

func init() {
	if err := json.Unmarshal(voicesJSON, &EdgeTTSVoices); err != nil {
		panic("edge: failed to parse voices.json: " + err.Error())
	}
	voiceToLang = make(map[string]string, 256)
	for lang, voices := range EdgeTTSVoices {
		for _, v := range voices {
			voiceToLang[v] = lang
		}
	}
}

// LookupVoiceLang returns the language for a known Edge TTS voice ID.
// Returns ("", false) if the voice is not recognized.
func LookupVoiceLang(voiceID string) (string, bool) {
	lang, ok := voiceToLang[voiceID]
	return lang, ok
}
