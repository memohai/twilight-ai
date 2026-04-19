package completions

import (
	"crypto/rand"
	"fmt"
	"strings"
)

func generateID() string {
	b := make([]byte, 12)
	if _, err := rand.Read(b); err != nil {
		panic("openai-completions: generateID entropy failure: " + err.Error())
	}
	return fmt.Sprintf("call_%x", b)
}

// parseDataURL extracts media type and raw data from a URL that may be a data URI.
// For non-data URIs, it returns "image/png" as the default media type and the original URL as data.
func parseDataURL(url string) (mediaType, data string) {
	mediaType = "image/png"
	if strings.HasPrefix(url, "data:") {
		rest := url[len("data:"):]
		if semi := strings.Index(rest, ";"); semi > 0 {
			mediaType = rest[:semi]
		}
	}
	data = url
	if ci := strings.Index(url, ","); ci >= 0 {
		data = url[ci+1:]
	}
	return mediaType, data
}
