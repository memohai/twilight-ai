package utils

import "strings"

// NormalizeFileData splits a FilePart payload into bare base64 data and a
// media type. FilePart.Data is bare base64 by convention; data URLs are
// tolerated and stripped so callers that still pass them keep working.
//
// Media type precedence: explicit argument, data URL header, then
// application/pdf as the fallback — file parts exist for provider-native
// document input, where PDF is the only interchange format every supported
// provider accepts, so an unlabeled payload is treated as one.
func NormalizeFileData(data, mediaType string) (string, string) {
	data = strings.TrimSpace(data)
	mediaType = strings.TrimSpace(mediaType)

	if strings.HasPrefix(strings.ToLower(data), "data:") {
		if idx := strings.Index(data, ","); idx >= 0 {
			header := data[len("data:"):idx]
			data = data[idx+1:]
			if mediaType == "" {
				if semi := strings.Index(header, ";"); semi >= 0 {
					mediaType = strings.TrimSpace(header[:semi])
				} else {
					mediaType = strings.TrimSpace(header)
				}
			}
		}
	}

	if mediaType == "" {
		mediaType = "application/pdf"
	}
	return data, mediaType
}

// FileDataURL renders a normalized file payload as a data URL, the shape the
// OpenAI-family file inputs expect in file_data fields.
func FileDataURL(base64Data, mediaType string) string {
	return "data:" + mediaType + ";base64," + base64Data
}

// OmittedFileNotice is the marker adapters without native file input emit in
// place of a FilePart. Silently sending megabytes of base64 as "text" is the
// failure mode file parts exist to fix, so the omission must be visible to
// both the model and anyone reading the request.
func OmittedFileNotice(filename, mediaType string) string {
	_, mediaType = NormalizeFileData("", mediaType)
	filename = strings.TrimSpace(filename)
	if filename == "" {
		filename = "attachment"
	}
	return "[file attachment omitted: this provider has no native file input; filename=" + filename + ", mediaType=" + mediaType + "]"
}
