package utils

import "testing"

func TestNormalizeFileData(t *testing.T) {
	cases := []struct {
		name      string
		data      string
		mediaType string
		wantData  string
		wantMime  string
	}{
		{
			name:      "bare base64 with explicit media type",
			data:      "JVBERi0xLjQ=",
			mediaType: "application/pdf",
			wantData:  "JVBERi0xLjQ=",
			wantMime:  "application/pdf",
		},
		{
			name:     "bare base64 defaults to pdf",
			data:     "JVBERi0xLjQ=",
			wantData: "JVBERi0xLjQ=",
			wantMime: "application/pdf",
		},
		{
			name:     "data URL is stripped and mime taken from header",
			data:     "data:application/pdf;base64,JVBERi0xLjQ=",
			wantData: "JVBERi0xLjQ=",
			wantMime: "application/pdf",
		},
		{
			name:      "explicit media type wins over data URL header",
			data:      "data:application/octet-stream;base64,JVBERi0xLjQ=",
			mediaType: "application/pdf",
			wantData:  "JVBERi0xLjQ=",
			wantMime:  "application/pdf",
		},
		{
			name:     "data URL header without base64 marker",
			data:     "data:text/plain,aGVsbG8=",
			wantData: "aGVsbG8=",
			wantMime: "text/plain",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotData, gotMime := NormalizeFileData(tc.data, tc.mediaType)
			if gotData != tc.wantData {
				t.Errorf("data: got %q, want %q", gotData, tc.wantData)
			}
			if gotMime != tc.wantMime {
				t.Errorf("mime: got %q, want %q", gotMime, tc.wantMime)
			}
		})
	}
}

func TestFileDataURL(t *testing.T) {
	got := FileDataURL("JVBERi0xLjQ=", "application/pdf")
	want := "data:application/pdf;base64,JVBERi0xLjQ="
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestOmittedFileNotice(t *testing.T) {
	got := OmittedFileNotice("report.pdf", "application/pdf")
	want := "[file attachment omitted: this provider has no native file input; filename=report.pdf, mediaType=application/pdf]"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}

	got = OmittedFileNotice("", "")
	want = "[file attachment omitted: this provider has no native file input; filename=attachment, mediaType=application/pdf]"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}
