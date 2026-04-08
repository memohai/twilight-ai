package utils

import (
	"context"
	"fmt"
	"io"
)

// StreamHTTPBody reads body in 32 KiB chunks and sends them on the returned channels.
// body is always closed when the goroutine exits.
// prefix is prepended to any non-EOF read error (e.g. "openai speech").
//
// Usage:
//
//	ch, errCh := utils.StreamHTTPBody(ctx, resp.Body, "myprovider speech")
//	return sdk.NewSpeechStreamResult(ch, contentType, errCh), nil
func StreamHTTPBody(ctx context.Context, body io.ReadCloser, prefix string) (ch <-chan []byte, errCh <-chan error) {
	dataCh := make(chan []byte, 8)
	errChan := make(chan error, 1)
	ch = dataCh
	errCh = errChan

	go func() {
		defer body.Close()
		defer close(dataCh)
		defer close(errChan)

		buf := make([]byte, 32*1024)
		for {
			select {
			case <-ctx.Done():
				errChan <- ctx.Err()
				return
			default:
			}
			n, readErr := body.Read(buf)
			if n > 0 {
				chunk := make([]byte, n)
				copy(chunk, buf[:n])
				select {
				case dataCh <- chunk:
				case <-ctx.Done():
					errChan <- ctx.Err()
					return
				}
			}
			if readErr == io.EOF {
				return
			}
			if readErr != nil {
				errChan <- fmt.Errorf("%s: stream read: %w", prefix, readErr)
				return
			}
		}
	}()

	return ch, errCh
}
