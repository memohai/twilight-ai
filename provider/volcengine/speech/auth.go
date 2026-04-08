package speech

// Volcengine V4 HMAC-SHA256 signing implementation.
// Reference: https://github.com/volcengine/volc-openapi-demos/blob/main/signature/golang/sign.go
//
// The Authorization header follows the format:
//
//	HMAC-SHA256 Credential=AccessKeyID/CredentialScope, SignedHeaders=..., Signature=...
//
// where CredentialScope = YYYYMMDD/region/service/request.

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

const (
	samiRegion  = "cn-north-1"
	samiService = "sami"
	samiHost    = "open.volcengineapi.com"
	samiVersion = "2021-07-27"
)

// tokenCache holds a cached SAMI access token with expiration tracking.
type tokenCache struct {
	mu        sync.Mutex
	token     string
	expiresAt time.Time
}

func (tc *tokenCache) get() (string, bool) {
	tc.mu.Lock()
	defer tc.mu.Unlock()
	if tc.token == "" || time.Now().After(tc.expiresAt.Add(-60*time.Second)) {
		return "", false
	}
	return tc.token, true
}

func (tc *tokenCache) set(token string, expiresAt int64) {
	tc.mu.Lock()
	defer tc.mu.Unlock()
	tc.token = token
	tc.expiresAt = time.Unix(expiresAt, 0)
}

// getToken calls the SAMI GetToken OpenAPI to obtain a bearer token.
// The request is authenticated with Volcengine V4 HMAC-SHA256 signing.
func getToken(ctx context.Context, accessKey, secretKey, appKey string, httpClient *http.Client) (token string, expiresAt int64, err error) {
	bodyBytes, marshalErr := json.Marshal(map[string]any{ //nolint:gosec // appkey is a variable, not a hardcoded credential
		"appkey":        appKey,
		"token_version": "volc-auth-v1",
		"expiration":    3600,
	})
	if marshalErr != nil {
		return "", 0, fmt.Errorf("volcengine speech: marshal token request: %w", marshalErr)
	}

	q := url.Values{}
	q.Set("Action", "GetToken")
	q.Set("Version", samiVersion)

	reqURL := "https://" + samiHost + "/?" + q.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, reqURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return "", 0, fmt.Errorf("volcengine speech: build token request: %w", err)
	}

	// Set required headers before signing.
	now := time.Now().UTC()
	date := now.Format("20060102T150405Z")
	req.Header.Set("X-Date", date)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Content-Sha256", hex.EncodeToString(hashSHA256(bodyBytes)))
	req.Host = samiHost

	// Apply V4 signature.
	if err := signRequest(req, accessKey, secretKey, bodyBytes, q.Encode()); err != nil {
		return "", 0, err
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return "", 0, fmt.Errorf("volcengine speech: token request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return "", 0, fmt.Errorf("volcengine speech: token request status %d: %s", resp.StatusCode, string(b))
	}

	var result struct {
		StatusCode int32  `json:"status_code"`
		StatusText string `json:"status_text"`
		Token      string `json:"token"`
		ExpiresAt  int64  `json:"expires_at"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", 0, fmt.Errorf("volcengine speech: decode token response: %w", err)
	}
	if result.Token == "" {
		return "", 0, fmt.Errorf("volcengine speech: empty token, status=%d msg=%s", result.StatusCode, result.StatusText)
	}
	return result.Token, result.ExpiresAt, nil
}

// signRequest applies the Volcengine V4 HMAC-SHA256 signature to the request.
// The request must have the X-Date header set before this function is called.
func signRequest(req *http.Request, accessKey, secretKey string, body []byte, rawQuery string) error {
	date := req.Header.Get("X-Date")
	if len(date) < 8 {
		return fmt.Errorf("volcengine speech: X-Date header missing or malformed: %q", date)
	}
	authDate := date[:8]

	payloadHash := hex.EncodeToString(hashSHA256(body))
	req.Header.Set("X-Content-Sha256", payloadHash)

	queryString := strings.ReplaceAll(rawQuery, "+", "%20")
	signedHeaders := []string{"content-type", "host", "x-content-sha256", "x-date"}

	var headerLines []string
	for _, h := range signedHeaders {
		if h == "host" {
			headerLines = append(headerLines, "host:"+req.Host)
		} else {
			headerLines = append(headerLines, h+":"+strings.TrimSpace(req.Header.Get(h)))
		}
	}
	headerString := strings.Join(headerLines, "\n")

	path := req.URL.Path
	if path == "" {
		path = "/"
	}

	canonicalRequest := strings.Join([]string{
		req.Method,
		path,
		queryString,
		headerString + "\n",
		strings.Join(signedHeaders, ";"),
		payloadHash,
	}, "\n")

	hashedCanonical := hex.EncodeToString(hashSHA256([]byte(canonicalRequest)))
	credentialScope := authDate + "/" + samiRegion + "/" + samiService + "/request"
	stringToSign := strings.Join([]string{
		"HMAC-SHA256",
		date,
		credentialScope,
		hashedCanonical,
	}, "\n")

	signingKey := getSignedKey(secretKey, authDate, samiRegion, samiService)
	signature := hex.EncodeToString(hmacSHA256(signingKey, stringToSign))

	authorization := "HMAC-SHA256" +
		" Credential=" + accessKey + "/" + credentialScope +
		", SignedHeaders=" + strings.Join(signedHeaders, ";") +
		", Signature=" + signature
	req.Header.Set("Authorization", authorization)
	return nil
}

func getSignedKey(secretKey, date, region, service string) []byte {
	kDate := hmacSHA256([]byte(secretKey), date)
	kRegion := hmacSHA256(kDate, region)
	kService := hmacSHA256(kRegion, service)
	return hmacSHA256(kService, "request")
}

func hmacSHA256(key []byte, content string) []byte {
	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(content))
	return mac.Sum(nil)
}

func hashSHA256(data []byte) []byte {
	h := sha256.New()
	h.Write(data)
	return h.Sum(nil)
}
