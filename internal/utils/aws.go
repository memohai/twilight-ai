package utils

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	v4 "github.com/aws/aws-sdk-go-v2/aws/signer/v4"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
)

const (
	emptySHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

	bedrockService = "bedrock"
)

type lazyCredentialsProvider struct {
	region string

	once sync.Once
	cp   aws.CredentialsProvider
	err  error
}

func NewBedrockDefaultCredentialsPreparer(region string) func(*http.Request) error {
	return NewSigV4Preparer(bedrockService, region, &lazyCredentialsProvider{region: region})
}

func NewBedrockStaticCredentialsPreparer(region, accessKeyID, secretAccessKey, sessionToken string) func(*http.Request) error {
	cp := aws.NewCredentialsCache(credentials.NewStaticCredentialsProvider(accessKeyID, secretAccessKey, sessionToken))
	return NewSigV4Preparer(bedrockService, region, cp)
}

func NewSigV4Preparer(service, region string, cp aws.CredentialsProvider) func(*http.Request) error {
	signer := v4.NewSigner()

	return func(req *http.Request) error {
		payloadHash, err := requestPayloadHash(req)
		if err != nil {
			return err
		}

		creds, err := cp.Retrieve(req.Context())
		if err != nil {
			return fmt.Errorf("retrieve AWS credentials: %w", err)
		}

		if err := signer.SignHTTP(req.Context(), creds, req, payloadHash, service, region, time.Now().UTC()); err != nil {
			return fmt.Errorf("sign AWS request: %w", err)
		}

		return nil
	}
}

func (p *lazyCredentialsProvider) Retrieve(ctx context.Context) (aws.Credentials, error) {
	p.once.Do(func() {
		cfg, err := config.LoadDefaultConfig(ctx, config.WithRegion(p.region))
		if err != nil {
			p.err = err
			return
		}
		p.cp = cfg.Credentials
	})
	if p.err != nil {
		return aws.Credentials{}, p.err
	}
	return p.cp.Retrieve(ctx)
}

func requestPayloadHash(req *http.Request) (string, error) {
	if req.Body == nil {
		return emptySHA256, nil
	}

	if req.GetBody != nil {
		body, err := req.GetBody()
		if err != nil {
			return "", fmt.Errorf("clone request body: %w", err)
		}
		defer body.Close()
		return hashReader(body)
	}

	data, err := io.ReadAll(req.Body)
	if err != nil {
		return "", fmt.Errorf("read request body: %w", err)
	}

	req.Body = io.NopCloser(bytes.NewReader(data))
	req.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(data)), nil
	}

	return hashBytes(data), nil
}

func hashReader(r io.Reader) (string, error) {
	h := sha256.New()
	if _, err := io.Copy(h, r); err != nil {
		return "", fmt.Errorf("hash request body: %w", err)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func hashBytes(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
