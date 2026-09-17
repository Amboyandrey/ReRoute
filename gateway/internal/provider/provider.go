// Package provider implements upstream LLM provider adapters. Requests
// arrive at the gateway already in OpenAI chat-completions format; each
// adapter is responsible for speaking that provider's wire protocol and,
// for streaming, forwarding an OpenAI-compatible SSE stream back to the
// caller.
package provider

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/Amboyandrey/ReRoute/gateway/internal/config"
)

// Result carries accounting info recovered from a completed (non-streamed
// usage is parsed from the response body; for streaming responses usage is
// best-effort since not all providers emit it in-stream).
type Result struct {
	PromptTokens     int
	CompletionTokens int
	Latency          time.Duration
	StatusCode       int
}

// Client sends a chat-completions request to a tier's upstream and writes
// the (possibly streamed) response body to w, mirroring status/headers.
// body is the raw OpenAI-format request JSON received from the caller.
type Client interface {
	ChatCompletion(ctx context.Context, tier config.Tier, body []byte, stream bool, w http.ResponseWriter) (Result, error)
}

// New returns the Client implementation for a tier's Kind.
func New(httpClient *http.Client) map[string]Client {
	return map[string]Client{
		"openai":    &OpenAICompatClient{HTTP: httpClient},
		"anthropic": &AnthropicClient{HTTP: httpClient},
	}
}

// ForKind is a small helper for tests / call sites with only a kind string.
func ForKind(kind string, clients map[string]Client) (Client, error) {
	c, ok := clients[kind]
	if !ok {
		return nil, fmt.Errorf("no provider client registered for kind %q", kind)
	}
	return c, nil
}

func copyHeaders(dst http.Header, src http.Header, skip map[string]bool) {
	for k, vv := range src {
		if skip[k] {
			continue
		}
		for _, v := range vv {
			dst.Add(k, v)
		}
	}
}

func drainAndClose(r io.ReadCloser) {
	if r == nil {
		return
	}
	_, _ = io.Copy(io.Discard, r)
	_ = r.Close()
}

func readAll(r io.Reader) ([]byte, error) {
	return io.ReadAll(r)
}
