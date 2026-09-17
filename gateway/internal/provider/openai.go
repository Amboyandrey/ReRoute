package provider

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/Amboyandrey/ReRoute/gateway/internal/config"
)

// OpenAICompatClient talks to any OpenAI-compatible upstream (OpenAI
// itself, Nebius AI Studio, vLLM's OpenAI server). The wire format matches
// the caller's request exactly, so this is a thin passthrough with
// streaming support and usage accounting.
type OpenAICompatClient struct {
	HTTP *http.Client
}

var hopByHopHeaders = map[string]bool{
	"Connection":          true,
	"Keep-Alive":          true,
	"Proxy-Authenticate":  true,
	"Proxy-Authorization": true,
	"Te":                  true,
	"Trailers":            true,
	"Transfer-Encoding":   true,
	"Upgrade":             true,
	"Content-Length":      true,
	"Content-Encoding":    true,
}

func (c *OpenAICompatClient) ChatCompletion(ctx context.Context, tier config.Tier, body []byte, stream bool, w http.ResponseWriter) (Result, error) {
	url := strings.TrimRight(tier.BaseURL, "/") + "/chat/completions"

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return Result{}, fmt.Errorf("building request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if key := tier.APIKey(); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}

	start := time.Now()
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return Result{}, fmt.Errorf("calling upstream %s: %w", tier.Name, err)
	}
	defer drainAndClose(resp.Body)

	result := Result{StatusCode: resp.StatusCode, Latency: time.Since(start)}

	if resp.StatusCode >= 400 {
		copyHeaders(w.Header(), resp.Header, hopByHopHeaders)
		w.WriteHeader(resp.StatusCode)
		buf, _ := readAll(resp.Body)
		_, _ = w.Write(buf)
		return result, fmt.Errorf("upstream %s returned status %d", tier.Name, resp.StatusCode)
	}

	copyHeaders(w.Header(), resp.Header, hopByHopHeaders)

	if !stream {
		w.WriteHeader(resp.StatusCode)
		buf, err := readAll(resp.Body)
		if err != nil {
			return result, err
		}
		_, _ = w.Write(buf)
		usage := parseNonStreamUsage(buf)
		result.PromptTokens = usage.PromptTokens
		result.CompletionTokens = usage.CompletionTokens
		return result, nil
	}

	// Streaming passthrough: forward SSE lines as they arrive, flushing
	// after each event so the client sees tokens incrementally.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(resp.StatusCode)
	flusher, canFlush := w.(http.Flusher)

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	var lastUsage usageInfo
	for scanner.Scan() {
		line := scanner.Text()
		if _, err := fmt.Fprintf(w, "%s\n", line); err != nil {
			return result, err
		}
		if u, ok := parseStreamUsageLine(line); ok {
			lastUsage = u
		}
		if canFlush {
			flusher.Flush()
		}
	}
	result.PromptTokens = lastUsage.PromptTokens
	result.CompletionTokens = lastUsage.CompletionTokens
	return result, scanner.Err()
}

type usageInfo struct {
	PromptTokens     int
	CompletionTokens int
}

func parseNonStreamUsage(body []byte) usageInfo {
	var parsed struct {
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return usageInfo{}
	}
	return usageInfo{PromptTokens: parsed.Usage.PromptTokens, CompletionTokens: parsed.Usage.CompletionTokens}
}

// parseStreamUsageLine looks for the final chunk's usage field, present on
// OpenAI-compatible streams when the client requests
// stream_options.include_usage.
func parseStreamUsageLine(line string) (usageInfo, bool) {
	const prefix = "data: "
	if !strings.HasPrefix(line, prefix) {
		return usageInfo{}, false
	}
	payload := strings.TrimPrefix(line, prefix)
	if payload == "[DONE]" {
		return usageInfo{}, false
	}
	var chunk struct {
		Usage *struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal([]byte(payload), &chunk); err != nil || chunk.Usage == nil {
		return usageInfo{}, false
	}
	return usageInfo{PromptTokens: chunk.Usage.PromptTokens, CompletionTokens: chunk.Usage.CompletionTokens}, true
}
