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

// AnthropicClient translates OpenAI-format chat-completions requests to
// Anthropic's Messages API and translates the response back, so callers
// see an OpenAI-shaped response regardless of which tier served it.
//
// Scope note: this covers plain text conversations (system + user/assistant
// messages, streaming and non-streaming). Tool calls and multimodal content
// are not translated yet — see TODOs below.
type AnthropicClient struct {
	HTTP *http.Client
}

const anthropicVersion = "2023-06-01"

type openAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIRequest struct {
	Model       string          `json:"model"`
	Messages    []openAIMessage `json:"messages"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Temperature *float64        `json:"temperature,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type anthropicRequest struct {
	Model       string             `json:"model"`
	System      string             `json:"system,omitempty"`
	Messages    []anthropicMessage `json:"messages"`
	MaxTokens   int                `json:"max_tokens"`
	Temperature *float64           `json:"temperature,omitempty"`
	Stream      bool               `json:"stream,omitempty"`
}

func toAnthropicRequest(oaiBody []byte, model string) ([]byte, bool, error) {
	var req openAIRequest
	if err := json.Unmarshal(oaiBody, &req); err != nil {
		return nil, false, fmt.Errorf("decoding OpenAI request: %w", err)
	}

	aReq := anthropicRequest{
		Model:       model,
		Temperature: req.Temperature,
		Stream:      req.Stream,
		MaxTokens:   req.MaxTokens,
	}
	if aReq.MaxTokens == 0 {
		aReq.MaxTokens = 4096 // Anthropic requires max_tokens; OpenAI does not.
	}

	for _, m := range req.Messages {
		if m.Role == "system" {
			if aReq.System != "" {
				aReq.System += "\n\n"
			}
			aReq.System += m.Content
			continue
		}
		role := m.Role
		if role != "user" && role != "assistant" {
			role = "user"
		}
		aReq.Messages = append(aReq.Messages, anthropicMessage{Role: role, Content: m.Content})
	}

	out, err := json.Marshal(aReq)
	return out, req.Stream, err
}

type anthropicNonStreamResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	StopReason string `json:"stop_reason"`
	Usage      struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

func toOpenAIResponse(body []byte) ([]byte, usageInfo, error) {
	var a anthropicNonStreamResponse
	if err := json.Unmarshal(body, &a); err != nil {
		return nil, usageInfo{}, fmt.Errorf("decoding Anthropic response: %w", err)
	}
	var text strings.Builder
	for _, block := range a.Content {
		if block.Type == "text" {
			text.WriteString(block.Text)
		}
	}

	resp := map[string]any{
		"id":     a.ID,
		"object": "chat.completion",
		"model":  a.Model,
		"choices": []map[string]any{
			{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": text.String(),
				},
				"finish_reason": mapStopReason(a.StopReason),
			},
		},
		"usage": map[string]any{
			"prompt_tokens":     a.Usage.InputTokens,
			"completion_tokens": a.Usage.OutputTokens,
			"total_tokens":      a.Usage.InputTokens + a.Usage.OutputTokens,
		},
	}
	out, err := json.Marshal(resp)
	usage := usageInfo{PromptTokens: a.Usage.InputTokens, CompletionTokens: a.Usage.OutputTokens}
	return out, usage, err
}

func mapStopReason(r string) string {
	switch r {
	case "end_turn", "stop_sequence":
		return "stop"
	case "max_tokens":
		return "length"
	default:
		return r
	}
}

func (c *AnthropicClient) ChatCompletion(ctx context.Context, tier config.Tier, body []byte, stream bool, w http.ResponseWriter) (Result, error) {
	aBody, _, err := toAnthropicRequest(body, tier.Model)
	if err != nil {
		return Result{}, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(tier.BaseURL, "/")+"/v1/messages", bytes.NewReader(aBody))
	if err != nil {
		return Result{}, fmt.Errorf("building request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("anthropic-version", anthropicVersion)
	if key := tier.APIKey(); key != "" {
		req.Header.Set("x-api-key", key)
	}

	start := time.Now()
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return Result{}, fmt.Errorf("calling upstream %s: %w", tier.Name, err)
	}
	defer drainAndClose(resp.Body)

	result := Result{StatusCode: resp.StatusCode, Latency: time.Since(start)}

	if resp.StatusCode >= 400 {
		buf, _ := readAll(resp.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		_, _ = w.Write(buf)
		return result, fmt.Errorf("upstream %s returned status %d", tier.Name, resp.StatusCode)
	}

	if !stream {
		buf, err := readAll(resp.Body)
		if err != nil {
			return result, err
		}
		oaiBody, usage, err := toOpenAIResponse(buf)
		if err != nil {
			return result, err
		}
		result.PromptTokens = usage.PromptTokens
		result.CompletionTokens = usage.CompletionTokens
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, err = w.Write(oaiBody)
		return result, err
	}

	// Streaming: translate Anthropic SSE events into OpenAI-style
	// chat.completion.chunk events.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	flusher, canFlush := w.(http.Flusher)

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	var usage usageInfo
	chunkID := "chatcmpl-stream"
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")

		var evt struct {
			Type  string `json:"type"`
			Delta struct {
				Text string `json:"text"`
			} `json:"delta"`
			Message struct {
				ID    string `json:"id"`
				Usage struct {
					InputTokens int `json:"input_tokens"`
				} `json:"usage"`
			} `json:"message"`
			Usage struct {
				OutputTokens int `json:"output_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(payload), &evt); err != nil {
			continue
		}

		switch evt.Type {
		case "message_start":
			if evt.Message.ID != "" {
				chunkID = evt.Message.ID
			}
			usage.PromptTokens = evt.Message.Usage.InputTokens
		case "content_block_delta":
			if evt.Delta.Text == "" {
				continue
			}
			chunk := map[string]any{
				"id":     chunkID,
				"object": "chat.completion.chunk",
				"choices": []map[string]any{
					{"index": 0, "delta": map[string]any{"content": evt.Delta.Text}},
				},
			}
			out, _ := json.Marshal(chunk)
			if _, err := fmt.Fprintf(w, "data: %s\n\n", out); err != nil {
				return result, err
			}
			if canFlush {
				flusher.Flush()
			}
		case "message_delta":
			usage.CompletionTokens = evt.Usage.OutputTokens
		case "message_stop":
			if _, err := fmt.Fprintf(w, "data: [DONE]\n\n"); err != nil {
				return result, err
			}
			if canFlush {
				flusher.Flush()
			}
		}
	}
	result.PromptTokens = usage.PromptTokens
	result.CompletionTokens = usage.CompletionTokens
	return result, scanner.Err()
}
