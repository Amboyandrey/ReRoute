// Package handler implements the OpenAI-compatible HTTP surface of the
// gateway: request parsing, tier selection via the router client, retries,
// and fallback across the tier chain.
package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/Amboyandrey/ReRoute/gateway/internal/config"
	"github.com/Amboyandrey/ReRoute/gateway/internal/metrics"
	"github.com/Amboyandrey/ReRoute/gateway/internal/provider"
	"github.com/Amboyandrey/ReRoute/gateway/internal/ratelimit"
	"github.com/Amboyandrey/ReRoute/gateway/internal/router"
)

// Gateway holds everything a request handler needs.
type Gateway struct {
	Config    *config.Config
	Router    *router.Client
	Providers map[string]provider.Client
	Limiter   *ratelimit.Limiter
	Log       *slog.Logger
}

type incomingChatRequest struct {
	Model    string `json:"model"`
	Stream   bool   `json:"stream"`
	Messages []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
}

// ChatCompletions handles POST /v1/chat/completions.
func (g *Gateway) ChatCompletions(w http.ResponseWriter, r *http.Request) {
	apiKey := extractAPIKey(r)
	if g.Limiter != nil && !g.Limiter.Allow(apiKey) {
		metrics.RateLimited.WithLabelValues(apiKey).Inc()
		http.Error(w, `{"error":{"message":"rate limit exceeded","type":"rate_limit_error"}}`, http.StatusTooManyRequests)
		return
	}

	body, err := readBody(r)
	if err != nil {
		http.Error(w, `{"error":{"message":"failed to read request body"}}`, http.StatusBadRequest)
		return
	}

	var parsed incomingChatRequest
	if err := json.Unmarshal(body, &parsed); err != nil {
		http.Error(w, `{"error":{"message":"invalid JSON body"}}`, http.StatusBadRequest)
		return
	}

	prompt := lastUserMessage(parsed)
	decision := g.Router.Decide(r.Context(), prompt)
	metrics.RouteDecisions.WithLabelValues(decision.Tier, decision.Reason).Inc()

	chain := g.tierChain(decision.Tier)
	if len(chain) == 0 {
		http.Error(w, `{"error":{"message":"no tiers configured"}}`, http.StatusInternalServerError)
		return
	}

	var lastErr error
	for i, tierName := range chain {
		tier, ok := g.Config.TierByName(tierName)
		if !ok {
			continue
		}
		client, ok := g.Providers[tier.Kind]
		if !ok {
			g.Log.Error("no provider client for tier kind", "tier", tier.Name, "kind", tier.Kind)
			continue
		}

		reqBody := body
		if tier.Model != "" {
			reqBody = overrideModel(body, tier.Model)
		}

		buf := newBufferedWriter(w)
		result, err := g.callWithRetries(r.Context(), client, tier, reqBody, parsed.Stream, buf)
		if err == nil {
			buf.Commit()
			strongTier, hasStrong := g.Config.TierByName(g.strongTierName())
			cost := tier.EstimateCost(result.PromptTokens, result.CompletionTokens)
			metrics.CostUSDTotal.WithLabelValues(tier.Name).Add(cost)
			if hasStrong && tier.Name != strongTier.Name {
				strongCost := strongTier.EstimateCost(result.PromptTokens, result.CompletionTokens)
				if strongCost > cost {
					metrics.CostSavedUSDTotal.WithLabelValues(tier.Name).Add(strongCost - cost)
				}
			}
			metrics.UpstreamLatency.WithLabelValues(tier.Name).Observe(result.Latency.Seconds())
			return
		}

		lastErr = err
		g.Log.Warn("upstream call failed", "tier", tier.Name, "error", err)

		if buf.Committed() {
			// Bytes already reached the client (e.g. the stream broke
			// mid-flight); it's too late to retry or fall back.
			return
		}
		if i+1 < len(chain) {
			metrics.Fallbacks.WithLabelValues(tierName, chain[i+1]).Inc()
		}
		// buf discarded: nothing was committed to the real ResponseWriter.
	}

	g.Log.Error("all tiers exhausted", "error", lastErr)
	http.Error(w, `{"error":{"message":"all upstream providers failed","type":"upstream_error"}}`, http.StatusBadGateway)
}

func (g *Gateway) callWithRetries(ctx context.Context, client provider.Client, tier config.Tier, body []byte, stream bool, buf *bufferedWriter) (provider.Result, error) {
	var lastErr error
	retries := g.Config.MaxRetries
	for attempt := 0; attempt <= retries; attempt++ {
		if attempt > 0 {
			buf.Reset()
		}
		result, err := client.ChatCompletion(ctx, tier, body, stream, buf)
		if err == nil {
			return result, nil
		}
		lastErr = err
		if buf.Committed() {
			// Streaming already reached the client; further retries would
			// corrupt the response, so stop here.
			return result, err
		}
		if result.StatusCode != 0 && result.StatusCode < 500 && result.StatusCode != http.StatusTooManyRequests {
			// Client-side error (4xx besides 429): retrying won't help.
			return result, err
		}
		if attempt < retries {
			backoff := time.Duration(attempt+1) * 200 * time.Millisecond
			select {
			case <-time.After(backoff):
			case <-ctx.Done():
				return result, ctx.Err()
			}
		}
	}
	return provider.Result{}, lastErr
}

// tierChain returns the ordered list of tiers to try, starting with the
// chosen tier and then continuing through the configured fallback order.
func (g *Gateway) tierChain(chosen string) []string {
	chain := []string{chosen}
	for _, t := range g.Config.FallbackOrder {
		if t == chosen {
			continue
		}
		chain = append(chain, t)
	}
	return chain
}

func (g *Gateway) strongTierName() string {
	if len(g.Config.FallbackOrder) > 0 {
		return g.Config.FallbackOrder[len(g.Config.FallbackOrder)-1]
	}
	return ""
}

func lastUserMessage(req incomingChatRequest) string {
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == "user" {
			return req.Messages[i].Content
		}
	}
	return ""
}

func extractAPIKey(r *http.Request) string {
	auth := r.Header.Get("Authorization")
	if strings.HasPrefix(auth, "Bearer ") {
		return strings.TrimPrefix(auth, "Bearer ")
	}
	return "anonymous"
}

func overrideModel(body []byte, model string) []byte {
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		return body
	}
	m["model"] = model
	out, err := json.Marshal(m)
	if err != nil {
		return body
	}
	return out
}

func readBody(r *http.Request) ([]byte, error) {
	defer r.Body.Close()
	buf := new(bytes.Buffer)
	if _, err := buf.ReadFrom(r.Body); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
