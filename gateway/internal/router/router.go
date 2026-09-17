// Package router calls the Python router service to decide which tier a
// prompt should go to, falling back to a static default when the service
// is slow, down, or errors — the gateway must never become unavailable
// because the ML path is unavailable.
package router

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// Decision is the routing outcome for one request.
type Decision struct {
	Tier   string
	Reason string // "router" | "fallback_default" | "router_timeout" | "router_error"
}

// Client calls the router service.
type Client struct {
	HTTP        *http.Client
	ServiceURL  string
	Timeout     time.Duration
	DefaultTier string
}

type decideRequest struct {
	Prompt string `json:"prompt"`
}

type decideResponse struct {
	Tier       string  `json:"tier"`
	Confidence float64 `json:"confidence"`
}

// Decide asks the router service which tier should handle prompt. On any
// failure (timeout, non-200, malformed body) it falls back to the
// configured default tier rather than erroring the request.
func (c *Client) Decide(ctx context.Context, prompt string) Decision {
	if c.ServiceURL == "" {
		return Decision{Tier: c.DefaultTier, Reason: "no_router_configured"}
	}

	ctx, cancel := context.WithTimeout(ctx, c.Timeout)
	defer cancel()

	body, err := json.Marshal(decideRequest{Prompt: prompt})
	if err != nil {
		return Decision{Tier: c.DefaultTier, Reason: "fallback_default"}
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.ServiceURL+"/route", bytes.NewReader(body))
	if err != nil {
		return Decision{Tier: c.DefaultTier, Reason: "fallback_default"}
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTP.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return Decision{Tier: c.DefaultTier, Reason: "router_timeout"}
		}
		return Decision{Tier: c.DefaultTier, Reason: "router_error"}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return Decision{Tier: c.DefaultTier, Reason: "router_error"}
	}

	var out decideResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil || out.Tier == "" {
		return Decision{Tier: c.DefaultTier, Reason: "router_error"}
	}

	return Decision{Tier: out.Tier, Reason: "router"}
}

// String is a convenience for logging.
func (d Decision) String() string {
	return fmt.Sprintf("tier=%s reason=%s", d.Tier, d.Reason)
}
