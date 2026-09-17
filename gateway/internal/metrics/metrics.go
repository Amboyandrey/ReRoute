// Package metrics defines the Prometheus metrics exported by the gateway.
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// RouteDecisions counts routing decisions by chosen tier and reason
	// (e.g. "router", "fallback_default", "router_timeout").
	RouteDecisions = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "reroute_route_decisions_total",
		Help: "Number of routing decisions by tier and reason.",
	}, []string{"tier", "reason"})

	// CostUSDTotal accumulates estimated cost of served requests by tier.
	CostUSDTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "reroute_cost_usd_total",
		Help: "Estimated cost in USD of requests served, by tier.",
	}, []string{"tier"})

	// CostSavedUSDTotal accumulates the estimated savings versus always
	// routing to the strong tier.
	CostSavedUSDTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "reroute_cost_saved_usd_total",
		Help: "Estimated USD saved versus always routing to the strong tier.",
	}, []string{"tier"})

	// UpstreamLatency observes upstream provider call latency.
	UpstreamLatency = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "reroute_upstream_latency_seconds",
		Help:    "Latency of upstream provider calls in seconds.",
		Buckets: prometheus.DefBuckets,
	}, []string{"tier"})

	// Fallbacks counts fallback events when a provider call fails.
	Fallbacks = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "reroute_fallbacks_total",
		Help: "Number of times the gateway fell back to another tier after a provider failure.",
	}, []string{"from_tier", "to_tier"})

	// RateLimited counts requests rejected by the rate limiter.
	RateLimited = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "reroute_rate_limited_total",
		Help: "Number of requests rejected due to per-key rate limiting.",
	}, []string{"api_key_name"})
)
