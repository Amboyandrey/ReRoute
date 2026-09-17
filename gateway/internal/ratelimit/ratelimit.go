// Package ratelimit implements a simple per-key token-bucket rate limiter.
package ratelimit

import (
	"sync"
	"time"
)

// bucket is a token bucket for a single API key.
type bucket struct {
	mu         sync.Mutex
	tokens     float64
	maxTokens  float64
	refillRate float64 // tokens per second
	lastRefill time.Time
}

func (b *bucket) allow() bool {
	b.mu.Lock()
	defer b.mu.Unlock()

	now := time.Now()
	elapsed := now.Sub(b.lastRefill).Seconds()
	b.lastRefill = now

	b.tokens += elapsed * b.refillRate
	if b.tokens > b.maxTokens {
		b.tokens = b.maxTokens
	}
	if b.tokens < 1 {
		return false
	}
	b.tokens--
	return true
}

// Limiter manages one token bucket per API key.
type Limiter struct {
	mu      sync.RWMutex
	buckets map[string]*bucket
	// defaults used for keys with no explicit config
	defaultRPM   int
	defaultBurst int
	configs      map[string]KeyConfig
}

// KeyConfig configures the limiter for one key.
type KeyConfig struct {
	RequestsPerMinute int
	BurstSize         int
}

// New creates a Limiter. defaultRPM/defaultBurst apply to any key not
// present in configs.
func New(defaultRPM, defaultBurst int, configs map[string]KeyConfig) *Limiter {
	if defaultRPM <= 0 {
		defaultRPM = 60
	}
	if defaultBurst <= 0 {
		defaultBurst = defaultRPM
	}
	return &Limiter{
		buckets:      make(map[string]*bucket),
		defaultRPM:   defaultRPM,
		defaultBurst: defaultBurst,
		configs:      configs,
	}
}

// Allow reports whether a request for the given API key may proceed,
// consuming a token if so.
func (l *Limiter) Allow(apiKey string) bool {
	l.mu.RLock()
	b, ok := l.buckets[apiKey]
	l.mu.RUnlock()
	if ok {
		return b.allow()
	}

	l.mu.Lock()
	defer l.mu.Unlock()
	// double-check after acquiring write lock
	if b, ok = l.buckets[apiKey]; ok {
		return b.allow()
	}

	rpm, burst := l.defaultRPM, l.defaultBurst
	if cfg, ok := l.configs[apiKey]; ok {
		if cfg.RequestsPerMinute > 0 {
			rpm = cfg.RequestsPerMinute
		}
		if cfg.BurstSize > 0 {
			burst = cfg.BurstSize
		}
	}

	b = &bucket{
		tokens:     float64(burst),
		maxTokens:  float64(burst),
		refillRate: float64(rpm) / 60.0,
		lastRefill: time.Now(),
	}
	l.buckets[apiKey] = b
	return b.allow()
}
