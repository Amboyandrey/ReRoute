// Package config loads gateway configuration from YAML plus environment
// variable overrides for secrets.
package config

import (
	"fmt"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

// Tier is one routing destination (cheap / mid / strong).
type Tier struct {
	Name      string `yaml:"name"`
	BaseURL   string `yaml:"base_url"`
	Model     string `yaml:"model"`
	APIKeyEnv string `yaml:"api_key_env"`
	// Kind selects the wire protocol for this upstream: "openai" (OpenAI
	// compatible, used by Nebius/vLLM/OpenAI) or "anthropic".
	Kind string `yaml:"kind"`
	// Pricing, USD per 1M tokens, used for cost / cost-saved metrics.
	InputCostPer1M  float64 `yaml:"input_cost_per_1m"`
	OutputCostPer1M float64 `yaml:"output_cost_per_1m"`
}

// EstimateCost returns the USD cost of a call with the given token counts.
func (t Tier) EstimateCost(promptTokens, completionTokens int) float64 {
	return float64(promptTokens)/1_000_000*t.InputCostPer1M +
		float64(completionTokens)/1_000_000*t.OutputCostPer1M
}

// APIKeyConfig configures rate limits for one incoming API key.
type APIKeyConfig struct {
	Key               string `yaml:"key"`
	Name              string `yaml:"name"`
	RequestsPerMinute int    `yaml:"requests_per_minute"`
	BurstSize         int    `yaml:"burst_size"`
}

// Config is the full gateway configuration.
type Config struct {
	Port             int            `yaml:"port"`
	RouterServiceURL string         `yaml:"router_service_url"`
	RouterTimeoutMS  int            `yaml:"router_timeout_ms"`
	DefaultTier      string         `yaml:"default_tier"`
	FallbackOrder    []string       `yaml:"fallback_order"`
	MaxRetries       int            `yaml:"max_retries"`
	Tiers            []Tier         `yaml:"tiers"`
	APIKeys          []APIKeyConfig `yaml:"api_keys"`
}

// RouterTimeout returns the router-service call timeout as a duration.
func (c *Config) RouterTimeout() time.Duration {
	if c.RouterTimeoutMS <= 0 {
		return 150 * time.Millisecond
	}
	return time.Duration(c.RouterTimeoutMS) * time.Millisecond
}

// TierByName returns the tier config with the given name.
func (c *Config) TierByName(name string) (Tier, bool) {
	for _, t := range c.Tiers {
		if t.Name == name {
			return t, true
		}
	}
	return Tier{}, false
}

// APIKey resolves the tier's provider API key from its configured env var.
func (t Tier) APIKey() string {
	if t.APIKeyEnv == "" {
		return ""
	}
	return os.Getenv(t.APIKeyEnv)
}

// Load reads and parses a YAML config file.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading config %s: %w", path, err)
	}
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parsing config %s: %w", path, err)
	}
	if cfg.Port == 0 {
		cfg.Port = 8080
	}
	if cfg.MaxRetries == 0 {
		cfg.MaxRetries = 2
	}
	return &cfg, nil
}
