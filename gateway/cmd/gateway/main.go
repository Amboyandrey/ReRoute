// Command gateway runs the ReRoute OpenAI-compatible data-plane server.
package main

import (
	"encoding/json"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/Amboyandrey/ReRoute/gateway/internal/config"
	"github.com/Amboyandrey/ReRoute/gateway/internal/handler"
	"github.com/Amboyandrey/ReRoute/gateway/internal/provider"
	"github.com/Amboyandrey/ReRoute/gateway/internal/ratelimit"
	"github.com/Amboyandrey/ReRoute/gateway/internal/router"
)

func main() {
	configPath := flag.String("config", "config.yaml", "path to gateway config YAML")
	flag.Parse()

	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Error("failed to load config", "error", err)
		os.Exit(1)
	}

	httpClient := &http.Client{Timeout: 120 * time.Second}

	keyConfigs := make(map[string]ratelimit.KeyConfig, len(cfg.APIKeys))
	for _, k := range cfg.APIKeys {
		keyConfigs[k.Key] = ratelimit.KeyConfig{
			RequestsPerMinute: k.RequestsPerMinute,
			BurstSize:         k.BurstSize,
		}
	}

	gw := &handler.Gateway{
		Config: cfg,
		Router: &router.Client{
			HTTP:        httpClient,
			ServiceURL:  cfg.RouterServiceURL,
			Timeout:     cfg.RouterTimeout(),
			DefaultTier: cfg.DefaultTier,
		},
		Providers: provider.New(httpClient),
		Limiter:   ratelimit.New(60, 60, keyConfigs),
		Log:       log,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", gw.ChatCompletions)
	mux.HandleFunc("GET /v1/models", modelsHandler(cfg))
	mux.HandleFunc("GET /healthz", healthHandler)
	mux.Handle("GET /metrics", promhttp.Handler())

	addr := ":" + strconv.Itoa(cfg.Port)
	log.Info("starting gateway", "addr", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Error("server stopped", "error", err)
		os.Exit(1)
	}
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{"status":"ok"}`))
}

func modelsHandler(cfg *config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		type modelEntry struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			OwnedBy string `json:"owned_by"`
		}
		data := make([]modelEntry, 0, len(cfg.Tiers))
		for _, t := range cfg.Tiers {
			data = append(data, modelEntry{ID: t.Name, Object: "model", OwnedBy: "reroute"})
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
	}
}
