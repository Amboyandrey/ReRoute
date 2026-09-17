package handler

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Amboyandrey/ReRoute/gateway/internal/config"
	"github.com/Amboyandrey/ReRoute/gateway/internal/provider"
	"github.com/Amboyandrey/ReRoute/gateway/internal/router"
)

// fakeUpstream returns a canned chat-completion JSON response, or an error
// status, depending on failNTimes.
func fakeUpstream(t *testing.T, failNTimes int, successBody string) *httptest.Server {
	t.Helper()
	calls := 0
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls <= failNTimes {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"error":"unavailable"}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(successBody))
	}))
}

func testGateway(t *testing.T, tiers []config.Tier, fallbackOrder []string, providers map[string]provider.Client) *Gateway {
	t.Helper()
	cfg := &config.Config{
		Port:          8080,
		DefaultTier:   fallbackOrder[0],
		FallbackOrder: fallbackOrder,
		MaxRetries:    1,
		Tiers:         tiers,
	}
	return &Gateway{
		Config: cfg,
		Router: &router.Client{
			HTTP:        http.DefaultClient,
			ServiceURL:  "", // no router configured -> falls back to DefaultTier
			DefaultTier: cfg.DefaultTier,
		},
		Providers: providers,
		Log:       slog.New(slog.NewTextHandler(new(strings.Builder), nil)),
	}
}

func TestChatCompletions_HappyPath(t *testing.T) {
	upstream := fakeUpstream(t, 0, `{"id":"1","choices":[{"message":{"role":"assistant","content":"hi"}}],"usage":{"prompt_tokens":5,"completion_tokens":3}}`)
	defer upstream.Close()

	tiers := []config.Tier{{Name: "cheap", Kind: "openai", BaseURL: upstream.URL}}
	gw := testGateway(t, tiers, []string{"cheap"}, provider.New(http.DefaultClient))

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"cheap","messages":[{"role":"user","content":"hello"}]}`))
	rec := httptest.NewRecorder()

	gw.ChatCompletions(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("invalid JSON response: %v", err)
	}
}

func TestChatCompletions_FallsBackOnUpstreamFailure(t *testing.T) {
	deadUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer deadUpstream.Close()

	goodUpstream := fakeUpstream(t, 0, `{"id":"2","choices":[{"message":{"role":"assistant","content":"fallback worked"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	defer goodUpstream.Close()

	tiers := []config.Tier{
		{Name: "cheap", Kind: "openai", BaseURL: deadUpstream.URL},
		{Name: "strong", Kind: "openai", BaseURL: goodUpstream.URL},
	}
	gw := testGateway(t, tiers, []string{"cheap", "strong"}, provider.New(http.DefaultClient))

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"cheap","messages":[{"role":"user","content":"hello"}]}`))
	rec := httptest.NewRecorder()

	gw.ChatCompletions(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 after fallback, got %d: %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "fallback worked") {
		t.Fatalf("expected fallback response body, got: %s", rec.Body.String())
	}
}

func TestChatCompletions_RetriesThenSucceeds(t *testing.T) {
	upstream := fakeUpstream(t, 1, `{"id":"3","choices":[{"message":{"role":"assistant","content":"ok after retry"}}],"usage":{"prompt_tokens":2,"completion_tokens":2}}`)
	defer upstream.Close()

	tiers := []config.Tier{{Name: "cheap", Kind: "openai", BaseURL: upstream.URL}}
	gw := testGateway(t, tiers, []string{"cheap"}, provider.New(http.DefaultClient))

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"cheap","messages":[{"role":"user","content":"hello"}]}`))
	rec := httptest.NewRecorder()

	gw.ChatCompletions(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 after retry, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestChatCompletions_AllTiersFail(t *testing.T) {
	deadUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer deadUpstream.Close()

	tiers := []config.Tier{{Name: "cheap", Kind: "openai", BaseURL: deadUpstream.URL}}
	gw := testGateway(t, tiers, []string{"cheap"}, provider.New(http.DefaultClient))

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"cheap","messages":[{"role":"user","content":"hello"}]}`))
	rec := httptest.NewRecorder()

	gw.ChatCompletions(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected 502 when all tiers fail, got %d", rec.Code)
	}
}

func TestChatCompletions_StreamingPassthrough(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		for _, chunk := range []string{
			`data: {"choices":[{"delta":{"content":"Hel"}}]}`,
			`data: {"choices":[{"delta":{"content":"lo"}}]}`,
			`data: [DONE]`,
		} {
			_, _ = w.Write([]byte(chunk + "\n"))
			flusher.Flush()
		}
	}))
	defer upstream.Close()

	tiers := []config.Tier{{Name: "cheap", Kind: "openai", BaseURL: upstream.URL}}
	gw := testGateway(t, tiers, []string{"cheap"}, provider.New(http.DefaultClient))

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"cheap","stream":true,"messages":[{"role":"user","content":"hello"}]}`))
	rec := httptest.NewRecorder()

	gw.ChatCompletions(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	body := rec.Body.String()
	if !strings.Contains(body, "Hel") || !strings.Contains(body, "[DONE]") {
		t.Fatalf("expected streamed SSE content, got: %q", body)
	}
}
