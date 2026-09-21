package embedding

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestOpenAIEmbedderBatchEmbedOmitsDimensionsByDefault(t *testing.T) {
	requestBody := captureOpenAIEmbeddingRequest(t, "text-embedding-3-small", 256, false)

	if _, ok := requestBody["dimensions"]; ok {
		t.Fatalf("expected request body to omit dimensions by default, got %v", requestBody)
	}
}

func TestOpenAIEmbedderBatchEmbedSendsDimensionsWhenOverrideEnabled(t *testing.T) {
	requestBody := captureOpenAIEmbeddingRequest(t, "text-embedding-3-small", 256, true)

	got, ok := requestBody["dimensions"]
	if !ok {
		t.Fatalf("expected request body to include dimensions, got %v", requestBody)
	}
	if got != float64(256) {
		t.Fatalf("unexpected dimensions value: got %v want 256", got)
	}
}

func TestOpenAIEmbedderBatchEmbedOmitsDimensionsForOpenAICompatibleModels(t *testing.T) {
	requestBody := captureOpenAIEmbeddingRequest(t, "text-embedding-v3", 1024, false)

	if _, ok := requestBody["dimensions"]; ok {
		t.Fatalf("expected request body to omit dimensions for OpenAI-compatible model, got %v", requestBody)
	}
}

func TestOpenAIEmbedderBatchEmbedOmitsDimensionsForFixedSizeModels(t *testing.T) {
	requestBody := captureOpenAIEmbeddingRequest(t, "text-embedding-ada-002", 1536, false)

	if _, ok := requestBody["dimensions"]; ok {
		t.Fatalf("expected request body to omit dimensions for fixed-size model, got %v", requestBody)
	}
}

func captureOpenAIEmbeddingRequest(t *testing.T, modelName string, dimensions int, supportsDimensionOverride bool) map[string]any {
	t.Helper()
	t.Setenv("SSRF_WHITELIST", "127.0.0.1")

	requestBody := map[string]any{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Fatalf("unexpected request path: %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
			t.Fatalf("decode request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"embedding":[0.1,0.2],"index":0}]}`))
	}))
	defer server.Close()

	embedder, err := NewOpenAIEmbedder(
		"test-key",
		server.URL,
		modelName,
		511,
		dimensions,
		"8f7d6082-5a15-4f84-ae55-88b2bdac4ba0",
		nil,
	)
	if err != nil {
		t.Fatalf("NewOpenAIEmbedder: %v", err)
	}
	embedder.SetSupportsDimensionOverride(supportsDimensionOverride)

	if _, err := embedder.BatchEmbed(context.Background(), []string{"hello"}); err != nil {
		t.Fatalf("BatchEmbed: %v", err)
	}

	return requestBody
}

// TestClassifyEmbeddingInput pins the pre-flight diagnostic to a TOKEN estimate.
//
// The previous implementation compared len(text) (a byte count) against 8192
// (a token budget), so every long CJK chunk tripped it. Production logged
// "OpenAIEmbedder BatchEmbed input[0]: INVALID length=16262 (must be
// [1, 8192])" for a chunk that embedded fine, and the noise buried real
// failures. The fixture below reproduces that exact 16262-byte Chinese input.
func TestClassifyEmbeddingInput(t *testing.T) {
	cjk := strings.Repeat("缸体或飞轮壳维修时，维修后状态检查。", 300)
	if len(cjk) < 16000 || len(cjk) > 16600 {
		t.Fatalf("fixture is %d bytes; it must reproduce the ~16262-byte production input", len(cjk))
	}

	lengthyEnglish := strings.Repeat("the quick brown fox jumps over the lazy dog ", 1000)

	cases := []struct {
		name           string
		text           string
		wantEmpty      bool
		wantOverWindow bool
	}{
		{
			name:      "empty input is a hard error",
			text:      "",
			wantEmpty: true,
		},
		{
			name: "short english is inside the window",
			text: "hello world",
		},
		{
			// Regression: 16262 bytes of Chinese is roughly 3.2k tokens, far
			// inside the 8192-token window. It must not be reported.
			name: "long cjk stays inside the window",
			text: cjk,
		},
		{
			name:           "very long english exceeds the window",
			text:           lengthyEnglish,
			wantOverWindow: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			stat := classifyEmbeddingInput(tc.text, maxInputTokens)

			if stat.Chars != len(tc.text) {
				t.Errorf("Chars = %d, want %d", stat.Chars, len(tc.text))
			}
			if stat.Empty != tc.wantEmpty {
				t.Errorf("Empty = %v, want %v", stat.Empty, tc.wantEmpty)
			}
			if stat.OverWindow != tc.wantOverWindow {
				t.Errorf("OverWindow = %v, want %v (tokens=%d chars=%d lang=%s)",
					stat.OverWindow, tc.wantOverWindow, stat.Tokens, stat.Chars, stat.Lang)
			}
			if tc.wantEmpty {
				if stat.Tokens != 0 || stat.Lang != "" {
					t.Errorf("empty input must not be estimated: %+v", stat)
				}
				return
			}
			if stat.Tokens <= 0 {
				t.Errorf("Tokens = %d, want > 0", stat.Tokens)
			}
		})
	}
}
