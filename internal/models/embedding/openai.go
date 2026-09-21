package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/Tencent/WeKnora/internal/logger"
	"github.com/Tencent/WeKnora/internal/textmetrics"
	secutils "github.com/Tencent/WeKnora/internal/utils"
)

// OpenAIEmbedder implements text vectorization functionality using OpenAI API
type OpenAIEmbedder struct {
	apiKey                    string
	baseURL                   string
	modelName                 string
	truncatePromptTokens      int
	dimensions                int
	modelID                   string
	httpClient                *http.Client
	timeout                   time.Duration
	maxRetries                int
	customHeaders             map[string]string
	supportsDimensionOverride bool
	EmbedderPooler
}

// maxInputTokens is the token window the pre-flight diagnostic compares
// against. It matches the common ceiling for OpenAI-compatible embedding
// models (OpenAI text-embedding-3-*, Bailian text-embedding-v3/v4). It is a
// reporting threshold only: it never blocks, truncates, or rewrites a request.
const maxInputTokens = 8192

// embeddingInputStat is the per-input diagnostic computed before a batch is
// sent. Tokens is an approximation (see internal/textmetrics), not
// a tokenizer count.
type embeddingInputStat struct {
	Tokens     int
	Chars      int
	Lang       string
	Empty      bool
	OverWindow bool
}

// classifyEmbeddingInput estimates the token footprint of one embedding input
// so the caller can report over-window inputs in the unit the provider actually
// enforces. It performs no I/O and no logging, so it can be unit tested without
// an HTTP server.
func classifyEmbeddingInput(text string, maxTokens int) embeddingInputStat {
	stat := embeddingInputStat{Chars: len(text)}
	if text == "" {
		stat.Empty = true
		return stat
	}
	stat.Lang = textmetrics.DetectLanguage(text)
	stat.Tokens = textmetrics.ApproxTokenCount(text, stat.Lang)
	if maxTokens > 0 && stat.Tokens > maxTokens {
		stat.OverWindow = true
	}
	return stat
}

// OpenAIEmbedRequest represents an OpenAI embedding request
type OpenAIEmbedRequest struct {
	Model                string   `json:"model"`
	Input                []string `json:"input"`
	EncodingFormat       string   `json:"encoding_format,omitempty"`
	Dimensions           int      `json:"dimensions,omitempty"`
	TruncatePromptTokens int      `json:"truncate_prompt_tokens,omitempty"`
}

// OpenAIEmbedResponse represents an OpenAI embedding response
type OpenAIEmbedResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
}

// NewOpenAIEmbedder creates a new OpenAI embedder
func NewOpenAIEmbedder(apiKey, baseURL, modelName string,
	truncatePromptTokens int, dimensions int, modelID string, pooler EmbedderPooler,
) (*OpenAIEmbedder, error) {
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}

	if modelName == "" {
		return nil, fmt.Errorf("model name is required")
	}

	if truncatePromptTokens == 0 {
		truncatePromptTokens = 511
	}

	timeout := 60 * time.Second

	if err := validateEmbeddingBaseURL(baseURL); err != nil {
		return nil, err
	}

	return &OpenAIEmbedder{
		apiKey:               apiKey,
		baseURL:              baseURL,
		modelName:            modelName,
		httpClient:           newEmbeddingHTTPClient(timeout),
		truncatePromptTokens: truncatePromptTokens,
		EmbedderPooler:       pooler,
		dimensions:           dimensions,
		modelID:              modelID,
		timeout:              timeout,
		maxRetries:           3, // Maximum retry count
	}, nil
}

// SetCustomHeaders 设置用户自定义 HTTP 请求头（类似 OpenAI Python SDK 的 extra_headers）。
// 保留头（Authorization、Content-Type 等）会在发送时被自动跳过。
func (e *OpenAIEmbedder) SetCustomHeaders(headers map[string]string) {
	e.customHeaders = headers
}

func (e *OpenAIEmbedder) SetSupportsDimensionOverride(supported bool) {
	e.supportsDimensionOverride = supported
}

// Embed converts text to vector
func (e *OpenAIEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	for range 3 {
		embeddings, err := e.BatchEmbed(ctx, []string{text})
		if err != nil {
			return nil, err
		}
		if len(embeddings) > 0 {
			return embeddings[0], nil
		}
	}
	return nil, fmt.Errorf("no embedding returned")
}

func (e *OpenAIEmbedder) doRequestWithRetry(ctx context.Context, jsonData []byte) (*http.Response, error) {
	var resp *http.Response
	var err error
	url := e.baseURL + "/embeddings"

	for i := 0; i <= e.maxRetries; i++ {
		if i > 0 {
			backoffTime := time.Duration(1<<uint(i-1)) * time.Second
			if backoffTime > 10*time.Second {
				backoffTime = 10 * time.Second
			}
			logger.GetLogger(ctx).
				Infof("OpenAIEmbedder retrying request (%d/%d), waiting %v", i, e.maxRetries, backoffTime)

			select {
			case <-time.After(backoffTime):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		// Rebuild request each time to ensure Body is valid.
		// IMPORTANT: declare `req` separately (var) so the assignment to `err`
		// below uses the outer-scope variable, not a fresh loop-local one.
		// Previously this read `req, err := http.NewRequestWithContext(...)`,
		// where `:=` introduced a new `err` shadowing the outer one. The
		// `resp, err = httpClient.Do(req)` line then wrote to the shadowed
		// `err` only, so when all retries failed with connection errors the
		// outer `err` stayed nil. The function returned `(nil, nil)`, and
		// callers (BatchEmbed line 195) blindly dereferenced `resp.Body` →
		// SIGSEGV nil-pointer panic that took down the whole process.
		// Reproduce: stop the embedding upstream (e.g. localhost:3130), make
		// any RAG query → backend SIGSEGV instead of returning HTTP 500.
		var req *http.Request
		req, err = http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonData))
		if err != nil {
			logger.GetLogger(ctx).Errorf("OpenAIEmbedder failed to create request: %v", err)
			continue
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+e.apiKey)
		secutils.ApplyCustomHeaders(req, e.customHeaders)

		resp, err = e.httpClient.Do(req)
		if err == nil {
			return resp, nil
		}

		logger.GetLogger(ctx).Errorf("OpenAIEmbedder request failed (attempt %d/%d): %v", i+1, e.maxRetries+1, err)
	}

	return nil, err
}

func (e *OpenAIEmbedder) BatchEmbed(ctx context.Context, texts []string) ([][]float32, error) {
	// Create request body
	reqBody := OpenAIEmbedRequest{
		Model:                e.modelName,
		Input:                texts,
		EncodingFormat:       "float",
		TruncatePromptTokens: e.truncatePromptTokens,
	}
	if e.supportsDimensionsParam() {
		reqBody.Dimensions = e.dimensions
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		logger.GetLogger(ctx).Errorf("OpenAIEmbedder EmbedBatch marshal request error: %v", err)
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	// Log request details for debugging
	logger.GetLogger(ctx).Debugf("OpenAIEmbedder BatchEmbed: model=%s, input_count=%d, truncate_tokens=%d",
		e.modelName, len(texts), e.truncatePromptTokens)

	// Diagnostics only — the request is sent either way. Two cases are worth
	// reporting: an empty input, which the provider always rejects, and an
	// input past the model's token window, which it may reject or silently
	// truncate.
	//
	// Length is reported as an estimated TOKEN count, never as len(text). A
	// byte count compared against 8192 tokens logged a false
	// "INVALID length=16262 (must be [1, 8192])" for every long CJK chunk and
	// buried real failures in noise: 16262 bytes of Chinese is roughly 5.4k
	// tokens, comfortably inside an 8192-token window.
	emptyInputs := 0
	overWindowInputs := 0
	for i, text := range texts {
		stat := classifyEmbeddingInput(text, maxInputTokens)

		textPreview := text
		if len(textPreview) > 200 {
			textPreview = textPreview[:200] + "..."
		}

		switch {
		case stat.Empty:
			emptyInputs++
			logger.GetLogger(ctx).Errorf(
				"OpenAIEmbedder BatchEmbed input[%d]: empty input, the provider will reject it", i)
		case stat.OverWindow:
			overWindowInputs++
			logger.GetLogger(ctx).Warnf(
				"OpenAIEmbedder BatchEmbed input[%d]: ~%d tokens (chars=%d lang=%s) exceeds the %d-token window; the provider may reject or truncate it, preview=%s",
				i, stat.Tokens, stat.Chars, stat.Lang, maxInputTokens, textPreview)
		default:
			logger.GetLogger(ctx).Debugf(
				"OpenAIEmbedder BatchEmbed input[%d]: ~%d tokens (chars=%d lang=%s), preview=%s",
				i, stat.Tokens, stat.Chars, stat.Lang, textPreview)
		}
	}

	if emptyInputs > 0 {
		logger.GetLogger(ctx).Errorf(
			"OpenAIEmbedder BatchEmbed: %d empty input(s); the provider will reject the request", emptyInputs)
	}
	if overWindowInputs > 0 {
		logger.GetLogger(ctx).Warnf(
			"OpenAIEmbedder BatchEmbed: %d input(s) exceed the estimated %d-token window and may be rejected or truncated",
			overWindowInputs, maxInputTokens)
	}

	// Send request (passing jsonData instead of constructing http.Request)
	resp, err := e.doRequestWithRetry(ctx, jsonData)
	if err != nil {
		logger.GetLogger(ctx).Errorf("OpenAIEmbedder EmbedBatch send request error: %v", err)
		return nil, fmt.Errorf("send request: %w", err)
	}
	if resp.Body != nil {
		defer resp.Body.Close()
	}

	// Read response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		logger.GetLogger(ctx).Errorf("OpenAIEmbedder EmbedBatch read response error: %v", err)
		return nil, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		// Log detailed error response from OpenAI API
		bodyStr := string(body)
		if len(bodyStr) > 1000 {
			bodyStr = bodyStr[:1000] + "... (truncated)"
		}
		logger.GetLogger(ctx).Errorf("OpenAIEmbedder EmbedBatch API error: Http Status %s, Response Body: %s", resp.Status, bodyStr)
		return nil, fmt.Errorf("EmbedBatch API error: Http Status %s, Response: %s", resp.Status, bodyStr)
	}

	// Parse response
	var response OpenAIEmbedResponse
	if err := json.Unmarshal(body, &response); err != nil {
		logger.GetLogger(ctx).Errorf("OpenAIEmbedder EmbedBatch unmarshal response error: %v", err)
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}

	// Extract embedding vectors
	embeddings := make([][]float32, 0, len(response.Data))
	for _, data := range response.Data {
		embeddings = append(embeddings, data.Embedding)
	}

	return embeddings, nil
}

// GetModelName returns the model name
func (e *OpenAIEmbedder) GetModelName() string {
	return e.modelName
}

func (e *OpenAIEmbedder) supportsDimensionsParam() bool {
	return e.supportsDimensionOverride && e.dimensions > 0
}

// GetDimensions returns the vector dimensions
func (e *OpenAIEmbedder) GetDimensions() int {
	return e.dimensions
}

// GetModelID returns the model ID
func (e *OpenAIEmbedder) GetModelID() string {
	return e.modelID
}
