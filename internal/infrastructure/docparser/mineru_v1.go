package docparser

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"path"
	"strings"
	"time"

	"github.com/Tencent/WeKnora/internal/logger"
	"github.com/Tencent/WeKnora/internal/utils"
)

// MinerU 4.0 replaced the legacy /file_parse endpoint with the V1 API:
// upload a file, submit a parse job, then download the produced artifacts.
// These helpers talk to that API; the legacy converter stays for MinerU <= 3.x.
const (
	mineruV1HealthPath   = "/v1/health"
	mineruV1UploadsPath  = "/v1/uploads"
	mineruV1JobsPath     = "/v1/parse/jobs"
	mineruV1FilesPath    = "/v1/files"
	mineruV1ProbeTimeout = 8 * time.Second
	mineruV1PollInterval = 2 * time.Second
	mineruV1DefaultTier  = "standard"
)

// mineruV1Tiers lists the parsing tiers introduced by MinerU 4.0.
var mineruV1Tiers = map[string]struct{}{
	"flash":    {},
	"basic":    {},
	"standard": {},
	"advanced": {},
}

// resolveMineruTier normalises a tier value. Legacy backend names (pipeline,
// vlm-*, hybrid-*) are not tiers in 4.0, so anything unknown falls back to
// the documented default reading quality.
func resolveMineruTier(raw string) string {
	tier := strings.ToLower(strings.TrimSpace(raw))
	if _, ok := mineruV1Tiers[tier]; ok {
		return tier
	}
	return mineruV1DefaultTier
}

// mineruV1MimeType maps a file extension to the MIME type MinerU expects.
func mineruV1MimeType(fileName, fileType string) string {
	ext := strings.ToLower(strings.TrimSpace(fileType))
	ext = strings.TrimPrefix(ext, ".")
	if ext == "" {
		ext = strings.ToLower(strings.TrimPrefix(path.Ext(fileName), "."))
	}
	switch ext {
	case "pdf":
		return "application/pdf"
	case "doc":
		return "application/msword"
	case "docx":
		return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
	case "ppt":
		return "application/vnd.ms-powerpoint"
	case "pptx":
		return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
	case "jpg", "jpeg":
		return "image/jpeg"
	case "png":
		return "image/png"
	case "bmp":
		return "image/bmp"
	case "tif", "tiff":
		return "image/tiff"
	case "":
		return "application/octet-stream"
	}
	if mt := mime.TypeByExtension("." + ext); mt != "" {
		if idx := strings.Index(mt, ";"); idx >= 0 {
			mt = strings.TrimSpace(mt[:idx])
		}
		return mt
	}
	return "application/octet-stream"
}

type mineruV1FileRef struct {
	ID       string `json:"id"`
	Filename string `json:"filename"`
	Bytes    int    `json:"bytes"`
}

type mineruV1UploadResponse struct {
	ID            string            `json:"id"`
	Status        string            `json:"status"`
	UploadURL     string            `json:"upload_url"`
	UploadMethod  string            `json:"upload_method"`
	UploadHeaders map[string]string `json:"upload_headers"`
	File          *mineruV1FileRef  `json:"file"`
}

type mineruV1Source struct {
	Type   string `json:"type"`
	FileID string `json:"file_id"`
}

type mineruV1JobFileEntry struct {
	Source    mineruV1Source `json:"source"`
	PageRange string         `json:"page_range,omitempty"`
}

type mineruV1CreateJobRequest struct {
	Files         []mineruV1JobFileEntry `json:"files"`
	Tier          string                 `json:"tier,omitempty"`
	OCRMode       string                 `json:"ocr_mode,omitempty"`
	OutputFormats []string               `json:"output_formats"`
}

type mineruV1OutputRef struct {
	FileID string `json:"file_id"`
	Bytes  int    `json:"bytes"`
}

type mineruV1OutputFiles struct {
	Markdown *mineruV1OutputRef `json:"markdown"`
	Zip      *mineruV1OutputRef `json:"zip"`
}

type mineruV1JobFileError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type mineruV1JobFileResult struct {
	FileID      string                `json:"file_id"`
	Name        string                `json:"name"`
	Status      string                `json:"status"`
	OutputFiles *mineruV1OutputFiles  `json:"output_files"`
	Error       *mineruV1JobFileError `json:"error"`
}

type mineruV1JobResponse struct {
	JobID  string                  `json:"job_id"`
	Status string                  `json:"status"`
	Files  []mineruV1JobFileResult `json:"files"`
}

// mineruV1Client is a small client for the MinerU >= 4.0 V1 API.
type mineruV1Client struct {
	endpoint string
	apiKey   string
	tier     string
}

func newMineruV1Client(endpoint, apiKey, tier string) *mineruV1Client {
	return &mineruV1Client{
		endpoint: strings.TrimRight(strings.TrimSpace(endpoint), "/"),
		apiKey:   strings.TrimSpace(apiKey),
		tier:     resolveMineruTier(tier),
	}
}

func (c *mineruV1Client) httpClient(timeout time.Duration) *http.Client {
	return utils.NewSSRFSafeHTTPClient(utils.SSRFSafeHTTPClientConfig{
		Timeout:      timeout,
		MaxRedirects: 5,
	})
}

func effectiveURLPort(u *url.URL) string {
	if port := u.Port(); port != "" {
		return port
	}
	if strings.EqualFold(u.Scheme, "https") {
		return "443"
	}
	return "80"
}

func sameOrigin(a, b *url.URL) bool {
	return strings.EqualFold(a.Scheme, b.Scheme) &&
		strings.EqualFold(a.Hostname(), b.Hostname()) &&
		effectiveURLPort(a) == effectiveURLPort(b)
}

// applyAuth attaches the optional API key, but only for same-origin requests so
// the key never leaks to a third-party upload target.
func (c *mineruV1Client) applyAuth(req *http.Request, target *url.URL) {
	if c.apiKey == "" || target == nil {
		return
	}
	base, err := url.Parse(c.endpoint)
	if err != nil || !sameOrigin(base, target) {
		return
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
}

func (c *mineruV1Client) doJSON(ctx context.Context, method, target string, body, out interface{}) error {
	var payload io.Reader
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("encode request: %w", err)
		}
		payload = bytes.NewReader(raw)
	}

	req, err := http.NewRequestWithContext(ctx, method, target, payload)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if parsed, err := url.Parse(target); err == nil {
		c.applyAuth(req, parsed)
	}

	resp, err := c.httpClient(mineruTimeout).Do(req)
	if err != nil {
		return fmt.Errorf("HTTP request: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return fmt.Errorf("status %d: %s", resp.StatusCode, truncateForLog(string(raw), 300))
	}
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return fmt.Errorf("decode response: %w (%s)", err, truncateForLog(string(raw), 200))
	}
	return nil
}

func truncateForLog(value string, limit int) string {
	if len(value) <= limit {
		return value
	}
	return value[:limit] + "..."
}

func (c *mineruV1Client) createUpload(ctx context.Context, content []byte, fileName, fileType string) (mineruV1UploadResponse, error) {
	sum := sha256.Sum256(content)
	request := map[string]interface{}{
		"filename":  fileName,
		"bytes":     len(content),
		"mime_type": mineruV1MimeType(fileName, fileType),
		"purpose":   "parse",
		"sha256sum": hex.EncodeToString(sum[:]),
	}

	var upload mineruV1UploadResponse
	if err := c.doJSON(ctx, http.MethodPost, c.endpoint+mineruV1UploadsPath, request, &upload); err != nil {
		return upload, fmt.Errorf("create upload: %w", err)
	}
	return upload, nil
}

func (c *mineruV1Client) putUploadBytes(ctx context.Context, upload mineruV1UploadResponse, content []byte) error {
	if strings.TrimSpace(upload.UploadURL) == "" {
		return fmt.Errorf("upload response has no upload_url")
	}
	target, err := url.Parse(upload.UploadURL)
	if err != nil {
		return fmt.Errorf("invalid upload_url: %w", err)
	}
	if !target.IsAbs() {
		base, err := url.Parse(c.endpoint)
		if err != nil {
			return fmt.Errorf("invalid endpoint: %w", err)
		}
		target = base.ResolveReference(target)
	}

	method := strings.ToUpper(strings.TrimSpace(upload.UploadMethod))
	if method == "" {
		method = http.MethodPut
	}
	if method != http.MethodPut {
		return fmt.Errorf("unsupported upload_method %q", method)
	}

	req, err := http.NewRequestWithContext(ctx, method, target.String(), bytes.NewReader(content))
	if err != nil {
		return fmt.Errorf("create upload request: %w", err)
	}
	for key, value := range upload.UploadHeaders {
		req.Header.Set(key, value)
	}
	c.applyAuth(req, target)

	resp, err := c.httpClient(mineruTimeout).Do(req)
	if err != nil {
		return fmt.Errorf("upload bytes: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		raw, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("upload bytes: status %d: %s", resp.StatusCode, truncateForLog(string(raw), 300))
	}
	return nil
}

func (c *mineruV1Client) completeUpload(ctx context.Context, uploadID, sha256sum string) (mineruV1UploadResponse, error) {
	body := map[string]interface{}{"sha256sum": sha256sum}
	target := fmt.Sprintf("%s/%s/complete", c.endpoint+mineruV1UploadsPath, url.PathEscape(uploadID))
	var completed mineruV1UploadResponse
	if err := c.doJSON(ctx, http.MethodPost, target, body, &completed); err != nil {
		return completed, fmt.Errorf("complete upload: %w", err)
	}
	return completed, nil
}

// uploadFile runs the create-upload / PUT bytes / complete-upload sequence and
// returns the MinerU file id used by parse jobs.
func (c *mineruV1Client) uploadFile(ctx context.Context, content []byte, fileName, fileType string) (string, error) {
	sum := sha256.Sum256(content)
	upload, err := c.createUpload(ctx, content, fileName, fileType)
	if err != nil {
		return "", err
	}

	switch upload.Status {
	case "completed":
		// Deduplicated upload; the file id is already available.
	case "pending":
		if err := c.putUploadBytes(ctx, upload, content); err != nil {
			return "", err
		}
		completed, err := c.completeUpload(ctx, upload.ID, hex.EncodeToString(sum[:]))
		if err != nil {
			return "", err
		}
		upload = completed
	default:
		return "", fmt.Errorf("unexpected upload status %q", upload.Status)
	}

	if upload.File != nil && strings.TrimSpace(upload.File.ID) != "" {
		return upload.File.ID, nil
	}
	return "", fmt.Errorf("upload response has no file id")
}

func (c *mineruV1Client) createJob(ctx context.Context, fileID, ocrMode, pageRange string) (mineruV1JobResponse, error) {
	entry := mineruV1JobFileEntry{
		Source: mineruV1Source{Type: "file_id", FileID: fileID},
	}
	if strings.TrimSpace(pageRange) != "" {
		entry.PageRange = strings.TrimSpace(pageRange)
	}

	request := mineruV1CreateJobRequest{
		Files:         []mineruV1JobFileEntry{entry},
		Tier:          c.tier,
		OCRMode:       ocrMode,
		OutputFormats: []string{"markdown", "zip"},
	}

	var job mineruV1JobResponse
	if err := c.doJSON(ctx, http.MethodPost, c.endpoint+mineruV1JobsPath, request, &job); err != nil {
		return job, fmt.Errorf("create parse job: %w", err)
	}
	if strings.TrimSpace(job.JobID) == "" {
		return job, fmt.Errorf("create parse job: response has no job_id")
	}
	return job, nil
}

func (c *mineruV1Client) pollJob(ctx context.Context, jobID string) (mineruV1JobResponse, error) {
	target := fmt.Sprintf("%s/%s", c.endpoint+mineruV1JobsPath, url.PathEscape(jobID))
	var job mineruV1JobResponse
	if err := c.doJSON(ctx, http.MethodGet, target, nil, &job); err != nil {
		return job, fmt.Errorf("poll parse job: %w", err)
	}
	return job, nil
}

func mineruV1JobFailure(job mineruV1JobResponse) string {
	for _, file := range job.Files {
		if file.Error != nil {
			return fmt.Sprintf("%s: %s (%s)", file.Name, file.Error.Message, file.Error.Code)
		}
		if file.Status == "failed" {
			return fmt.Sprintf("%s: parsing failed", file.Name)
		}
	}
	return ""
}

func (c *mineruV1Client) waitForJob(ctx context.Context, initial mineruV1JobResponse) (mineruV1JobResponse, error) {
	job := initial
	deadline := time.Now().Add(mineruTimeout)

	for {
		switch job.Status {
		case "completed", "partial":
			return job, nil
		case "failed", "canceled":
			detail := mineruV1JobFailure(job)
			if detail == "" {
				detail = job.Status
			}
			return job, fmt.Errorf("parse job %s failed: %s", job.JobID, detail)
		}

		if time.Now().After(deadline) {
			return job, fmt.Errorf("parse job %s timed out after %s (last status %q)", job.JobID, mineruTimeout, job.Status)
		}

		select {
		case <-ctx.Done():
			return job, ctx.Err()
		case <-time.After(mineruV1PollInterval):
		}

		next, err := c.pollJob(ctx, job.JobID)
		if err != nil {
			return job, err
		}
		job = next
	}
}

func (c *mineruV1Client) downloadFile(ctx context.Context, fileID string) ([]byte, error) {
	target := fmt.Sprintf("%s/%s/content", c.endpoint+mineruV1FilesPath, url.PathEscape(fileID))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		return nil, fmt.Errorf("create download request: %w", err)
	}
	if parsed, err := url.Parse(target); err == nil {
		c.applyAuth(req, parsed)
	}

	resp, err := c.httpClient(mineruTimeout).Do(req)
	if err != nil {
		return nil, fmt.Errorf("download artifact: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		raw, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("download artifact: status %d: %s", resp.StatusCode, truncateForLog(string(raw), 300))
	}
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read artifact: %w", err)
	}
	return raw, nil
}

// extractMineruV1Zip reads the Markdown entry and image assets from the ZIP
// bundle produced by the MinerU V1 API (markdown.md + images/...).
func extractMineruV1Zip(raw []byte) (string, map[string]string, error) {
	reader, err := zip.NewReader(bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		return "", nil, fmt.Errorf("open zip: %w", err)
	}

	images := make(map[string]string)
	var markdown string
	var fallbackMarkdown string

	for _, file := range reader.File {
		if file.FileInfo().IsDir() {
			continue
		}
		name := strings.TrimPrefix(path.Clean(file.Name), "./")
		lowerName := strings.ToLower(name)

		switch {
		case strings.HasSuffix(lowerName, ".md"):
			content, err := readMineruV1ZipEntry(file)
			if err != nil {
				return "", nil, err
			}
			if path.Base(lowerName) == "markdown.md" && markdown == "" {
				markdown = string(content)
			} else if fallbackMarkdown == "" {
				fallbackMarkdown = string(content)
			}
		case strings.HasPrefix(lowerName, "images/"), strings.Contains(lowerName, "/images/"):
			content, err := readMineruV1ZipEntry(file)
			if err != nil {
				return "", nil, err
			}
			mimeType := mime.TypeByExtension(path.Ext(lowerName))
			if mimeType == "" {
				continue
			}
			images[name] = "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(content)
		}
	}

	if markdown == "" {
		markdown = fallbackMarkdown
	}
	if markdown == "" {
		return "", nil, fmt.Errorf("zip bundle has no markdown entry")
	}
	return markdown, images, nil
}

func readMineruV1ZipEntry(file *zip.File) ([]byte, error) {
	handle, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("open zip entry %s: %w", file.Name, err)
	}
	defer handle.Close()
	content, err := io.ReadAll(handle)
	if err != nil {
		return nil, fmt.Errorf("read zip entry %s: %w", file.Name, err)
	}
	return content, nil
}

type mineruV1HealthResponse struct {
	Status  string `json:"status"`
	Version string `json:"version"`
}

// probeMineruV1 reports whether the endpoint serves the MinerU 4.0 V1 API.
// Older MinerU releases answer /file_parse only and fail this probe.
func probeMineruV1(ctx context.Context, endpoint, apiKey string) bool {
	endpoint = strings.TrimRight(strings.TrimSpace(endpoint), "/")
	if endpoint == "" {
		return false
	}

	probeCtx, cancel := context.WithTimeout(ctx, mineruV1ProbeTimeout)
	defer cancel()

	client := newMineruV1Client(endpoint, apiKey, mineruV1DefaultTier)
	req, err := http.NewRequestWithContext(probeCtx, http.MethodGet, endpoint+mineruV1HealthPath, nil)
	if err != nil {
		return false
	}
	if parsed, err := url.Parse(endpoint); err == nil {
		client.applyAuth(req, parsed)
	}

	resp, err := client.httpClient(mineruV1ProbeTimeout).Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return false
	}

	var health mineruV1HealthResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, 64*1024)).Decode(&health); err != nil {
		return false
	}
	return strings.EqualFold(strings.TrimSpace(health.Status), "ok")
}

// parseV1 runs the full MinerU 4.0 V1 parsing flow and returns markdown plus
// base64-encoded images keyed by their bundle path.
func (c *mineruV1Client) parse(ctx context.Context, content []byte, fileName, fileType, ocrMode, pageRange string) (string, map[string]string, error) {
	fileID, err := c.uploadFile(ctx, content, fileName, fileType)
	if err != nil {
		return "", nil, err
	}
	logger.Infof(ctx, "[MinerU] v1 upload ready file_id=%s tier=%s", fileID, c.tier)

	job, err := c.createJob(ctx, fileID, ocrMode, pageRange)
	if err != nil {
		return "", nil, err
	}
	logger.Infof(ctx, "[MinerU] v1 parse job submitted job_id=%s status=%s", job.JobID, job.Status)

	job, err = c.waitForJob(ctx, job)
	if err != nil {
		return "", nil, err
	}
	return c.fetchArtifacts(ctx, job)
}

func (c *mineruV1Client) fetchArtifacts(ctx context.Context, job mineruV1JobResponse) (string, map[string]string, error) {
	if len(job.Files) == 0 {
		return "", nil, fmt.Errorf("parse job %s returned no files", job.JobID)
	}

	var lastErr error
	for _, file := range job.Files {
		if file.Status != "completed" {
			if file.Error != nil {
				lastErr = fmt.Errorf("%s: %s (%s)", file.Name, file.Error.Message, file.Error.Code)
			} else {
				lastErr = fmt.Errorf("%s: status %s", file.Name, file.Status)
			}
			continue
		}
		if file.OutputFiles == nil {
			lastErr = fmt.Errorf("%s: no output files", file.Name)
			continue
		}

		if file.OutputFiles.Zip != nil && strings.TrimSpace(file.OutputFiles.Zip.FileID) != "" {
			raw, err := c.downloadFile(ctx, file.OutputFiles.Zip.FileID)
			if err != nil {
				return "", nil, err
			}
			markdown, images, err := extractMineruV1Zip(raw)
			if err == nil {
				logger.Infof(ctx, "[MinerU] v1 artifacts ready markdown=%d chars images=%d", len(markdown), len(images))
				return markdown, images, nil
			}
			logger.Warnf(ctx, "[MinerU] v1 zip bundle unusable (%v), falling back to markdown artifact", err)
		}

		if file.OutputFiles.Markdown != nil && strings.TrimSpace(file.OutputFiles.Markdown.FileID) != "" {
			raw, err := c.downloadFile(ctx, file.OutputFiles.Markdown.FileID)
			if err != nil {
				return "", nil, err
			}
			return string(raw), map[string]string{}, nil
		}
		lastErr = fmt.Errorf("%s: completed without markdown or zip artifact", file.Name)
	}

	if lastErr == nil {
		lastErr = fmt.Errorf("parse job %s produced no usable artifacts", job.JobID)
	}
	return "", nil, lastErr
}