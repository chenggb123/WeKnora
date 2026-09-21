package docparser

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/Tencent/WeKnora/internal/types"
	"github.com/Tencent/WeKnora/internal/utils"
)

const mineruV1TestMarkdown = "# Sample\n\nHello MinerU 4.0\n\n![figure](images/fig1.png)\n"

func buildMineruV1Bundle(t *testing.T) []byte {
	t.Helper()

	var buf bytes.Buffer
	writer := zip.NewWriter(&buf)

	mdEntry, err := writer.Create("markdown.md")
	if err != nil {
		t.Fatalf("create markdown entry: %v", err)
	}
	if _, err := mdEntry.Write([]byte(mineruV1TestMarkdown)); err != nil {
		t.Fatalf("write markdown entry: %v", err)
	}

	imgEntry, err := writer.Create("images/fig1.png")
	if err != nil {
		t.Fatalf("create image entry: %v", err)
	}
	if _, err := imgEntry.Write([]byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a}); err != nil {
		t.Fatalf("write image entry: %v", err)
	}

	if err := writer.Close(); err != nil {
		t.Fatalf("close zip: %v", err)
	}
	return buf.Bytes()
}

func allowLoopbackSSRF(t *testing.T) {
	t.Helper()
	utils.SetSSRFWhitelistFromRaw("127.0.0.1")
	t.Cleanup(func() { utils.SetSSRFWhitelistFromRaw("") })
}

// newMineruV1TestServer mocks the MinerU >= 4.0 V1 API: upload -> job -> files.
func newMineruV1TestServer(t *testing.T, bundle []byte, wantAPIKey string, jobRequest *mineruV1CreateJobRequest) (*httptest.Server, *int32, *[]byte) {
	t.Helper()

	var polls int32
	uploaded := make([]byte, 0)
	server := httptest.NewServer(nil)
	mux := http.NewServeMux()
	server.Config.Handler = mux

	checkAuth := func(w http.ResponseWriter, r *http.Request) bool {
		if wantAPIKey == "" {
			return true
		}
		if r.Header.Get("Authorization") != "Bearer "+wantAPIKey {
			w.WriteHeader(http.StatusUnauthorized)
			return false
		}
		return true
	}

	mux.HandleFunc("/v1/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if !checkAuth(w, r) {
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"status": "ok", "version": "4.0.4"})
	})

	mux.HandleFunc("/v1/uploads", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode upload request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":            "up1",
			"status":        "pending",
			"upload_url":    server.URL + "/upload/up1",
			"upload_method": "PUT",
			"upload_headers": map[string]string{
				"X-Test-Header": "kept",
			},
		})
	})

	mux.HandleFunc("/upload/up1", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPut {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		if r.Header.Get("X-Test-Header") != "kept" {
			t.Errorf("upload headers were not forwarded")
		}
		raw, _ := io.ReadAll(r.Body)
		uploaded = raw
		w.WriteHeader(http.StatusOK)
	})

	mux.HandleFunc("/v1/uploads/up1/complete", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":     "up1",
			"status": "completed",
			"file":   map[string]any{"id": "file-1", "filename": "sample.pdf"},
		})
	})

	mux.HandleFunc("/v1/parse/jobs", func(w http.ResponseWriter, r *http.Request) {
		var body mineruV1CreateJobRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode job request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if jobRequest != nil {
			*jobRequest = body
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"job_id": "job-1",
			"status": "queued",
			"files":  []any{},
		})
	})

	mux.HandleFunc("/v1/parse/jobs/job-1", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if atomic.AddInt32(&polls, 1) < 2 {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"job_id": "job-1",
				"status": "running",
				"files":  []any{},
			})
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"job_id": "job-1",
			"status": "completed",
			"files": []any{
				map[string]any{
					"name":   "sample.pdf",
					"status": "completed",
					"output_files": map[string]any{
						"markdown": map[string]any{"file_id": "out-md", "bytes": len(mineruV1TestMarkdown)},
						"zip":      map[string]any{"file_id": "out-zip", "bytes": len(bundle)},
					},
				},
			},
		})
	})

	mux.HandleFunc("/v1/files/out-zip/content", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/zip")
		_, _ = w.Write(bundle)
	})

	return server, &polls, &uploaded
}

func TestMinerUV1ParseFlow(t *testing.T) {
	allowLoopbackSSRF(t)

	bundle := buildMineruV1Bundle(t)
	var jobRequest mineruV1CreateJobRequest
	server, polls, uploaded := newMineruV1TestServer(t, bundle, "", &jobRequest)
	defer server.Close()

	content := []byte("%PDF-1.7 mineru v1 test")
	reader := NewMinerUReader(map[string]string{
		"mineru_endpoint":      server.URL,
		"mineru_tier":          "basic",
		"mineru_parse_method":  "ocr",
		"mineru_api_version":   "v1",
		"mineru_page_range":    "1-3",
	})

	result, err := reader.Read(context.Background(), &types.ReadRequest{
		FileName:    "sample.pdf",
		FileType:    "pdf",
		FileContent: content,
	})
	if err != nil {
		t.Fatalf("Read returned error: %v", err)
	}
	if result.Error != "" {
		t.Fatalf("Read returned result error: %s", result.Error)
	}
	if !strings.Contains(result.MarkdownContent, "Hello MinerU 4.0") {
		t.Fatalf("markdown not extracted, got %q", result.MarkdownContent)
	}
	if len(result.ImageRefs) != 1 {
		t.Fatalf("expected 1 image ref, got %d", len(result.ImageRefs))
	}
	if result.ImageRefs[0].Filename != "images/fig1.png" {
		t.Fatalf("unexpected image filename %q", result.ImageRefs[0].Filename)
	}
	if len(*uploaded) != len(content) || string(*uploaded) != string(content) {
		t.Fatalf("uploaded bytes mismatch")
	}
	if *polls < 2 {
		t.Fatalf("expected polling until terminal status, polls=%d", *polls)
	}
	if jobRequest.Tier != "basic" {
		t.Fatalf("tier = %q, want basic", jobRequest.Tier)
	}
	if jobRequest.OCRMode != "ocr" {
		t.Fatalf("ocr_mode = %q, want ocr", jobRequest.OCRMode)
	}
	if len(jobRequest.Files) != 1 || jobRequest.Files[0].PageRange != "1-3" {
		t.Fatalf("unexpected job files payload: %+v", jobRequest.Files)
	}
	if jobRequest.Files[0].Source.FileID != "file-1" {
		t.Fatalf("source file id = %q, want file-1", jobRequest.Files[0].Source.FileID)
	}
	if strings.Join(jobRequest.OutputFormats, ",") != "markdown,zip" {
		t.Fatalf("output_formats = %v", jobRequest.OutputFormats)
	}
}

func TestMinerUV1ProbeFallsBackToLegacy(t *testing.T) {
	allowLoopbackSSRF(t)

	legacy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Legacy MinerU (<= 3.x) has no V1 health endpoint.
		http.NotFound(w, r)
	}))
	defer legacy.Close()

	if probeMineruV1(context.Background(), legacy.URL, "") {
		t.Fatalf("legacy server must not be detected as V1")
	}
	if version := (&MinerUReader{endpoint: legacy.URL}).resolveAPIVersion(context.Background()); version != "legacy" {
		t.Fatalf("resolveAPIVersion = %q, want legacy", version)
	}
}

func TestResolveMineruTier(t *testing.T) {
	cases := map[string]string{
		"":          "standard",
		"flash":     "flash",
		"BASIC":     "basic",
		"advanced":  "advanced",
		"pipeline":  "standard",
		"hybrid-http-client": "standard",
	}
	for input, want := range cases {
		if got := resolveMineruTier(input); got != want {
			t.Errorf("resolveMineruTier(%q) = %q, want %q", input, got, want)
		}
	}
}