package service

import (
	"regexp"
	"strings"
	"unicode"

	htmltomd "github.com/JohannesKaufmann/html-to-markdown/v2"
	"github.com/Tencent/WeKnora/internal/infrastructure/docparser"
)

var (
	htmlTagPattern    = regexp.MustCompile(`<[^>]+>`)
	codeBlockPattern  = regexp.MustCompile("(?s)^\\s*```[a-zA-Z]*\\s*\n(.*?)\n\\s*```\\s*$")
	htmlDocPattern    = regexp.MustCompile(`(?i)^\s*(<\!DOCTYPE|<html|<body|<div|<p[\s>]|<table|<h[1-6][\s>])`)
	multipleNewlines  = regexp.MustCompile(`\n{3,}`)
	knownEmptyReplies = []string{
		"无文字内容",
		"无法识别",
		"no text",
		"no text content",
		"no content",
		"empty",
		"图片中没有文字",
		"图片中没有可识别的文字",
	}
)

// Degenerate-output heuristic thresholds.
//
// Vision models fed a non-text figure (FEA contour plots, diagrams, charts)
// frequently fall into a repetition loop and keep emitting the same token or
// line until max_tokens is exhausted (finish_reason=length). The result is a
// multi-kilobyte block of near-pure repetition that, left unchecked, becomes an
// image_ocr chunk, gets embedded, and pollutes retrieval. Both OvisOCR2 and
// PaddleOCR-VL reproduce this on the same input, so the guard has to live here
// rather than in model selection.
//
// The thresholds are deliberately conservative: every rule additionally
// requires degenerateMinBytes of output, so short OCR results ("Before  After")
// and legitimately terse labels are never discarded.
const (
	// degenerateMinBytes is the shortest output any rule will judge.
	degenerateMinBytes = 400
	// degenerateMinTokens is the fewest whitespace-separated tokens the
	// token-uniqueness rule will judge.
	degenerateMinTokens = 50
	// degenerateMaxTokenRatio discards output whose distinct-token share
	// falls below this fraction.
	degenerateMaxTokenRatio = 0.12
	// degenerateMinLines and degenerateLineShare discard output where one
	// repeated line dominates: at least degenerateMinLines non-empty lines
	// and a single distinct line covering degenerateLineShare of them.
	degenerateMinLines  = 8
	degenerateLineShare = 0.5
	// degenerateMaxRuneRun discards output containing a single non-space
	// rune repeated this many times in a row.
	degenerateMaxRuneRun = 100
)

// sanitizeOCRText cleans up VLM OCR output by stripping HTML wrappers,
// converting HTML to markdown, and filtering out useless responses.
func sanitizeOCRText(raw string) string {
	text := strings.TrimSpace(raw)
	if text == "" {
		return ""
	}

	text = stripMarkdownCodeBlock(text)

	// If stripping HTML tags leaves almost no text, the response is useless
	// (e.g. "<html><body><div class="image"><img/></div></body></html>").
	plainText := strings.TrimSpace(htmlTagPattern.ReplaceAllString(text, ""))
	if len(plainText) < 10 && htmlTagPattern.MatchString(text) {
		return ""
	}

	if looksLikeHTML(text) {
		text = ocrHTMLToMarkdown(text)
		text = strings.TrimSpace(text)
		if text == "" {
			return ""
		}
	}

	// Convert inline HTML <table> blocks to GFM tables. This is a safe no-op
	// when no <table> is present, so it runs unconditionally: a markdown body
	// with an embedded HTML table does not satisfy looksLikeHTML (it neither
	// starts with a tag nor is dominated by tag characters), yet leaving the
	// raw markup in place makes the chunker split inside table rows.
	text = docparser.NormalizeHTMLTables(text)

	if isKnownEmptyReply(text) {
		return ""
	}

	if isDegenerateOCRText(text) {
		return ""
	}

	text = multipleNewlines.ReplaceAllString(text, "\n\n")
	return strings.TrimSpace(text)
}

// isDegenerateOCRText reports whether text looks like a model repetition loop
// rather than real OCR output. It returns false for anything shorter than
// degenerateMinBytes so short answers are always preserved.
func isDegenerateOCRText(text string) bool {
	if len(text) < degenerateMinBytes {
		return false
	}
	return hasLongRepeatedRuneRun(text) ||
		hasDominantRepeatedLine(text) ||
		hasDegenerateTokenUniqueness(text)
}

// hasLongRepeatedRuneRun detects a single non-space rune repeated many times in
// a row (e.g. a runaway "0.000000..." filename inside an <img> tag).
func hasLongRepeatedRuneRun(text string) bool {
	var prev rune
	run := 0
	for _, r := range text {
		if unicode.IsSpace(r) {
			prev, run = 0, 0
			continue
		}
		if r == prev {
			run++
		} else {
			prev, run = r, 1
		}
		if run >= degenerateMaxRuneRun {
			return true
		}
	}
	return false
}

// hasDominantRepeatedLine detects output where one line is emitted over and
// over: the classic "line\n\nline\n\nline..." loop. Lines are compared after
// whitespace normalisation so spacing changes do not defeat the check.
func hasDominantRepeatedLine(text string) bool {
	counts := make(map[string]int)
	total := 0
	for _, line := range strings.Split(text, "\n") {
		norm := strings.Join(strings.Fields(line), " ")
		if norm == "" {
			continue
		}
		counts[norm]++
		total++
	}
	if total < degenerateMinLines {
		return false
	}
	for _, n := range counts {
		if float64(n)/float64(total) >= degenerateLineShare {
			return true
		}
	}
	return false
}

// hasDegenerateTokenUniqueness detects output built from a tiny set of tokens
// repeated many times, which covers loops that never emit a newline (e.g.
// "$0.00 = 0.00$  $0.00 = 0.00$  ...").
func hasDegenerateTokenUniqueness(text string) bool {
	fields := strings.Fields(text)
	if len(fields) < degenerateMinTokens {
		return false
	}
	unique := make(map[string]struct{}, len(fields))
	for _, f := range fields {
		unique[f] = struct{}{}
	}
	return float64(len(unique))/float64(len(fields)) < degenerateMaxTokenRatio
}

// stripMarkdownCodeBlock removes a markdown code-fence wrapper that some
// models add around their output (e.g. ```html\n...\n``` or ```markdown\n...\n```).
func stripMarkdownCodeBlock(text string) string {
	if m := codeBlockPattern.FindStringSubmatch(text); len(m) == 2 {
		return strings.TrimSpace(m[1])
	}
	return text
}

// looksLikeHTML returns true when the text appears to be an HTML document
// or contains a significant amount of HTML tags.
func looksLikeHTML(text string) bool {
	if htmlDocPattern.MatchString(text) {
		return true
	}
	tags := htmlTagPattern.FindAllString(text, -1)
	if len(tags) == 0 {
		return false
	}
	tagChars := 0
	for _, t := range tags {
		tagChars += len(t)
	}
	return float64(tagChars)/float64(len(text)) > 0.3
}

// ocrHTMLToMarkdown converts HTML content to markdown, falling back to the
// original text on failure.
func ocrHTMLToMarkdown(content string) string {
	md, err := htmltomd.ConvertString(content)
	if err != nil {
		return content
	}
	return md
}

// isKnownEmptyReply checks whether the text matches a known "no content"
// reply pattern that VLM models produce when the image has no text.
// Trailing punctuation (., !, ?) is stripped before comparison so that
// responses like "No text content." still match "no text content".
func isKnownEmptyReply(text string) bool {
	lower := strings.ToLower(strings.TrimSpace(text))
	lower = strings.TrimRight(lower, ".!?。！？")
	for _, phrase := range knownEmptyReplies {
		if lower == strings.ToLower(phrase) {
			return true
		}
	}
	return false
}
