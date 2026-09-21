// Package chunker - tokens.go re-exports the language-aware token estimator.
//
// The implementation lives in internal/textmetrics, a leaf package, because it
// is needed on both sides of an existing dependency edge: the chunker uses it
// to size chunks against an embedding model's window, and
// internal/models/embedding needs the same numbers for its pre-flight
// diagnostics. The chunker itself cannot be imported there — chunker →
// docparser → searchutil → types/interfaces → models/embedding is already a
// cycle.
//
// The wrappers below keep the chunker API unchanged for existing callers.
package chunker

import "github.com/Tencent/WeKnora/internal/textmetrics"

// Language identifiers used by the token estimator and the heuristic splitter.
const (
	LangEnglish = textmetrics.LangEnglish
	LangGerman  = textmetrics.LangGerman
	LangChinese = textmetrics.LangChinese
	LangMixed   = textmetrics.LangMixed
)

// ApproxTokenCount returns a conservative token estimate for s in the given
// language. An empty or unknown lang falls back to "mixed".
func ApproxTokenCount(s string, lang string) int {
	return textmetrics.ApproxTokenCount(s, lang)
}

// ApproxTokenCountFromRuneLen is the allocation-free variant of
// ApproxTokenCount when the caller has already computed the rune length.
func ApproxTokenCountFromRuneLen(runeLen int, lang string) int {
	return textmetrics.ApproxTokenCountFromRuneLen(runeLen, lang)
}

// DetectLanguage returns a coarse language label by counting CJK runes vs.
// Latin runes. The result is one of LangChinese, LangGerman, LangEnglish or
// LangMixed.
func DetectLanguage(s string) string {
	return textmetrics.DetectLanguage(s)
}

// CharsForTokenLimit converts a token limit into an approximate character
// budget for a given language.
func CharsForTokenLimit(tokens int, lang string) int {
	return textmetrics.CharsForTokenLimit(tokens, lang)
}
