package service

import (
	"fmt"
	"regexp"
	"strings"
	"testing"
)

func TestSanitizeOCRText(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "empty string",
			input: "",
			want:  "",
		},
		{
			name:  "whitespace only",
			input: "   \n\t  ",
			want:  "",
		},
		{
			name:  "pure HTML skeleton with no text",
			input: `<html><body><div class="image"><img/></div></body></html>`,
			want:  "",
		},
		{
			name:  "HTML with only whitespace text",
			input: "<html><body>  \n  </body></html>",
			want:  "",
		},
		{
			name:  "valid markdown passes through",
			input: "# 标题\n\n这是一段正文，包含一些内容。\n\n| 列1 | 列2 |\n| --- | --- |\n| 数据1 | 数据2 |",
			want:  "# 标题\n\n这是一段正文，包含一些内容。\n\n| 列1 | 列2 |\n| --- | --- |\n| 数据1 | 数据2 |",
		},
		{
			name:  "code block wrapper stripped",
			input: "```markdown\n# 文档标题\n\n正文内容在这里。\n```",
			want:  "# 文档标题\n\n正文内容在这里。",
		},
		{
			name:  "html code block wrapper stripped",
			input: "```html\n<p>这是一段内容</p>\n```",
			want:  "这是一段内容",
		},
		{
			name:  "HTML document converted to markdown",
			input: "<html><body><h1>标题</h1><p>这是一段很长的正文内容，用来测试 HTML 到 Markdown 的转换。</p></body></html>",
			want:  "# 标题\n\n这是一段很长的正文内容，用来测试 HTML 到 Markdown 的转换。",
		},
		{
			name:  "known empty reply - Chinese",
			input: "无文字内容",
			want:  "",
		},
		{
			name:  "known empty reply - no text",
			input: "No text",
			want:  "",
		},
		{
			name:  "known empty reply - 图片中没有文字",
			input: "图片中没有文字",
			want:  "",
		},
		{
			name:  "plain text with minimal HTML not converted",
			input: "这是一段正常文本，价格 <100 元。",
			want:  "这是一段正常文本，价格 <100 元。",
		},
		{
			name:  "multiple blank lines collapsed",
			input: "段落一\n\n\n\n\n段落二",
			want:  "段落一\n\n段落二",
		},
		{
			name:  "HTML with substantial text content is converted",
			input: "<div><h2>报告摘要</h2><p>本季度营收同比增长 15%，净利润达到 2.3 亿元。</p><table><tr><th>指标</th><th>数值</th></tr><tr><td>营收</td><td>10亿</td></tr></table></div>",
			want:  "", // placeholder; will be checked for non-empty
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := sanitizeOCRText(tt.input)

			if tt.name == "HTML with substantial text content is converted" {
				if got == "" {
					t.Errorf("sanitizeOCRText() returned empty for substantial HTML content")
				}
				if got == tt.input {
					t.Errorf("sanitizeOCRText() did not convert HTML, got original")
				}
				return
			}

			if got != tt.want {
				t.Errorf("sanitizeOCRText() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestStripMarkdownCodeBlock(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "no code block",
			input: "just normal text",
			want:  "just normal text",
		},
		{
			name:  "markdown code block",
			input: "```markdown\n# Title\nContent here\n```",
			want:  "# Title\nContent here",
		},
		{
			name:  "html code block",
			input: "```html\n<p>hello</p>\n```",
			want:  "<p>hello</p>",
		},
		{
			name:  "plain code block",
			input: "```\nsome text\n```",
			want:  "some text",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := stripMarkdownCodeBlock(tt.input)
			if got != tt.want {
				t.Errorf("stripMarkdownCodeBlock() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestLooksLikeHTML(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  bool
	}{
		{
			name:  "HTML document",
			input: "<html><body><p>text</p></body></html>",
			want:  true,
		},
		{
			name:  "DOCTYPE",
			input: "<!DOCTYPE html><html><body></body></html>",
			want:  true,
		},
		{
			name:  "body tag",
			input: "<body><p>content</p></body>",
			want:  true,
		},
		{
			name:  "plain markdown",
			input: "# Title\n\nSome paragraph text",
			want:  false,
		},
		{
			name:  "text with minor HTML",
			input: "This is mostly text with a <b>bold</b> word.",
			want:  false,
		},
		{
			name:  "heavy HTML tags",
			input: "<div><p><span>x</span></p></div><div><p><span>y</span></p></div>",
			want:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := looksLikeHTML(tt.input)
			if got != tt.want {
				t.Errorf("looksLikeHTML() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestSanitizeOCRText_ConvertsInlineHTMLTable guards against a regression where
// an HTML <table> embedded in a markdown body survived sanitization untouched:
// the mixed content does not satisfy looksLikeHTML, so the old HTML-to-markdown
// path never ran and the raw <table> markup reached the chunker.
func TestSanitizeOCRText_ConvertsInlineHTMLTable(t *testing.T) {
	para := "本季度公司整体经营情况保持稳定，营收与利润两项核心指标均实现同比增长。" +
		"为了便于管理层快速了解经营成果，下表汇总了报告期内的主要财务数据，" +
		"供后续经营分析、预算编制以及年度考核等工作参考使用，" +
		"请结合实际业务情况综合判断，切勿脱离业务背景单独解读其中任何一项数字。"
	tail := "以上数据均来自财务部门审核后的正式报表，统计口径与上一报告期保持一致，" +
		"未发生会计政策变更或追溯调整。如有疑问请与财务部门联系确认，" +
		"最终解释权归公司财务部门所有。"
	input := "# 报告\n\n" + para + "\n\n" +
		`<table><tr><th>指标</th><th>数值</th></tr>` +
		`<tr><td>营收</td><td>10亿</td></tr>` +
		`<tr><td>利润</td><td>2.3亿</td></tr></table>` +
		"\n\n" + tail + "\n\n"

	if looksLikeHTML(input) {
		t.Fatalf("test input unexpectedly looks like HTML; the regression test must exercise the mixed-content path")
	}

	got := sanitizeOCRText(input)

	if strings.Contains(got, "<table") {
		t.Fatalf("expected inline HTML table to be converted, got:\n%s", got)
	}
	if !strings.Contains(got, "# 报告") || !strings.Contains(got, "最终解释权归公司财务部门所有。") {
		t.Fatalf("expected surrounding markdown to be preserved, got:\n%s", got)
	}
	for _, want := range []string{"指标", "数值", "营收", "10亿", "利润", "2.3亿"} {
		if !strings.Contains(got, want) {
			t.Fatalf("expected converted table to retain %q, got:\n%s", want, got)
		}
	}
	if !regexp.MustCompile(`\|[-:]{3,}\|`).MatchString(got) {
		t.Fatalf("expected a GFM table separator row, got:\n%s", got)
	}
}

func TestIsKnownEmptyReply(t *testing.T) {
	tests := []struct {
		input string
		want  bool
	}{
		{"无文字内容", true},
		{"无法识别", true},
		{"no text", true},
		{"No Text", true},
		{"NO CONTENT", true},
		{"empty", true},
		{"这是正常内容", false},
		{"", false},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := isKnownEmptyReply(tt.input)
			if got != tt.want {
				t.Errorf("isKnownEmptyReply(%q) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

// TestSanitizeOCRText_DiscardsRepetitionLoops covers the degenerate outputs
// vision models produce on non-text figures (FEA contour plots, diagrams).
// Both OvisOCR2 and PaddleOCR-VL loop on the same image and run until
// max_tokens, so the sanitizer is the single place that can stop the garbage
// from becoming an image_ocr chunk. Every fixture mirrors a shape observed in
// production: a repeated line, a repeated token group on one line, a tiny
// cycling vocabulary, and a runaway repeated rune.
func TestSanitizeOCRText_DiscardsRepetitionLoops(t *testing.T) {
	cases := []struct {
		name  string
		input string
	}{
		{
			name:  "repeated identical line",
			input: strings.Repeat("$\\text{H}_2\\text{SO}_4$\n\n", 60),
		},
		{
			name:  "repeated token group without newlines",
			input: strings.Repeat("$0.00 = 0.00$  ", 200),
		},
		{
			name: "small cycling token vocabulary",
			input: strings.Repeat(
				"08.8 = 05A + 2T_ST = 05A + 04.2 = 1A + 2T_A = 2A + 00.4 = 0A + 00.0 = 2A + ",
				20,
			),
		},
		{
			name:  "runaway repeated rune",
			input: "0." + strings.Repeat("0", 400),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if len(tc.input) < degenerateMinBytes {
				t.Fatalf("fixture is %d bytes, below degenerateMinBytes=%d; it would not exercise the guard",
					len(tc.input), degenerateMinBytes)
			}
			got := sanitizeOCRText(tc.input)
			if got != "" {
				preview := got
				if len(preview) > 60 {
					preview = preview[:60]
				}
				t.Fatalf("sanitizeOCRText() kept a repetition loop (%d bytes): %q", len(got), preview)
			}
		})
	}
}

// TestSanitizeOCRText_KeepsLegitimateContent locks in the conservative side of
// the guard: real OCR output, however long or terse, must survive untouched.
func TestSanitizeOCRText_KeepsLegitimateContent(t *testing.T) {
	chineseProse := "缸体或飞轮壳维修时，维修后状态如下述方式进行检查。首先确认结合面无油污、无锈蚀，" +
		"随后按对角线顺序分三次拧紧螺栓，扭矩依次递增到规定值。若发现局部变形超过允许范围，" +
		"应重新加工配合面并复测平面度。胶线应连续均匀，不允许出现断胶或堆胶现象，" +
		"涂胶后需在规定时间内完成装配，避免胶体表干影响密封效果。" +
		"装配完成后进行气密性试验，保压期间压降不得超过工艺文件规定的上限。"

	longTable := "| 序号 | 检查项 | 标准值 | 实测值 | 结论 |\n| --- | --- | --- | --- | --- |\n"
	for i := 1; i <= 20; i++ {
		longTable += fmt.Sprintf("| %d | 项目%d | 12.%02d | 12.%02d | 合格 |\n", i, i, i, i+1)
	}

	cases := []struct {
		name  string
		input string
	}{
		{name: "long chinese prose", input: chineseProse},
		{name: "long markdown table with distinct rows", input: longTable},
		{name: "terse label pair", input: "Before  After  Add machining  chamfering  Add two ribs"},
		{
			// Short and repetitive, but below the size floor: terse labels are
			// common on engineering drawings and must not be dropped.
			name:  "short repetitive text below size floor",
			input: "1  1  1  1  1  1  1  1",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := sanitizeOCRText(tc.input); got == "" {
				t.Fatalf("sanitizeOCRText() discarded legitimate content (%d bytes)", len(tc.input))
			}
		})
	}
}

// variedOCRSample is genuinely non-repetitive OCR output used to prove the
// heuristics do not fire on real documents.
var variedOCRSample = strings.Join([]string{
	"第 1 项：结合面平面度检查，标准值 0.05 mm，实测 0.03 mm，判定合格。",
	"第 2 项：螺栓扭矩复核，标准值 210 N·m，实测 208 N·m，记录于检验单 A-17。",
	"第 3 项：胶线连续性目视检查，未见断胶与堆胶，符合工艺文件 QG-2024-11 要求。",
	"第 4 项：飞轮壳与缸体配合间隙测量，塞尺 0.10 mm 不入，满足装配条件。",
	"第 5 项：加强筋焊接质量检查，焊缝饱满无咬边，按目视标准判定可用。",
	"第 6 项：表面清洁度确认，使用无水乙醇擦拭后静置五分钟，无残留油膜。",
	"第 7 项：气密性试验，保压五分钟压降 0.002 MPa，低于允许上限值。",
	"第 8 项：复装后运转测试，怠速运行十分钟无异响、无渗漏现象发生。",
	"第 9 项：检验人员签字确认，班组长复核通过，相关记录已归档保存。",
	"第 10 项：不合格品处置流程说明，需填写偏离单并提交工程师评审。",
}, "\n")

func TestIsDegenerateOCRText(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  bool
	}{
		{
			name:  "empty",
			input: "",
			want:  false,
		},
		{
			// Below the size floor nothing is judged, so terse-but-repetitive
			// labels survive.
			name:  "below size floor is never degenerate",
			input: strings.Repeat("$x$ ", 50),
			want:  false,
		},
		{
			name:  "repeated line",
			input: strings.Repeat("same line here\n", 40),
			want:  true,
		},
		{
			name:  "varied prose is not degenerate",
			input: variedOCRSample,
			want:  false,
		},
		{
			name:  "repeated rune",
			input: strings.Repeat("a", 500),
			want:  true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := isDegenerateOCRText(tc.input); got != tc.want {
				t.Errorf("isDegenerateOCRText() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestHasLongRepeatedRuneRun(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  bool
	}{
		{name: "no run", input: "abcabcabcabc", want: false},
		{name: "one below threshold", input: strings.Repeat("a", degenerateMaxRuneRun-1), want: false},
		{name: "at threshold", input: strings.Repeat("a", degenerateMaxRuneRun), want: true},
		{
			// Whitespace breaks the run, so two 60-rune runs do not add up.
			name:  "spaces reset the run counter",
			input: strings.Repeat("a", 60) + " " + strings.Repeat("a", 60),
			want:  false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := hasLongRepeatedRuneRun(tc.input); got != tc.want {
				t.Errorf("hasLongRepeatedRuneRun() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestHasDominantRepeatedLine(t *testing.T) {
	distinct := make([]string, 0, 10)
	for i := 0; i < 10; i++ {
		distinct = append(distinct, fmt.Sprintf("line number %d with distinct words", i))
	}

	cases := []struct {
		name  string
		input string
		want  bool
	}{
		{name: "no lines", input: "", want: false},
		{name: "fewer lines than the minimum", input: strings.Repeat("x\n", degenerateMinLines-1), want: false},
		{name: "one line dominates", input: strings.Repeat("x\n", 20), want: true},
		{name: "distinct lines", input: strings.Join(distinct, "\n"), want: false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := hasDominantRepeatedLine(tc.input); got != tc.want {
				t.Errorf("hasDominantRepeatedLine() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestHasDegenerateTokenUniqueness(t *testing.T) {
	var varied []string
	for i := 0; i < 60; i++ {
		varied = append(varied, fmt.Sprintf("token%d", i))
	}

	cases := []struct {
		name  string
		input string
		want  bool
	}{
		{name: "below token floor", input: strings.Repeat("a b c ", 10), want: false},
		{name: "tiny cycling vocabulary", input: strings.Repeat("a b c ", 40), want: true},
		{name: "distinct tokens", input: strings.Join(varied, " "), want: false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := hasDegenerateTokenUniqueness(tc.input); got != tc.want {
				t.Errorf("hasDegenerateTokenUniqueness() = %v, want %v", got, tc.want)
			}
		})
	}
}
