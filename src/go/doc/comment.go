// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Godoc comment extraction and comment -> HTML formatting.

package doc

import (
	"bytes"
	"internal/lazyregexp"
	"io"
	"strings"
	"text/template" // for HTMLEscape
	"unicode"
	"unicode/utf8"
)

const (
	ldquo = "&ldquo;"
	rdquo = "&rdquo;"
	ulquo = "“"
	urquo = "”"
)

var (
	htmlQuoteReplacer    = strings.NewReplacer(ulquo, ldquo, urquo, rdquo)
	unicodeQuoteReplacer = strings.NewReplacer("``", ulquo, "''", urquo)
)

// Formatter is an interface to allow external custom formatting of godoc comments.
// Internally this should write to a writer.
type Formatter interface {
	// Put makes sure that a string is propperly written to.
	// One thing Put should do is making sure the string is propperly escaped
	// Nice indicates if the text should be formatted or converted
	Put(text string, nice bool)

	// WriteURL writes the URL to the writer using match as a display name.
	// italics indicates wether or not it should be italic. If nice is set,
	// also turn `` and '' into appropirate quotes.
	WriteURL(url, match string, italics, nice bool)

	StartPara()
	PreParaLine(line string)
	PostParaLine(line string)
	EndPara()

	StartHead()
	PreHeadLine(line string)
	PostHeadLine(line string)
	EndHead()

	StartRaw()
	PreRawLine(line string)
	PostRawLine(line string)
	EndRaw()
}

type htmlFormatter struct {
	out    io.Writer
	headID string
}

var (
	htmlPreLink  = []byte(`<a href="`)
	htmlPostLink = []byte(`">`)
	htmlEndLink  = []byte("</a>")
	htmlStartI   = []byte("<i>")
	htmlEndI     = []byte("</i>")
	htmlStartP   = []byte("<p>\n")
	htmlEndP     = []byte("</p>\n")
	htmlStartPre = []byte("<pre>")
	htmlEndPre   = []byte("</pre>\n")
	htmlPreH     = []byte(`<h3 id="`)
	htmlPostH    = []byte(`">`)
	htmlEndH     = []byte("</h3>\n")
)

// Escape escapes text for HTML. If nice is set,
// also turn `` and '' into appropirate quotes.
func (f *htmlFormatter) Put(text string, nice bool) {
	if nice {
		var buf bytes.Buffer
		template.HTMLEscape(&buf, []byte(text))
		// Now we convert the unicode quotes to their HTML escaped entities to maintain old behavior.
		// We need to use a temp buffer to read the string back and do the conversion,
		// otherwise HTMLEscape will escape & to &amp;
		htmlQuoteReplacer.WriteString(f.out, buf.String())
		return
	}
	template.HTMLEscape(f.out, []byte(text))
}

func (f *htmlFormatter) WriteURL(url, match string, italics, nice bool) {
	if len(url) > 0 {
		f.out.Write(htmlPreLink)
		f.Put(url, false)
		f.out.Write(htmlPostLink)
	}
	if italics {
		f.out.Write(htmlStartI)
	}
	f.Put(match, nice)
	if italics {
		f.out.Write(htmlEndI)
	}
	if len(url) > 0 {
		f.out.Write(htmlEndLink)
	}
}

func (f *htmlFormatter) StartPara() {
	f.out.Write(htmlStartP)
}

func (f *htmlFormatter) PreParaLine(line string)  {}
func (f *htmlFormatter) PostParaLine(line string) {}

func (f *htmlFormatter) EndPara() {
	f.out.Write(htmlEndP)
}

func (f *htmlFormatter) StartHead() {
	f.out.Write(htmlPreH)
	f.headID = ""
}

func (f *htmlFormatter) PreHeadLine(line string) {
	if f.headID == "" {
		f.headID = anchorID(line)
		f.out.Write([]byte(f.headID))
		f.out.Write(htmlPostH)
	}
}

func (f *htmlFormatter) PostHeadLine(line string) {}

func (f *htmlFormatter) EndHead() {
	if f.headID == "" {
		f.out.Write(htmlPostH)
	}
	f.out.Write(htmlEndH)
}

func (f *htmlFormatter) StartRaw() {
	f.out.Write(htmlStartPre)
}

func (f *htmlFormatter) PreRawLine(line string)  {}
func (f *htmlFormatter) PostRawLine(line string) {}

func (f *htmlFormatter) EndRaw() {
	f.out.Write(htmlEndPre)
}

type textFormatter struct {
	out       io.Writer
	line      string
	didPrint  bool
	width     int
	indent    string
	preIndent string
	n         int
	pendSpace int
}

var newline = []byte("\n")
var whiteSpace = []byte(" ")
var prefix = []byte("// ")

// Escape escapes text for HTML. If nice is set,
// also turn `` and '' into appropirate quotes.
func (f *textFormatter) Put(text string, nice bool) {
	if nice {
		text = convertQuotes(text)
		f.line += text
		return
	}
	f.out.Write([]byte(text))
}

func (f *textFormatter) WriteURL(url, match string, italics, nice bool) {
	if url == "" {
		url = match
	}
	f.Put(url, nice)
}

func (f *textFormatter) StartPara() {
	if f.didPrint {
		f.out.Write(newline)
	}
	f.didPrint = true
}

func (f *textFormatter) PreParaLine(line string) {}

func (f *textFormatter) PostParaLine(line string) {
	f.write(f.line)
	f.line = ""
}

func (f *textFormatter) EndPara() {
	f.flush()
}

func (f *textFormatter) StartHead() {
	if f.didPrint {
		f.out.Write(newline)
	}
	f.didPrint = true
}

func (f *textFormatter) PreHeadLine(line string) {
	f.out.Write(newline)
}

func (f *textFormatter) PostHeadLine(line string) {
	f.write(f.line)
	f.line = ""
}

func (f *textFormatter) EndHead() {
	f.flush()
}

func (f *textFormatter) StartRaw() {
	f.out.Write(newline)
}

func (f *textFormatter) PreRawLine(line string) {
	if !isBlank(line) {
		f.out.Write([]byte(f.preIndent))
	} else if len(line) == 0 {
		// line is blank, but no newline is present.
		// We should make sure there is a newline
		f.out.Write(newline)
	}
}

func (f *textFormatter) PostRawLine(line string) {}
func (f *textFormatter) EndRaw()                 {}

func (f *textFormatter) write(text string) {
	f.didPrint = true

	needsPrefix := false
	isComment := strings.HasPrefix(text, "//")
	for _, field := range strings.Fields(text) {
		runeCount := utf8.RuneCountInString(field)
		// wrap if line is too long
		if f.n > 0 && f.n+f.pendSpace+runeCount > f.width {
			f.out.Write(newline)
			f.n = 0
			f.pendSpace = 0
			needsPrefix = isComment
		}
		if f.n == 0 {
			f.out.Write([]byte(f.indent))
		}
		if needsPrefix {
			f.out.Write(prefix)
			needsPrefix = false
		}
		f.out.Write(whiteSpace[:f.pendSpace])
		f.out.Write([]byte(field))
		f.n += f.pendSpace + runeCount
		f.pendSpace = 1
	}
}

func (f *textFormatter) flush() {
	if f.n == 0 {
		return
	}
	f.out.Write(newline)
	f.pendSpace = 0
	f.n = 0
}

type markdownFormatter struct {
	out      io.Writer
	didPrint bool
}

var (
	mdEscape      = lazyregexp.New(`([\\\x60*{}[\]()#+\-.!_>~|"$%&'\/:;<=?@^])`)
	mdURLReplacer = strings.NewReplacer(`(`, `\(`, `)`, `\)`)

	mdHeader    = []byte("### ")
	mdIndent    = []byte("&nbsp;&nbsp;&nbsp;&nbsp;")
	mdLinkStart = []byte("[")
	mdLinkDiv   = []byte("](")
	mdLinkEnd   = []byte(")")
)

// Escape escapes text for HTML. If nice is set,
// also turn `` and '' into appropirate quotes.
func (f *markdownFormatter) Put(text string, nice bool) {
	text = mdEscape.ReplaceAllString(text, `\$1`)
	f.out.Write([]byte(text))
}

func (f *markdownFormatter) WriteURL(url, match string, italics, nice bool) {
	if len(url) > 0 {
		f.out.Write(mdLinkStart)
	}
	f.Put(match, nice)
	if italics {
		f.out.Write(htmlStartI)
	}
	if len(url) > 0 {
		f.out.Write(mdLinkDiv)
		f.out.Write([]byte(mdURLReplacer.Replace(url)))
		f.out.Write(mdLinkEnd)
	}
}

func (f *markdownFormatter) StartPara() {
	if f.didPrint {
		f.out.Write(newline)
	}
	f.didPrint = true
}

func (f *markdownFormatter) PreParaLine(line string)  {}
func (f *markdownFormatter) PostParaLine(line string) {}
func (f *markdownFormatter) EndPara()                 {}

func (f *markdownFormatter) StartHead() {
	if f.didPrint {
		f.out.Write(newline)
	}
	f.didPrint = true
}

func (f *markdownFormatter) PreHeadLine(line string) {
	f.out.Write(mdHeader)
}

func (f *markdownFormatter) PostHeadLine(line string) {
	f.out.Write(newline)
}

func (f *markdownFormatter) EndHead() {}

func (f *markdownFormatter) StartRaw() {
	if f.didPrint {
		f.out.Write(newline)
	}
	f.didPrint = true
	f.out.Write(newline)
}

func (f *markdownFormatter) PreRawLine(line string) {
	if !isBlank(line) {
		f.out.Write(mdIndent)
	} else if len(line) == 0 {
		// line is blank, but no newline is present.
		// We should make sure there is a newline
		f.out.Write(newline)
	}
}

func (f *markdownFormatter) PostRawLine(line string) {}
func (f *markdownFormatter) EndRaw()                 {}

// ToHTML converts comment text to formatted HTML.
// The comment was prepared by DocReader,
// so it is known not to have leading, trailing blank lines
// nor to have trailing spaces at the end of lines.
// The comment markers have already been removed.
//
// Each span of unindented non-blank lines is converted into
// a single paragraph. There is one exception to the rule: a span that
// consists of a single line, is followed by another paragraph span,
// begins with a capital letter, and contains no punctuation
// other than parentheses and commas is formatted as a heading.
//
// A span of indented lines is converted into a <pre> block,
// with the common indent prefix removed.
//
// URLs in the comment text are converted into links; if the URL also appears
// in the words map, the link is taken from the map (if the corresponding map
// value is the empty string, the URL is not converted into a link).
//
// Go identifiers that appear in the words map are italicized; if the corresponding
// map value is not the empty string, it is considered a URL and the word is converted
// into a link.
func ToHTML(w io.Writer, text string, words map[string]string) {
	f := &htmlFormatter{
		out: w,
	}
	FormatComment(w, f, text, words)
}

// ToText prepares comment text for presentation in textual output.
// It wraps paragraphs of text to width or fewer Unicode code points
// and then prefixes each line with the indent. In preformatted sections
// (such as program text), it prefixes each non-blank line with preIndent.
func ToText(w io.Writer, text string, indent, preIndent string, width int) {
	f := &textFormatter{
		out:       w,
		width:     width,
		indent:    indent,
		preIndent: preIndent,
	}
	FormatComment(w, f, text, nil)
}

// ToMarkdown converts comment text to formatted markdown.
// The comment was prepared by DocReader,
// so it is known not to have leading, trailing blank lines
// nor to have trailing spaces at the end of lines.
// The comment markers have already been removed.
//
// Each line is converted into a markdown line and empty lines are just converted to
// newlines. Heading are prefixed with `### ` to make it a markdown heading.
//
// A span of indented lines retains a 4 space prefix block, with the common indent
// prefix removed unless empty, in which case it will be converted to a newline.
//
// URLs in the comment text are converted into links.

func ToMarkdown(w io.Writer, text string) {
	f := &markdownFormatter{
		out: w,
	}
	FormatComment(w, f, text, nil)
}

//FormatComment formats a comment according to the Formatter
func FormatComment(w io.Writer, f Formatter, text string, words map[string]string) {
	for _, b := range blocks(text) {
		switch b.op {
		case opPara:
			f.StartPara()
			for _, line := range b.lines {
				escapedLine := convertQuotes(line)
				f.PreParaLine(escapedLine)
				emphasize(w, f, line, words, true)
				f.PostParaLine(escapedLine)
			}
			f.EndPara()
		case opHead:
			f.StartHead()
			for _, line := range b.lines {
				line = convertQuotes(line)
				f.PreHeadLine(line)
				f.Put(line, true)
				f.PostHeadLine(line)
			}
			f.EndHead()
		case opRaw:
			f.StartRaw()
			for _, line := range b.lines {
				f.PreRawLine(line)
				emphasize(w, f, line, nil, false)
				f.PostRawLine(line)
			}
			f.EndRaw()
		}
	}
}

const (
	// Regexp for Go identifiers
	identRx = `[\pL_][\pL_0-9]*`

	// Regexp for URLs
	// Match parens, and check later for balance - see #5043, #22285
	// Match .,:;?! within path, but not at end - see #18139, #16565
	// This excludes some rare yet valid urls ending in common punctuation
	// in order to allow sentences ending in URLs.

	// protocol (required) e.g. http
	protoPart = `(https?|ftp|file|gopher|mailto|nntp)`
	// host (required) e.g. www.example.com or [::1]:8080
	hostPart = `([a-zA-Z0-9_@\-.\[\]:]+)`
	// path+query+fragment (optional) e.g. /path/index.html?q=foo#bar
	pathPart = `([.,:;?!]*[a-zA-Z0-9$'()*+&#=@~_/\-\[\]%])*`

	urlRx = protoPart + `://` + hostPart + pathPart
)

var matchRx = lazyregexp.New(`(` + urlRx + `)|(` + identRx + `)`)

// Emphasize and escape a line of text for HTML. URLs are converted into links;
// if the URL also appears in the words map, the link is taken from the map (if
// the corresponding map value is the empty string, the URL is not converted
// into a link). Go identifiers that appear in the words map are italicized; if
// the corresponding map value is not the empty string, it is considered a URL
// and the word is converted into a link. If nice is set, the remaining text's
// appearance is improved where it makes sense (e.g., `` is turned into &ldquo;
// and '' into &rdquo;).
func emphasize(w io.Writer, f Formatter, line string, words map[string]string, nice bool) {
	for {
		m := matchRx.FindStringSubmatchIndex(line)
		if m == nil {
			break
		}
		// m >= 6 (two parenthesized sub-regexps in matchRx, 1st one is urlRx)

		// write text before match
		pre := line[0:m[0]]
		if nice {
			pre = convertQuotes(line[0:m[0]])
		}
		f.Put(pre, nice)

		// adjust match for URLs
		match := line[m[0]:m[1]]
		if strings.Contains(match, "://") {
			m0, m1 := m[0], m[1]
			for _, s := range []string{"()", "{}", "[]"} {
				open, close := s[:1], s[1:] // E.g., "(" and ")"
				// require opening parentheses before closing parentheses (#22285)
				if i := strings.Index(match, close); i >= 0 && i < strings.Index(match, open) {
					m1 = m0 + i
					match = line[m0:m1]
				}
				// require balanced pairs of parentheses (#5043)
				for i := 0; strings.Count(match, open) != strings.Count(match, close) && i < 10; i++ {
					m1 = strings.LastIndexAny(line[:m1], s)
					match = line[m0:m1]
				}
			}
			if m1 != m[1] {
				// redo matching with shortened line for correct indices
				m = matchRx.FindStringSubmatchIndex(line[:m[0]+len(match)])
			}
		}

		// analyze match
		url := ""
		italics := false
		if words != nil {
			url, italics = words[match]
		}
		if m[2] >= 0 {
			// match against first parenthesized sub-regexp; must be match against urlRx
			if !italics {
				// no alternative URL in words list, use match instead
				url = match
			}
			italics = false // don't italicize URLs
		}
		if nice {
			match = convertQuotes(match)
		}

		// write match
		f.WriteURL(url, match, italics, nice)

		// advance
		line = line[m[1]:]
	}
	if nice {
		line = convertQuotes(line)
	}
	f.Put(line, nice)
}

func convertQuotes(text string) string {
	return unicodeQuoteReplacer.Replace(text)
}

func unindent(block []string) {
	if len(block) == 0 {
		return
	}

	// compute maximum common white prefix
	prefix := block[0][0:indentLen(block[0])]
	for _, line := range block {
		if !isBlank(line) {
			prefix = commonPrefix(prefix, line[0:indentLen(line)])
		}
	}
	n := len(prefix)

	// remove
	for i, line := range block {
		if !isBlank(line) {
			block[i] = line[n:]
		}
	}
}

func indentLen(s string) int {
	i := 0
	for i < len(s) && (s[i] == ' ' || s[i] == '\t') {
		i++
	}
	return i
}

func isBlank(s string) bool {
	return len(s) == 0 || (len(s) == 1 && s[0] == '\n')
}

func commonPrefix(a, b string) string {
	i := 0
	for i < len(a) && i < len(b) && a[i] == b[i] {
		i++
	}
	return a[0:i]
}

type op int

const (
	opPara op = iota
	opHead
	opRaw
)

type block struct {
	op    op
	lines []string
}

var nonAlphaNumRx = lazyregexp.New(`[^a-zA-Z0-9]`)

func anchorID(line string) string {
	// Add a "hdr-" prefix to avoid conflicting with IDs used for package symbols.
	return "hdr-" + nonAlphaNumRx.ReplaceAllString(line, "_")
}

func blocks(text string) []block {
	var (
		out  []block
		para []string

		lastWasBlank   = false
		lastWasHeading = false
	)

	close := func() {
		if para != nil {
			out = append(out, block{opPara, para})
			para = nil
		}
	}

	lines := strings.SplitAfter(text, "\n")
	unindent(lines)
	for i := 0; i < len(lines); {
		line := lines[i]
		if isBlank(line) {
			// close paragraph
			close()
			i++
			lastWasBlank = true
			continue
		}
		if indentLen(line) > 0 {
			// close paragraph
			close()

			// count indented or blank lines
			j := i + 1
			for j < len(lines) && (isBlank(lines[j]) || indentLen(lines[j]) > 0) {
				j++
			}
			// but not trailing blank lines
			for j > i && isBlank(lines[j-1]) {
				j--
			}
			pre := lines[i:j]
			i = j

			unindent(pre)

			// put those lines in a pre block
			out = append(out, block{opRaw, pre})
			lastWasHeading = false
			continue
		}

		if lastWasBlank && !lastWasHeading && i+2 < len(lines) &&
			isBlank(lines[i+1]) && !isBlank(lines[i+2]) && indentLen(lines[i+2]) == 0 {
			// current line is non-blank, surrounded by blank lines
			// and the next non-blank line is not indented: this
			// might be a heading.
			if head := heading(line); head != "" {
				close()
				out = append(out, block{opHead, []string{head}})
				i += 2
				lastWasHeading = true
				continue
			}
		}

		// open paragraph
		lastWasBlank = false
		lastWasHeading = false
		para = append(para, lines[i])
		i++
	}
	close()

	return out
}

// heading returns the trimmed line if it passes as a section heading;
// otherwise it returns the empty string.
func heading(line string) string {
	line = strings.TrimSpace(line)
	if len(line) == 0 {
		return ""
	}

	// a heading must start with an uppercase letter
	r, _ := utf8.DecodeRuneInString(line)
	if !unicode.IsLetter(r) || !unicode.IsUpper(r) {
		return ""
	}

	// it must end in a letter or digit:
	r, _ = utf8.DecodeLastRuneInString(line)
	if !unicode.IsLetter(r) && !unicode.IsDigit(r) {
		return ""
	}

	// exclude lines with illegal characters. we allow "(),"
	if strings.ContainsAny(line, ";:!?+*/=[]{}_^°&§~%#@<\">\\") {
		return ""
	}

	// allow "'" for possessive "'s" only
	for b := line; ; {
		i := strings.IndexRune(b, '\'')
		if i < 0 {
			break
		}
		if i+1 >= len(b) || b[i+1] != 's' || (i+2 < len(b) && b[i+2] != ' ') {
			return "" // not followed by "s "
		}
		b = b[i+2:]
	}

	// allow "." when followed by non-whiteSpace
	for b := line; ; {
		i := strings.IndexRune(b, '.')
		if i < 0 {
			break
		}
		if i+1 >= len(b) || b[i+1] == ' ' {
			return "" // not followed by non-whiteSpace
		}
		b = b[i+1:]
	}

	return line
}
