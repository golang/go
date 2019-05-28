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

// Escape comment text for HTML. If nice is set,
// also turn `` into &ldquo; and '' into &rdquo;.
func commentEscape(w io.Writer, text string, nice bool) {
	if nice {
		// In the first pass, we convert `` and '' into their unicode equivalents.
		// This prevents them from being escaped in HTMLEscape.
		text = convertQuotes(text)
		var buf bytes.Buffer
		template.HTMLEscape(&buf, []byte(text))
		// Now we convert the unicode quotes to their HTML escaped entities to maintain old behavior.
		// We need to use a temp buffer to read the string back and do the conversion,
		// otherwise HTMLEscape will escape & to &amp;
		htmlQuoteReplacer.WriteString(w, buf.String())
		return
	}
	template.HTMLEscape(w, []byte(text))
}

func convertQuotes(text string) string {
	return unicodeQuoteReplacer.Replace(text)
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

var (
	html_a      = []byte(`<a href="`)
	html_aq     = []byte(`">`)
	html_enda   = []byte("</a>")
	html_i      = []byte("<i>")
	html_endi   = []byte("</i>")
	html_p      = []byte("<p>\n")
	html_endp   = []byte("</p>\n")
	html_pre    = []byte("<pre>")
	html_endpre = []byte("</pre>\n")
	html_h      = []byte(`<h3 id="`)
	html_hq     = []byte(`">`)
	html_endh   = []byte("</h3>\n")
)

// Emphasize and escape a line of text for HTML. URLs are converted into links;
// if the URL also appears in the words map, the link is taken from the map (if
// the corresponding map value is the empty string, the URL is not converted
// into a link). Go identifiers that appear in the words map are italicized; if
// the corresponding map value is not the empty string, it is considered a URL
// and the word is converted into a link. If nice is set, the remaining text's
// appearance is improved where it makes sense (e.g., `` is turned into &ldquo;
// and '' into &rdquo;).
func emphasize(w io.Writer, line string, words map[string]string, nice bool) {
	for {
		m := matchRx.FindStringSubmatchIndex(line)
		if m == nil {
			break
		}
		// m >= 6 (two parenthesized sub-regexps in matchRx, 1st one is urlRx)

		// write text before match
		commentEscape(w, line[0:m[0]], nice)

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

		// write match
		if len(url) > 0 {
			w.Write(html_a)
			template.HTMLEscape(w, []byte(url))
			w.Write(html_aq)
		}
		if italics {
			w.Write(html_i)
		}
		commentEscape(w, match, nice)
		if italics {
			w.Write(html_endi)
		}
		if len(url) > 0 {
			w.Write(html_enda)
		}

		// advance
		line = line[m[1]:]
	}
	commentEscape(w, line, nice)
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

	// allow "." when followed by non-space
	for b := line; ; {
		i := strings.IndexRune(b, '.')
		if i < 0 {
			break
		}
		if i+1 >= len(b) || b[i+1] == ' ' {
			return "" // not followed by non-space
		}
		b = b[i+1:]
	}

	return line
}

type op int

const (
	opPara op = iota
	opHead
	opPre
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
	for _, b := range blocks(text) {
		switch b.op {
		case opPara:
			w.Write(html_p)
			for _, line := range b.lines {
				emphasize(w, line, words, true)
			}
			w.Write(html_endp)
		case opHead:
			w.Write(html_h)
			id := ""
			for _, line := range b.lines {
				if id == "" {
					id = anchorID(line)
					w.Write([]byte(id))
					w.Write(html_hq)
				}
				commentEscape(w, line, true)
			}
			if id == "" {
				w.Write(html_hq)
			}
			w.Write(html_endh)
		case opPre:
			w.Write(html_pre)
			for _, line := range b.lines {
				emphasize(w, line, nil, false)
			}
			w.Write(html_endpre)
		}
	}
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
			out = append(out, block{opPre, pre})
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

// ToText prepares comment text for presentation in textual output.
// It wraps paragraphs of text to width or fewer Unicode code points
// and then prefixes each line with the indent. In preformatted sections
// (such as program text), it prefixes each non-blank line with preIndent.
func ToText(w io.Writer, text string, indent, preIndent string, width int) {
	l := lineWrapper{
		out:    w,
		width:  width,
		indent: indent,
	}
	for _, b := range blocks(text) {
		switch b.op {
		case opPara:
			// l.write will add leading newline if required
			for _, line := range b.lines {
				line = convertQuotes(line)
				l.write(line)
			}
			l.flush()
		case opHead:
			w.Write(nl)
			for _, line := range b.lines {
				line = convertQuotes(line)
				l.write(line + "\n")
			}
			l.flush()
		case opPre:
			w.Write(nl)
			for _, line := range b.lines {
				if isBlank(line) {
					w.Write([]byte("\n"))
				} else {
					w.Write([]byte(preIndent))
					w.Write([]byte(line))
				}
			}
		}
	}
}

type lineWrapper struct {
	out       io.Writer
	printed   bool
	width     int
	indent    string
	n         int
	pendSpace int
}

var nl = []byte("\n")
var space = []byte(" ")
var prefix = []byte("// ")

func (l *lineWrapper) write(text string) {
	if l.n == 0 && l.printed {
		l.out.Write(nl) // blank line before new paragraph
	}
	l.printed = true

	needsPrefix := false
	isComment := strings.HasPrefix(text, "//")
	for _, f := range strings.Fields(text) {
		w := utf8.RuneCountInString(f)
		// wrap if line is too long
		if l.n > 0 && l.n+l.pendSpace+w > l.width {
			l.out.Write(nl)
			l.n = 0
			l.pendSpace = 0
			needsPrefix = isComment
		}
		if l.n == 0 {
			l.out.Write([]byte(l.indent))
		}
		if needsPrefix {
			l.out.Write(prefix)
			needsPrefix = false
		}
		l.out.Write(space[:l.pendSpace])
		l.out.Write([]byte(f))
		l.n += l.pendSpace + w
		l.pendSpace = 1
	}
}

func (l *lineWrapper) flush() {
	if l.n == 0 {
		return
	}
	l.out.Write(nl)
	l.pendSpace = 0
	l.n = 0
}
