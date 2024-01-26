// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package markdown

import (
	"bytes"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
)

/*
text node can be

 - other literal text
 - run of * or _ characters
 - [
 - ![

keep delimiter stack pointing at non-other literal text
each node contains

 - type of delimiter [ ![ _ *
 - number of delimiters
 - active or not
 - potential opener, potential closer, or obth

when a ] is hit, call look for link or image
when end is hit, call process emphasis

look for link or image:

	find topmost [ or ![
	if none, emit literal ]
	if its inactive, remove and emit literal ]
	parse ahead to look for rest of link; if none, remove and emit literal ]
	run process emphasis on the interior,
	remove opener
	if this was a link (not an image), set all [ before opener to inactive, to avoid links inside links

process emphasis

	walk forward in list to find a closer.
	walk back to find first potential matching opener.
	if found:
		strong for length >= 2
		insert node
		drop delimiters between opener and closer
		remove 1 or 2 from open/close count, removing if now empty
		if closing has some left, go around again on this node
	if not:
		set openers bottom for this kind of element to before current_position
		if the closer at current pos is not an opener, remove it

seems needlessly complex. two passes

scan and find ` ` first.

pass 1. scan and find [ and ]() and leave the rest alone.

each completed one invokes emphasis on inner text and then on the overall list.

*/

type Inline interface {
	PrintHTML(*bytes.Buffer)
	PrintText(*bytes.Buffer)
	printMarkdown(*bytes.Buffer)
}

type Plain struct {
	Text string
}

func (*Plain) Inline() {}

func (x *Plain) PrintHTML(buf *bytes.Buffer) {
	htmlEscaper.WriteString(buf, x.Text)
}

func (x *Plain) printMarkdown(buf *bytes.Buffer) {
	buf.WriteString(x.Text)
}

func (x *Plain) PrintText(buf *bytes.Buffer) {
	htmlEscaper.WriteString(buf, x.Text)
}

type openPlain struct {
	Plain
	i int // position in input where bracket is
}

type emphPlain struct {
	Plain
	canOpen  bool
	canClose bool
	i        int // position in output where emph is
	n        int // length of original span
}

type Escaped struct {
	Plain
}

func (x *Escaped) printMarkdown(buf *bytes.Buffer) {
	buf.WriteByte('\\')
	x.Plain.printMarkdown(buf)
}

type Code struct {
	Text     string
	numTicks int
}

func (*Code) Inline() {}

func (x *Code) PrintHTML(buf *bytes.Buffer) {
	fmt.Fprintf(buf, "<code>%s</code>", htmlEscaper.Replace(x.Text))
}

func (x *Code) printMarkdown(buf *bytes.Buffer) {
	ticks := strings.Repeat("`", x.numTicks)
	buf.WriteString(ticks)
	buf.WriteString(x.Text)
	buf.WriteString(ticks)
}

func (x *Code) PrintText(buf *bytes.Buffer) {
	htmlEscaper.WriteString(buf, x.Text)
}

type Strong struct {
	Marker string
	Inner  []Inline
}

func (x *Strong) Inline() {
}

func (x *Strong) PrintHTML(buf *bytes.Buffer) {
	buf.WriteString("<strong>")
	for _, c := range x.Inner {
		c.PrintHTML(buf)
	}
	buf.WriteString("</strong>")
}

func (x *Strong) printMarkdown(buf *bytes.Buffer) {
	buf.WriteString(x.Marker)
	for _, c := range x.Inner {
		c.printMarkdown(buf)
	}
	buf.WriteString(x.Marker)
}

func (x *Strong) PrintText(buf *bytes.Buffer) {
	for _, c := range x.Inner {
		c.PrintText(buf)
	}
}

type Del struct {
	Marker string
	Inner  []Inline
}

func (x *Del) Inline() {

}

func (x *Del) PrintHTML(buf *bytes.Buffer) {
	buf.WriteString("<del>")
	for _, c := range x.Inner {
		c.PrintHTML(buf)
	}
	buf.WriteString("</del>")
}

func (x *Del) printMarkdown(buf *bytes.Buffer) {
	buf.WriteString(x.Marker)
	for _, c := range x.Inner {
		c.printMarkdown(buf)
	}
	buf.WriteString(x.Marker)
}

func (x *Del) PrintText(buf *bytes.Buffer) {
	for _, c := range x.Inner {
		c.PrintText(buf)
	}
}

type Emph struct {
	Marker string
	Inner  []Inline
}

func (*Emph) Inline() {}

func (x *Emph) PrintHTML(buf *bytes.Buffer) {
	buf.WriteString("<em>")
	for _, c := range x.Inner {
		c.PrintHTML(buf)
	}
	buf.WriteString("</em>")
}

func (x *Emph) printMarkdown(buf *bytes.Buffer) {
	buf.WriteString(x.Marker)
	for _, c := range x.Inner {
		c.printMarkdown(buf)
	}
	buf.WriteString(x.Marker)
}

func (x *Emph) PrintText(buf *bytes.Buffer) {
	for _, c := range x.Inner {
		c.PrintText(buf)
	}
}

func (p *parseState) emit(i int) {
	if p.emitted < i {
		p.list = append(p.list, &Plain{p.s[p.emitted:i]})
		p.emitted = i
	}
}

func (p *parseState) skip(i int) {
	p.emitted = i
}

func (p *parseState) inline(s string) []Inline {
	s = trimSpaceTab(s)
	// Scan text looking for inlines.
	// Leaf inlines are converted immediately.
	// Non-leaf inlines have potential starts pushed on a stack while we await completion.
	// Links take priority over other emphasis, so the emphasis must be delayed.
	p.s = s
	p.list = nil
	p.emitted = 0
	var opens []int // indexes of open ![ and [ Plains in p.list
	var lastLinkOpen int
	backticks := false
	i := 0
	for i < len(s) {
		var parser func(*parseState, string, int) (Inline, int, int, bool)
		switch s[i] {
		case '\\':
			parser = parseEscape
		case '`':
			if !backticks {
				backticks = true
				p.backticks.reset()
			}
			parser = p.backticks.parseCodeSpan
		case '<':
			parser = parseAutoLinkOrHTML
		case '[':
			parser = parseLinkOpen
		case '!':
			parser = parseImageOpen
		case '_', '*':
			parser = parseEmph
		case '.':
			if p.SmartDot {
				parser = parseDot
			}
		case '-':
			if p.SmartDash {
				parser = parseDash
			}
		case '"', '\'':
			if p.SmartQuote {
				parser = parseEmph
			}
		case '~':
			if p.Strikethrough {
				parser = parseEmph
			}
		case '\n': // TODO what about eof
			parser = parseBreak
		case '&':
			parser = parseHTMLEntity
		case ':':
			if p.Emoji {
				parser = parseEmoji
			}
		}
		if parser != nil {
			if x, start, end, ok := parser(p, s, i); ok {
				p.emit(start)
				if _, ok := x.(*openPlain); ok {
					opens = append(opens, len(p.list))
				}
				p.list = append(p.list, x)
				i = end
				p.skip(i)
				continue
			}
		}
		if s[i] == ']' && len(opens) > 0 {
			oi := opens[len(opens)-1]
			open := p.list[oi].(*openPlain)
			opens = opens[:len(opens)-1]
			if open.Text[0] == '!' || lastLinkOpen <= open.i {
				if x, end, ok := p.parseLinkClose(s, i, open); ok {
					p.corner = p.corner || x.corner || linkCorner(x.URL)
					p.emit(i)
					x.Inner = p.emph(nil, p.list[oi+1:])
					if open.Text[0] == '!' {
						p.list[oi] = (*Image)(x)
					} else {
						p.list[oi] = x
					}
					p.list = p.list[:oi+1]
					p.skip(end)
					i = end
					if open.Text[0] == '[' {
						// No links around links.
						lastLinkOpen = open.i
					}
					continue
				}
			}
		}
		i++
	}
	p.emit(len(s))
	p.list = p.emph(p.list[:0], p.list)
	p.list = p.mergePlain(p.list)
	p.list = p.autoLinkText(p.list)

	return p.list
}

func (ps *parseState) emph(dst, src []Inline) []Inline {
	const chars = "_*~\"'"
	var stack [len(chars)][]*emphPlain
	stackOf := func(c byte) int {
		return strings.IndexByte(chars, c)
	}

	trimStack := func() {
		for i := range stack {
			stk := &stack[i]
			for len(*stk) > 0 && (*stk)[len(*stk)-1].i >= len(dst) {
				*stk = (*stk)[:len(*stk)-1]
			}
		}
	}

Src:
	for i := 0; i < len(src); i++ {
		if open, ok := src[i].(*openPlain); ok {
			// Convert unused link/image open marker to plain text.
			dst = append(dst, &open.Plain)
			continue
		}
		p, ok := src[i].(*emphPlain)
		if !ok {
			dst = append(dst, src[i])
			continue
		}
		if p.canClose {
			stk := &stack[stackOf(p.Text[0])]
		Loop:
			for p.Text != "" {
				// Looking for same symbol and compatible with p.Text.
				for i := len(*stk) - 1; i >= 0; i-- {
					start := (*stk)[i]
					if (p.Text[0] == '*' || p.Text[0] == '_') && (p.canOpen && p.canClose || start.canOpen && start.canClose) && (p.n+start.n)%3 == 0 && (p.n%3 != 0 || start.n%3 != 0) {
						continue
					}
					if p.Text[0] == '~' && len(p.Text) != len(start.Text) { // ~ matches ~, ~~ matches ~~
						continue
					}
					if p.Text[0] == '"' {
						dst[start.i].(*emphPlain).Text = "“"
						p.Text = "”"
						dst = append(dst, p)
						*stk = (*stk)[:i]
						// no trimStack
						continue Src
					}
					if p.Text[0] == '\'' {
						dst[start.i].(*emphPlain).Text = "‘"
						p.Text = "’"
						dst = append(dst, p)
						*stk = (*stk)[:i]
						// no trimStack
						continue Src
					}
					var d int
					if len(p.Text) >= 2 && len(start.Text) >= 2 {
						// strong
						d = 2
					} else {
						// emph
						d = 1
					}
					del := p.Text[0] == '~'
					x := &Emph{Marker: p.Text[:d], Inner: append([]Inline(nil), dst[start.i+1:]...)}
					start.Text = start.Text[:len(start.Text)-d]
					p.Text = p.Text[d:]
					if start.Text == "" {
						dst = dst[:start.i]
					} else {
						dst = dst[:start.i+1]
					}
					trimStack()
					if del {
						dst = append(dst, (*Del)(x))
					} else if d == 2 {
						dst = append(dst, (*Strong)(x))
					} else {
						dst = append(dst, x)
					}
					continue Loop
				}
				break
			}
		}
		if p.Text != "" {
			stk := &stack[stackOf(p.Text[0])]
			if p.Text == "'" {
				p.Text = "’"
			}
			if p.Text == "\"" {
				if p.canClose {
					p.Text = "”"
				} else {
					p.Text = "“"
				}
			}
			if p.canOpen {
				p.i = len(dst)
				dst = append(dst, p)
				*stk = append(*stk, p)
			} else {
				dst = append(dst, &p.Plain)
			}
		}
	}
	return dst
}

func mdUnescape(s string) string {
	if !strings.Contains(s, `\`) && !strings.Contains(s, `&`) {
		return s
	}
	return mdUnescaper.Replace(s)
}

var mdUnescaper = func() *strings.Replacer {
	var list = []string{
		`\!`, `!`,
		`\"`, `"`,
		`\#`, `#`,
		`\$`, `$`,
		`\%`, `%`,
		`\&`, `&`,
		`\'`, `'`,
		`\(`, `(`,
		`\)`, `)`,
		`\*`, `*`,
		`\+`, `+`,
		`\,`, `,`,
		`\-`, `-`,
		`\.`, `.`,
		`\/`, `/`,
		`\:`, `:`,
		`\;`, `;`,
		`\<`, `<`,
		`\=`, `=`,
		`\>`, `>`,
		`\?`, `?`,
		`\@`, `@`,
		`\[`, `[`,
		`\\`, `\`,
		`\]`, `]`,
		`\^`, `^`,
		`\_`, `_`,
		"\\`", "`",
		`\{`, `{`,
		`\|`, `|`,
		`\}`, `}`,
		`\~`, `~`,
	}

	for name, repl := range htmlEntity {
		list = append(list, name, repl)
	}
	return strings.NewReplacer(list...)
}()

func isPunct(c byte) bool {
	return '!' <= c && c <= '/' || ':' <= c && c <= '@' || '[' <= c && c <= '`' || '{' <= c && c <= '~'
}

func parseEscape(p *parseState, s string, i int) (Inline, int, int, bool) {
	if i+1 < len(s) {
		c := s[i+1]
		if isPunct(c) {
			return &Escaped{Plain{s[i+1 : i+2]}}, i, i + 2, true
		}
		if c == '\n' { // TODO what about eof
			if i > 0 && s[i-1] == '\\' {
				p.corner = true // goldmark mishandles \\\ newline
			}
			end := i + 2
			for end < len(s) && (s[end] == ' ' || s[end] == '\t') {
				end++
			}
			return &HardBreak{}, i, end, true
		}
	}
	return nil, 0, 0, false
}

func parseDot(p *parseState, s string, i int) (Inline, int, int, bool) {
	if i+2 < len(s) && s[i+1] == '.' && s[i+2] == '.' {
		return &Plain{"…"}, i, i + 3, true
	}
	return nil, 0, 0, false
}

func parseDash(p *parseState, s string, i int) (Inline, int, int, bool) {
	if i+1 >= len(s) || s[i+1] != '-' {
		return nil, 0, 0, false
	}

	n := 2
	for i+n < len(s) && s[i+n] == '-' {
		n++
	}

	// Mimic cmark-gfm. Can't make this stuff up.
	em, en := 0, 0
	switch {
	case n%3 == 0:
		em = n / 3
	case n%2 == 0:
		en = n / 2
	case n%3 == 2:
		em = (n - 2) / 3
		en = 1
	case n%3 == 1:
		em = (n - 4) / 3
		en = 2
	}
	return &Plain{strings.Repeat("—", em) + strings.Repeat("–", en)}, i, i + n, true
}

// Inline code span markers must fit on punched cards, to match cmark-gfm.
const maxBackticks = 80

type backtickParser struct {
	last    [maxBackticks]int
	scanned bool
}

func (b *backtickParser) reset() {
	*b = backtickParser{}
}

func (b *backtickParser) parseCodeSpan(p *parseState, s string, i int) (Inline, int, int, bool) {
	start := i
	// Count leading backticks. Need to find that many again.
	n := 1
	for i+n < len(s) && s[i+n] == '`' {
		n++
	}

	// If we've already scanned the whole string (for a different count),
	// we can skip a failed scan by checking whether we saw this count.
	// To enable this optimization, following cmark-gfm, we declare by fiat
	// that more than maxBackticks backquotes is too many.
	if n > len(b.last) || b.scanned && b.last[n-1] < i+n {
		goto NoMatch
	}

	for end := i + n; end < len(s); {
		if s[end] != '`' {
			end++
			continue
		}
		estart := end
		for end < len(s) && s[end] == '`' {
			end++
		}
		m := end - estart
		if !b.scanned && m < len(b.last) {
			b.last[m-1] = estart
		}
		if m == n {
			// Match.
			// Line endings are converted to single spaces.
			text := s[i+n : estart]
			text = strings.ReplaceAll(text, "\n", " ")

			// If enclosed text starts and ends with a space and is not all spaces,
			// one space is removed from start and end, to allow `` ` `` to quote a single backquote.
			if len(text) >= 2 && text[0] == ' ' && text[len(text)-1] == ' ' && trimSpace(text) != "" {
				text = text[1 : len(text)-1]
			}

			return &Code{text, n}, start, end, true
		}
	}
	b.scanned = true

NoMatch:
	// No match, so none of these backticks count: skip them all.
	// For example ``x` is not a single backtick followed by a code span.
	// Returning nil, 0, false would advance to the second backtick and try again.
	return &Plain{s[i : i+n]}, start, i + n, true
}

func parseAutoLinkOrHTML(p *parseState, s string, i int) (Inline, int, int, bool) {
	if x, end, ok := parseAutoLinkURI(s, i); ok {
		return x, i, end, true
	}
	if x, end, ok := parseAutoLinkEmail(s, i); ok {
		return x, i, end, true
	}
	if x, end, ok := parseHTMLTag(p, s, i); ok {
		return x, i, end, true
	}
	return nil, 0, 0, false
}

func isLetter(c byte) bool {
	return 'A' <= c && c <= 'Z' || 'a' <= c && c <= 'z'
}

func isLDH(c byte) bool {
	return isLetterDigit(c) || c == '-'
}

func isLetterDigit(c byte) bool {
	return 'A' <= c && c <= 'Z' || 'a' <= c && c <= 'z' || '0' <= c && c <= '9'
}

func parseLinkOpen(_ *parseState, s string, i int) (Inline, int, int, bool) {
	return &openPlain{Plain{s[i : i+1]}, i + 1}, i, i + 1, true
}

func parseImageOpen(_ *parseState, s string, i int) (Inline, int, int, bool) {
	if i+1 < len(s) && s[i+1] == '[' {
		return &openPlain{Plain{s[i : i+2]}, i + 2}, i, i + 2, true
	}
	return nil, 0, 0, false
}

func parseEmph(p *parseState, s string, i int) (Inline, int, int, bool) {
	c := s[i]
	j := i + 1
	if c == '*' || c == '~' || c == '_' {
		for j < len(s) && s[j] == c {
			j++
		}
	}
	if c == '~' && j-i != 2 {
		// Goldmark does not accept ~text~
		// and incorrectly accepts ~~~text~~~.
		// Only ~~ is correct.
		p.corner = true
	}
	if c == '~' && j-i > 2 {
		return &Plain{s[i:j]}, i, j, true
	}

	var before, after rune
	if i == 0 {
		before = ' '
	} else {
		before, _ = utf8.DecodeLastRuneInString(s[:i])
	}
	if j >= len(s) {
		after = ' '
	} else {
		after, _ = utf8.DecodeRuneInString(s[j:])
	}

	// “A left-flanking delimiter run is a delimiter run that is
	// (1) not followed by Unicode whitespace, and either
	// (2a) not followed by a Unicode punctuation character, or
	// (2b) followed by a Unicode punctuation character
	// and preceded by Unicode whitespace or a Unicode punctuation character.
	// For purposes of this definition, the beginning and the end
	// of the line count as Unicode whitespace.”
	leftFlank := !isUnicodeSpace(after) &&
		(!isUnicodePunct(after) || isUnicodeSpace(before) || isUnicodePunct(before))

	// “A right-flanking delimiter run is a delimiter run that is
	// (1) not preceded by Unicode whitespace, and either
	// (2a) not preceded by a Unicode punctuation character, or
	// (2b) preceded by a Unicode punctuation character
	// and followed by Unicode whitespace or a Unicode punctuation character.
	// For purposes of this definition, the beginning and the end
	// of the line count as Unicode whitespace.”
	rightFlank := !isUnicodeSpace(before) &&
		(!isUnicodePunct(before) || isUnicodeSpace(after) || isUnicodePunct(after))

	var canOpen, canClose bool

	switch c {
	case '\'', '"':
		canOpen = leftFlank && !rightFlank && before != ']' && before != ')'
		canClose = rightFlank
	case '*', '~':
		// “A single * character can open emphasis iff
		// it is part of a left-flanking delimiter run.”

		// “A double ** can open strong emphasis iff
		// it is part of a left-flanking delimiter run.”
		canOpen = leftFlank

		// “A single * character can close emphasis iff
		// it is part of a right-flanking delimiter run.”

		// “A double ** can close strong emphasis iff
		// it is part of a right-flanking delimiter run.”
		canClose = rightFlank
	case '_':
		// “A single _ character can open emphasis iff
		// it is part of a left-flanking delimiter run and either
		// (a) not part of a right-flanking delimiter run or
		// (b) part of a right-flanking delimiter run preceded by a Unicode punctuation character.”

		// “A double __ can open strong emphasis iff
		// it is part of a left-flanking delimiter run and either
		// (a) not part of a right-flanking delimiter run or
		// (b) part of a right-flanking delimiter run preceded by a Unicode punctuation character.”
		canOpen = leftFlank && (!rightFlank || isUnicodePunct(before))

		// “A single _ character can close emphasis iff
		// it is part of a right-flanking delimiter run and either
		// (a) not part of a left-flanking delimiter run or
		// (b) part of a left-flanking delimiter run followed by a Unicode punctuation character.”

		// “A double __ can close strong emphasis iff
		// it is part of a right-flanking delimiter run and either
		// (a) not part of a left-flanking delimiter run or
		// (b) part of a left-flanking delimiter run followed by a Unicode punctuation character.”
		canClose = rightFlank && (!leftFlank || isUnicodePunct(after))
	}

	return &emphPlain{Plain: Plain{s[i:j]}, canOpen: canOpen, canClose: canClose, n: j - i}, i, j, true
}

func isUnicodeSpace(r rune) bool {
	if r < 0x80 {
		return r == ' ' || r == '\t' || r == '\f' || r == '\n'
	}
	return unicode.In(r, unicode.Zs)
}

func isUnicodePunct(r rune) bool {
	if r < 0x80 {
		return isPunct(byte(r))
	}
	return unicode.In(r, unicode.Punct)
}

func (p *parseState) parseLinkClose(s string, i int, open *openPlain) (*Link, int, bool) {
	if i+1 < len(s) {
		switch s[i+1] {
		case '(':
			// Inline link - [Text](Dest Title), with Title omitted or both Dest and Title omitted.
			i := skipSpace(s, i+2)
			var dest, title string
			var titleChar byte
			var corner bool
			if i < len(s) && s[i] != ')' {
				var ok bool
				dest, i, ok = parseLinkDest(s, i)
				if !ok {
					break
				}
				i = skipSpace(s, i)
				if i < len(s) && s[i] != ')' {
					title, titleChar, i, ok = parseLinkTitle(s, i)
					if title == "" {
						corner = true
					}
					if !ok {
						break
					}
					i = skipSpace(s, i)
				}
			}
			if i < len(s) && s[i] == ')' {
				return &Link{URL: dest, Title: title, TitleChar: titleChar, corner: corner}, i + 1, true
			}
			// NOTE: Test malformed ( ) with shortcut reference
			// TODO fall back on syntax error?

		case '[':
			// Full reference link - [Text][Label]
			label, i, ok := parseLinkLabel(p, s, i+1)
			if !ok {
				break
			}
			if link, ok := p.links[normalizeLabel(label)]; ok {
				return &Link{URL: link.URL, Title: link.Title, corner: link.corner}, i, true
			}
			// Note: Could break here, but CommonMark dingus does not
			// fall back to trying Text for [Text][Label] when Label is unknown.
			// Unclear from spec what the correct answer is.
			return nil, 0, false
		}
	}

	// Collapsed or shortcut reference link: [Text][] or [Text].
	end := i + 1
	if strings.HasPrefix(s[end:], "[]") {
		end += 2
	}

	if link, ok := p.links[normalizeLabel(s[open.i:i])]; ok {
		return &Link{URL: link.URL, Title: link.Title, corner: link.corner}, end, true
	}
	return nil, 0, false
}

func skipSpace(s string, i int) int {
	// Note: Blank lines have already been removed.
	for i < len(s) && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n') {
		i++
	}
	return i
}

func linkCorner(url string) bool {
	for i := 0; i < len(url); i++ {
		if url[i] == '%' {
			if i+2 >= len(url) || !isHexDigit(url[i+1]) || !isHexDigit(url[i+2]) {
				// Goldmark and the Dingus re-escape such percents as %25,
				// but the spec does not seem to require this behavior.
				return true
			}
		}
	}
	return false
}

func (p *parseState) mergePlain(list []Inline) []Inline {
	out := list[:0]
	start := 0
	for i := 0; ; i++ {
		if i < len(list) && toPlain(list[i]) != nil {
			continue
		}
		// Non-Plain or end of list.
		if start < i {
			out = append(out, mergePlain1(list[start:i]))
		}
		if i >= len(list) {
			break
		}
		out = append(out, list[i])
		start = i + 1
	}
	return out
}

func toPlain(x Inline) *Plain {
	// TODO what about Escaped?
	switch x := x.(type) {
	case *Plain:
		return x
	case *emphPlain:
		return &x.Plain
	case *openPlain:
		return &x.Plain
	}
	return nil
}

func mergePlain1(list []Inline) *Plain {
	if len(list) == 1 {
		return toPlain(list[0])
	}
	var all []string
	for _, pl := range list {
		all = append(all, toPlain(pl).Text)
	}
	return &Plain{Text: strings.Join(all, "")}
}

func parseEmoji(p *parseState, s string, i int) (Inline, int, int, bool) {
	for j := i + 1; ; j++ {
		if j >= len(s) || j-i > 2+maxEmojiLen {
			break
		}
		if s[j] == ':' {
			name := s[i+1 : j]
			if utf, ok := emoji[name]; ok {
				return &Emoji{s[i : j+1], utf}, i, j + 1, true
			}
			break
		}
	}
	return nil, 0, 0, false
}

type Emoji struct {
	Name string // emoji :name:, including colons
	Text string // Unicode for emoji sequence
}

func (*Emoji) Inline() {}

func (x *Emoji) PrintHTML(buf *bytes.Buffer) {
	htmlEscaper.WriteString(buf, x.Text)
}

func (x *Emoji) printMarkdown(buf *bytes.Buffer) {
	buf.WriteString(x.Text)
}

func (x *Emoji) PrintText(buf *bytes.Buffer) {
	htmlEscaper.WriteString(buf, x.Text)
}
