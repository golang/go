// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package markdown

import (
	"bytes"
	"strconv"
	"strings"
	"unicode"
)

type HTMLBlock struct {
	Position
	Text []string
}

func (b *HTMLBlock) PrintHTML(buf *bytes.Buffer) {
	for _, s := range b.Text {
		buf.WriteString(s)
		buf.WriteString("\n")
	}
}

func (b *HTMLBlock) printMarkdown(buf *bytes.Buffer, s mdState) {
	if s.prefix1 != "" {
		buf.WriteString(s.prefix1)
	} else {
		buf.WriteString(s.prefix)
	}
	b.PrintHTML(buf)
}

type htmlBuilder struct {
	endBlank bool
	text     []string
	endFunc  func(string) bool
}

func (c *htmlBuilder) extend(p *parseState, s line) (line, bool) {
	if c.endBlank && s.isBlank() {
		return s, false
	}
	t := s.string()
	c.text = append(c.text, t)
	if c.endFunc != nil && c.endFunc(t) {
		return line{}, false
	}
	return line{}, true
}

func (c *htmlBuilder) build(p buildState) Block {
	return &HTMLBlock{
		p.pos(),
		c.text,
	}
}

func newHTML(p *parseState, s line) (line, bool) {
	peek := s
	if p.startHTML(&peek) {
		return line{}, true
	}
	return s, false
}

func (p *parseState) startHTML(s *line) bool {
	tt := *s
	tt.trimSpace(0, 3, false)
	if tt.peek() != '<' {
		return false
	}
	t := tt.string()

	var end string
	switch {
	case strings.HasPrefix(t, "<!--"):
		end = "-->"
	case strings.HasPrefix(t, "<?"):
		end = "?>"
	case strings.HasPrefix(t, "<![CDATA["):
		end = "]]>"
	case strings.HasPrefix(t, "<!") && len(t) >= 3 && isLetter(t[2]):
		if 'a' <= t[2] && t[2] <= 'z' {
			// Goldmark and the Dingus only accept <!UPPER> not <!lower>.
			p.corner = true
		}
		end = ">"
	}
	if end != "" {
		b := &htmlBuilder{endFunc: func(s string) bool { return strings.Contains(s, end) }}
		p.addBlock(b)
		b.text = append(b.text, s.string())
		if b.endFunc(t) {
			p.closeBlock()
		}
		return true
	}

	// case 6
	i := 1
	if i < len(t) && t[i] == '/' {
		i++
	}
	buf := make([]byte, 0, 16)
	for ; i < len(t) && len(buf) < 16; i++ {
		c := t[i]
		if 'A' <= c && c <= 'Z' {
			c += 'a' - 'A'
		}
		if !('a' <= c && c <= 'z') && !('0' <= c && c <= '9') {
			break
		}
		buf = append(buf, c)
	}
	var sep byte
	if i < len(t) {
		switch t[i] {
		default:
			goto Next
		case ' ', '\t', '>':
			// ok
			sep = t[i]
		case '/':
			if i+1 >= len(t) || t[i+1] != '>' {
				goto Next
			}
		}
	}

	if len(buf) == 0 {
		goto Next
	}
	{
		c := buf[0]
		var ok bool
		for _, name := range htmlTags {
			if name[0] == c && len(name) == len(buf) && name == string(buf) {
				if sep == '\t' {
					// Goldmark recognizes space here but not tab.
					// testdata/extra.txt 143.md
					p.corner = true
				}
				ok = true
				break
			}
		}
		if !ok {
			goto Next
		}
	}

	{
		b := &htmlBuilder{endBlank: true}
		p.addBlock(b)
		b.text = append(b.text, s.string())
		return true
	}

Next:
	// case 1
	if len(t) > 1 && t[1] != '/' && (i >= len(t) || t[i] == ' ' || t[i] == '\t' || t[i] == '>') {
		switch string(buf) {
		case "pre", "script", "style", "textarea":
			b := &htmlBuilder{endFunc: hasEndPre}
			p.addBlock(b)
			b.text = append(b.text, s.string())
			if hasEndPre(t) {
				p.closeBlock()
			}
			return true
		}
	}

	// case 7
	if p.para() == nil {
		if _, e, ok := parseHTMLOpenTag(p, t, 0); ok && skipSpace(t, e) == len(t) {
			if e != len(t) {
				// Goldmark disallows trailing space
				p.corner = true
			}
			b := &htmlBuilder{endBlank: true}
			p.addBlock(b)
			b.text = append(b.text, s.string())
			return true
		}
		if _, e, ok := parseHTMLClosingTag(p, t, 0); ok && skipSpace(t, e) == len(t) {
			b := &htmlBuilder{endBlank: true}
			p.addBlock(b)
			b.text = append(b.text, s.string())
			return true
		}
	}

	return false
}

func hasEndPre(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] == '<' && i+1 < len(s) && s[i+1] == '/' {
			buf := make([]byte, 0, 8)
			for i += 2; i < len(s) && len(buf) < 8; i++ {
				c := s[i]
				if 'A' <= c && c <= 'Z' {
					c += 'a' - 'A'
				}
				if c < 'a' || 'z' < c {
					break
				}
				buf = append(buf, c)
			}
			if i < len(s) && s[i] == '>' {
				switch string(buf) {
				case "pre", "script", "style", "textarea":
					return true
				}
			}
		}
	}
	return false
}

func parseHTMLTag(p *parseState, s string, i int) (Inline, int, bool) {
	// “An HTML tag consists of an open tag, a closing tag, an HTML comment,
	// a processing instruction, a declaration, or a CDATA section.”
	if i+3 <= len(s) && s[i] == '<' {
		switch s[i+1] {
		default:
			return parseHTMLOpenTag(p, s, i)
		case '/':
			return parseHTMLClosingTag(p, s, i)
		case '!':
			switch s[i+2] {
			case '-':
				return parseHTMLComment(s, i)
			case '[':
				return parseHTMLCDATA(s, i)
			default:
				return parseHTMLDecl(p, s, i)
			}
		case '?':
			return parseHTMLProcInst(s, i)
		}
	}
	return nil, 0, false
}

func parseHTMLOpenTag(p *parseState, s string, i int) (Inline, int, bool) {
	if i >= len(s) || s[i] != '<' {
		return nil, 0, false
	}
	// “An open tag consists of a < character, a tag name, zero or more attributes,
	// optional spaces, tabs, and up to one line ending, an optional / character, and a > character.”
	if name, j, ok := parseTagName(s, i+1); ok {
		switch name {
		case "pre", "script", "style", "textarea":
			// Goldmark treats these as starting a new HTMLBlock
			// and ending the paragraph they appear in.
			p.corner = true
		}
		for {
			if j >= len(s) || s[j] != ' ' && s[j] != '\t' && s[j] != '\n' && s[j] != '/' && s[j] != '>' {
				return nil, 0, false
			}
			_, k, ok := parseAttr(p, s, j)
			if !ok {
				break
			}
			j = k
		}
		k := skipSpace(s, j)
		if k != j {
			// Goldmark mishandles spaces before >.
			p.corner = true
		}
		j = k
		if j < len(s) && s[j] == '/' {
			j++
		}
		if j < len(s) && s[j] == '>' {
			return &HTMLTag{s[i : j+1]}, j + 1, true
		}
	}
	return nil, 0, false
}

func parseHTMLClosingTag(p *parseState, s string, i int) (Inline, int, bool) {
	// “A closing tag consists of the string </, a tag name,
	// optional spaces, tabs, and up to one line ending, and the character >.”
	if i+2 >= len(s) || s[i] != '<' || s[i+1] != '/' {
		return nil, 0, false
	}
	if skipSpace(s, i+2) != i+2 {
		// Goldmark allows spaces here but the spec and the Dingus do not.
		p.corner = true
	}

	if _, j, ok := parseTagName(s, i+2); ok {
		j = skipSpace(s, j)
		if j < len(s) && s[j] == '>' {
			return &HTMLTag{s[i : j+1]}, j + 1, true
		}
	}
	return nil, 0, false
}

func parseTagName(s string, i int) (string, int, bool) {
	// “A tag name consists of an ASCII letter followed by zero or more ASCII letters, digits, or hyphens (-).”
	if i < len(s) && isLetter(s[i]) {
		j := i + 1
		for j < len(s) && isLDH(s[j]) {
			j++
		}
		return s[i:j], j, true
	}
	return "", 0, false
}

func parseAttr(p *parseState, s string, i int) (string, int, bool) {
	// “An attribute consists of spaces, tabs, and up to one line ending,
	// an attribute name, and an optional attribute value specification.”
	i = skipSpace(s, i)
	if _, j, ok := parseAttrName(s, i); ok {
		if _, k, ok := parseAttrValueSpec(p, s, j); ok {
			j = k
		}
		return s[i:j], j, true
	}
	return "", 0, false
}

func parseAttrName(s string, i int) (string, int, bool) {
	// “An attribute name consists of an ASCII letter, _, or :,
	// followed by zero or more ASCII letters, digits, _, ., :, or -.”
	if i+1 < len(s) && (isLetter(s[i]) || s[i] == '_' || s[i] == ':') {
		j := i + 1
		for j < len(s) && (isLDH(s[j]) || s[j] == '_' || s[j] == '.' || s[j] == ':') {
			j++
		}
		return s[i:j], j, true
	}
	return "", 0, false
}

func parseAttrValueSpec(p *parseState, s string, i int) (string, int, bool) {
	// “An attribute value specification consists of
	// optional spaces, tabs, and up to one line ending,
	// a = character,
	// optional spaces, tabs, and up to one line ending,
	// and an attribute value.”
	i = skipSpace(s, i)
	if i+1 < len(s) && s[i] == '=' {
		i = skipSpace(s, i+1)
		if _, j, ok := parseAttrValue(s, i); ok {
			p.corner = p.corner || strings.Contains(s[i:j], "\ufffd")
			return s[i:j], j, true
		}
	}
	return "", 0, false
}

func parseAttrValue(s string, i int) (string, int, bool) {
	// “An attribute value consists of
	// an unquoted attribute value,
	// a single-quoted attribute value,
	// or a double-quoted attribute value.”
	// TODO: No escaping???
	if i < len(s) && (s[i] == '\'' || s[i] == '"') {
		// “A single-quoted attribute value consists of ',
		// zero or more characters not including ', and a final '.”
		// “A double-quoted attribute value consists of ",
		// zero or more characters not including ", and a final ".”
		if j := strings.IndexByte(s[i+1:], s[i]); j >= 0 {
			end := i + 1 + j + 1
			return s[i:end], end, true
		}
	}

	// “An unquoted attribute value is a nonempty string of characters
	// not including spaces, tabs, line endings, ", ', =, <, >, or `.”
	j := i
	for j < len(s) && strings.IndexByte(" \t\n\"'=<>`", s[j]) < 0 {
		j++
	}
	if j > i {
		return s[i:j], j, true
	}
	return "", 0, false
}

func parseHTMLComment(s string, i int) (Inline, int, bool) {
	// “An HTML comment consists of <!-- + text + -->,
	// where text does not start with > or ->,
	// does not end with -, and does not contain --.”
	if !strings.HasPrefix(s[i:], "<!-->") &&
		!strings.HasPrefix(s[i:], "<!--->") {
		if x, end, ok := parseHTMLMarker(s, i, "<!--", "-->"); ok {
			if t := x.(*HTMLTag).Text; !strings.Contains(t[len("<!--"):len(t)-len("->")], "--") {
				return x, end, ok
			}
		}
	}
	return nil, 0, false
}

func parseHTMLCDATA(s string, i int) (Inline, int, bool) {
	// “A CDATA section consists of the string <![CDATA[,
	// a string of characters not including the string ]]>, and the string ]]>.”
	return parseHTMLMarker(s, i, "<![CDATA[", "]]>")
}

func parseHTMLDecl(p *parseState, s string, i int) (Inline, int, bool) {
	// “A declaration consists of the string <!, an ASCII letter,
	// zero or more characters not including the character >, and the character >.”
	if i+2 < len(s) && isLetter(s[i+2]) {
		if 'a' <= s[i+2] && s[i+2] <= 'z' {
			p.corner = true // goldmark requires uppercase
		}
		return parseHTMLMarker(s, i, "<!", ">")
	}
	return nil, 0, false
}

func parseHTMLProcInst(s string, i int) (Inline, int, bool) {
	// “A processing instruction consists of the string <?,
	// a string of characters not including the string ?>, and the string ?>.”
	return parseHTMLMarker(s, i, "<?", "?>")
}

func parseHTMLMarker(s string, i int, prefix, suffix string) (Inline, int, bool) {
	if strings.HasPrefix(s[i:], prefix) {
		if j := strings.Index(s[i+len(prefix):], suffix); j >= 0 {
			end := i + len(prefix) + j + len(suffix)
			return &HTMLTag{s[i:end]}, end, true
		}
	}
	return nil, 0, false
}

func parseHTMLEntity(_ *parseState, s string, i int) (Inline, int, int, bool) {
	start := i
	if i+1 < len(s) && s[i+1] == '#' {
		i += 2
		var r, end int
		if i < len(s) && (s[i] == 'x' || s[i] == 'X') {
			// hex
			i++
			j := i
			for j < len(s) && isHexDigit(s[j]) {
				j++
			}
			if j-i < 1 || j-i > 6 || j >= len(s) || s[j] != ';' {
				return nil, 0, 0, false
			}
			r64, _ := strconv.ParseInt(s[i:j], 16, 0)
			r = int(r64)
			end = j + 1
		} else {
			// decimal
			j := i
			for j < len(s) && isDigit(s[j]) {
				j++
			}
			if j-i < 1 || j-i > 7 || j >= len(s) || s[j] != ';' {
				return nil, 0, 0, false
			}
			r, _ = strconv.Atoi(s[i:j])
			end = j + 1
		}
		if r > unicode.MaxRune || r == 0 {
			r = unicode.ReplacementChar
		}
		return &Plain{string(rune(r))}, start, end, true
	}

	// Max name in list is 32 bytes. Try for 64 for good measure.
	for j := i + 1; j < len(s) && j-i < 64; j++ {
		if s[j] == '&' { // Stop possible quadratic search on &&&&&&&.
			break
		}
		if s[j] == ';' {
			if r, ok := htmlEntity[s[i:j+1]]; ok {
				return &Plain{r}, start, j + 1, true
			}
			break
		}
	}

	return nil, 0, 0, false
}

type HTMLTag struct {
	Text string
}

func (*HTMLTag) Inline() {}

func (x *HTMLTag) PrintHTML(buf *bytes.Buffer) {
	buf.WriteString(x.Text)
}

func (x *HTMLTag) printMarkdown(buf *bytes.Buffer) {
	x.PrintHTML(buf)
}

func (x *HTMLTag) PrintText(buf *bytes.Buffer) {}
