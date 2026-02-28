// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package markdown

import (
	"bytes"
	"fmt"
	"strings"
)

type Heading struct {
	Position
	Level int
	Text  *Text
	// The HTML id attribute. The parser populates this field if
	// [Parser.HeadingIDs] is true and the heading ends with text like "{#id}".
	ID string
}

func (b *Heading) PrintHTML(buf *bytes.Buffer) {
	fmt.Fprintf(buf, "<h%d", b.Level)
	if b.ID != "" {
		fmt.Fprintf(buf, ` id="%s"`, htmlQuoteEscaper.Replace(b.ID))
	}
	buf.WriteByte('>')
	b.Text.PrintHTML(buf)
	fmt.Fprintf(buf, "</h%d>\n", b.Level)
}

func (b *Heading) printMarkdown(buf *bytes.Buffer, s mdState) {
	// TODO: handle setext headings properly.
	buf.WriteString(s.prefix)
	for i := 0; i < b.Level; i++ {
		buf.WriteByte('#')
	}
	buf.WriteByte(' ')
	// The prefix has already been printed for this line of text.
	s.prefix = ""
	b.Text.printMarkdown(buf, s)
	if b.ID != "" {
		// A heading text is a block, so it ends in a newline. Move the newline
		// after the ID.
		buf.Truncate(buf.Len() - 1)
		fmt.Fprintf(buf, " {#%s}\n", b.ID)
	}
}

func newATXHeading(p *parseState, s line) (line, bool) {
	peek := s
	var n int
	if peek.trimHeading(&n) {
		s := peek.string()
		s = trimRightSpaceTab(s)
		// Remove trailing '#'s.
		if t := strings.TrimRight(s, "#"); t != trimRightSpaceTab(t) || t == "" {
			s = t
		}
		var id string
		if p.HeadingIDs {
			// Parse and remove ID attribute.
			// It must come before trailing '#'s to more closely follow the spec:
			//    The optional closing sequence of #s must be preceded by spaces or tabs
			//    and may be followed by spaces or tabs only.
			// But Goldmark allows it to come after.
			id, s = extractID(p, s)

			// Goldmark is strict about the id syntax.
			for _, c := range id {
				if c >= 0x80 || !isLetterDigit(byte(c)) {
					p.corner = true
				}
			}
		}
		pos := Position{p.lineno, p.lineno}
		p.doneBlock(&Heading{pos, n, p.newText(pos, s), id})
		return line{}, true
	}
	return s, false
}

// extractID removes an ID attribute from s if one is present.
// It returns the attribute value and the resulting string.
// The attribute has the form "{#...}", where the "..." can contain
// any character other than '}'.
// The attribute must be followed only by whitespace.
func extractID(p *parseState, s string) (id, s2 string) {
	i := strings.LastIndexByte(s, '{')
	if i < 0 {
		return "", s
	}
	if i+1 >= len(s) || s[i+1] != '#' {
		p.corner = true // goldmark accepts {}
		return "", s
	}
	j := i + strings.IndexByte(s[i:], '}')
	if j < 0 || trimRightSpaceTab(s[j+1:]) != "" {
		return "", s
	}
	id = strings.TrimSpace(s[i+2 : j])
	if id == "" {
		p.corner = true // goldmark accepts {#}
		return "", s
	}
	return s[i+2 : j], s[:i]
}

func newSetextHeading(p *parseState, s line) (line, bool) {
	var n int
	peek := s
	if p.nextB() == p.para() && peek.trimSetext(&n) {
		p.closeBlock()
		para, ok := p.last().(*Paragraph)
		if !ok {
			return s, false
		}
		p.deleteLast()
		p.doneBlock(&Heading{Position{para.StartLine, p.lineno}, n, para.Text, ""})
		return line{}, true
	}
	return s, false
}

func (s *line) trimHeading(width *int) bool {
	t := *s
	t.trimSpace(0, 3, false)
	if !t.trim('#') {
		return false
	}
	n := 1
	for n < 6 && t.trim('#') {
		n++
	}
	if !t.trimSpace(1, 1, true) {
		return false
	}
	*width = n
	*s = t
	return true
}

func (s *line) trimSetext(n *int) bool {
	t := *s
	t.trimSpace(0, 3, false)
	c := t.peek()
	if c == '-' || c == '=' {
		for t.trim(c) {
		}
		t.skipSpace()
		if t.eof() {
			if c == '=' {
				*n = 1
			} else {
				*n = 2
			}
			return true
		}
	}
	return false
}
