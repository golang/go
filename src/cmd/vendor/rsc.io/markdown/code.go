// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package markdown

import (
	"bytes"
	"fmt"
	"strings"
)

type CodeBlock struct {
	Position
	Fence string
	Info  string
	Text  []string
}

func (b *CodeBlock) PrintHTML(buf *bytes.Buffer) {
	if buf.Len() > 0 && buf.Bytes()[buf.Len()-1] != '\n' {
		buf.WriteString("\n")
	}
	buf.WriteString("<pre><code")
	if b.Info != "" {
		// https://spec.commonmark.org/0.30/#info-string
		// “The first word of the info string is typically used to
		// specify the language of the code sample...”
		// No definition of what “first word” means though.
		// The Dingus splits on isUnicodeSpace, but Goldmark only uses space.
		lang := b.Info
		for i, c := range lang {
			if isUnicodeSpace(c) {
				lang = lang[:i]
				break
			}
		}
		fmt.Fprintf(buf, " class=\"language-%s\"", htmlQuoteEscaper.Replace(lang))
	}
	buf.WriteString(">")
	if b.Fence == "" { // TODO move
		for len(b.Text) > 0 && trimSpaceTab(b.Text[len(b.Text)-1]) == "" {
			b.Text = b.Text[:len(b.Text)-1]
		}
	}
	for _, s := range b.Text {
		buf.WriteString(htmlEscaper.Replace(s))
		buf.WriteString("\n")
	}
	buf.WriteString("</code></pre>\n")
}

// func initialSpaces(s string) int {
// 	for i := 0; i < len(s); i++ {
// 		if s[i] != ' ' {
// 			return i
// 		}
// 	}
// 	return len(s)
// }

func (b *CodeBlock) printMarkdown(buf *bytes.Buffer, s mdState) {
	prefix1 := s.prefix1
	if prefix1 == "" {
		prefix1 = s.prefix
	}
	if b.Fence == "" {
		for i, line := range b.Text {
			// Ignore final empty line (why is it even there?).
			if i == len(b.Text)-1 && len(line) == 0 {
				break
			}
			// var iline string
			// is := initialSpaces(line)
			// if is < 4 {
			// 	iline = "    " + line
			// } else {
			// 	iline = "\t" + line[4:]
			// }
			// Indent by 4 spaces.
			pre := s.prefix
			if i == 0 {
				pre = prefix1
			}
			fmt.Fprintf(buf, "%s%s%s\n", pre, "    ", line)
		}
	} else {
		fmt.Fprintf(buf, "%s%s\n", prefix1, b.Fence)
		for _, line := range b.Text {
			fmt.Fprintf(buf, "%s%s\n", s.prefix, line)
		}
		fmt.Fprintf(buf, "%s%s\n", s.prefix, b.Fence)
	}
}

func newPre(p *parseState, s line) (line, bool) {
	peek2 := s
	if p.para() == nil && peek2.trimSpace(4, 4, false) && !peek2.isBlank() {
		b := &preBuilder{ /*indent: strings.TrimSuffix(s.string(), peek2.string())*/ }
		p.addBlock(b)
		p.corner = p.corner || peek2.nl != '\n' // goldmark does not normalize to \n
		b.text = append(b.text, peek2.string())
		return line{}, true
	}
	return s, false
}

func newFence(p *parseState, s line) (line, bool) {
	var fence, info string
	var n int
	peek := s
	if peek.trimFence(&fence, &info, &n) {
		if fence[0] == '~' && info != "" {
			// goldmark does not handle info after ~~~
			p.corner = true
		} else if info != "" && !isLetter(info[0]) {
			// goldmark does not allow numbered info.
			// goldmark does not treat a tab as introducing a new word.
			p.corner = true
		}
		for _, c := range info {
			if isUnicodeSpace(c) {
				if c != ' ' {
					// goldmark only breaks on space
					p.corner = true
				}
				break
			}
		}

		p.addBlock(&fenceBuilder{fence, info, n, nil})
		return line{}, true
	}
	return s, false
}

func (s *line) trimFence(fence, info *string, n *int) bool {
	t := *s
	*n = 0
	for *n < 3 && t.trimSpace(1, 1, false) {
		*n++
	}
	switch c := t.peek(); c {
	case '`', '~':
		f := t.string()
		n := 0
		for i := 0; ; i++ {
			if !t.trim(c) {
				if i >= 3 {
					break
				}
				return false
			}
			n++
		}
		txt := mdUnescaper.Replace(t.trimString())
		if c == '`' && strings.Contains(txt, "`") {
			return false
		}
		txt = trimSpaceTab(txt)
		*info = txt

		*fence = f[:n]
		*s = line{}
		return true
	}
	return false
}

// For indented code blocks.
type preBuilder struct {
	indent string
	text   []string
}

func (c *preBuilder) extend(p *parseState, s line) (line, bool) {
	if !s.trimSpace(4, 4, true) {
		return s, false
	}
	c.text = append(c.text, s.string())
	p.corner = p.corner || s.nl != '\n' // goldmark does not normalize to \n
	return line{}, true
}

func (b *preBuilder) build(p buildState) Block {
	return &CodeBlock{p.pos(), "", "", b.text}
}

type fenceBuilder struct {
	fence string
	info  string
	n     int
	text  []string
}

func (c *fenceBuilder) extend(p *parseState, s line) (line, bool) {
	var fence, info string
	var n int
	if t := s; t.trimFence(&fence, &info, &n) && strings.HasPrefix(fence, c.fence) && info == "" {
		return line{}, false
	}
	if !s.trimSpace(c.n, c.n, false) {
		p.corner = true // goldmark mishandles fenced blank lines with not enough spaces
		s.trimSpace(0, c.n, false)
	}
	c.text = append(c.text, s.string())
	p.corner = p.corner || s.nl != '\n' // goldmark does not normalize to \n
	return line{}, true
}

func (c *fenceBuilder) build(p buildState) Block {
	return &CodeBlock{
		p.pos(),
		c.fence,
		c.info,
		c.text,
	}
}
