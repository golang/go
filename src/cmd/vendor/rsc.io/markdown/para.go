// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package markdown

import (
	"bytes"
	"strings"
)

type Empty struct {
	Position
}

func (b *Empty) PrintHTML(buf *bytes.Buffer) {}

func (b *Empty) printMarkdown(*bytes.Buffer, mdState) {}

type Paragraph struct {
	Position
	Text *Text
}

func (b *Paragraph) PrintHTML(buf *bytes.Buffer) {
	buf.WriteString("<p>")
	b.Text.PrintHTML(buf)
	buf.WriteString("</p>\n")
}

func (b *Paragraph) printMarkdown(buf *bytes.Buffer, s mdState) {
	// // Ignore prefix when in a list.
	// if s.bullet == 0 {
	// 	buf.WriteString(s.prefix)
	// }
	b.Text.printMarkdown(buf, s)
}

type paraBuilder struct {
	text  []string
	table *tableBuilder
}

func (b *paraBuilder) extend(p *parseState, s line) (line, bool) {
	return s, false
}

func (b *paraBuilder) build(p buildState) Block {
	if b.table != nil {
		return b.table.build(p)
	}

	s := strings.Join(b.text, "\n")
	for s != "" {
		end, ok := parseLinkRefDef(p, s)
		if !ok {
			break
		}
		s = s[skipSpace(s, end):]
	}

	if s == "" {
		return &Empty{p.pos()}
	}

	// Recompute EndLine because a line of b.text
	// might have been taken away to start a table.
	pos := p.pos()
	pos.EndLine = pos.StartLine + len(b.text) - 1
	return &Paragraph{
		pos,
		p.newText(pos, s),
	}
}

func newPara(p *parseState, s line) (line, bool) {
	// Process paragraph continuation text or start new paragraph.
	b := p.para()
	indented := p.lineDepth == len(p.stack)-2 // fully indented, not playing "pargraph continuation text" games
	text := s.trimSpaceString()

	if b != nil && b.table != nil {
		if indented && text != "" && text != "|" {
			// Continue table.
			b.table.addRow(text)
			return line{}, true
		}
		// Blank or unindented line ends table.
		// (So does a new block structure, but the caller has checked that already.)
		// So does a line with just a pipe:
		// https://github.com/github/cmark-gfm/pull/127 and
		// https://github.com/github/cmark-gfm/pull/128
		// fixed a buffer overread by rejecting | by itself as a table line.
		// That seems to violate the spec, but we will play along.
		b = nil
	}

	// If we are looking for tables and this is a table start, start a table.
	if p.Table && b != nil && indented && len(b.text) > 0 && isTableStart(b.text[len(b.text)-1], text) {
		hdr := b.text[len(b.text)-1]
		b.text = b.text[:len(b.text)-1]
		tb := new(paraBuilder)
		p.addBlock(tb)
		tb.table = new(tableBuilder)
		tb.table.start(hdr, text)
		return line{}, true
	}

	if b != nil {
		for i := p.lineDepth; i < len(p.stack); i++ {
			p.stack[i].pos.EndLine = p.lineno
		}
	} else {
		// Note: Ends anything without a matching prefix.
		b = new(paraBuilder)
		p.addBlock(b)
	}
	b.text = append(b.text, text)
	return line{}, true
}
