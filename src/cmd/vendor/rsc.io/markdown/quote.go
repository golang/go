// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package markdown

import (
	"bytes"
)

type Quote struct {
	Position
	Blocks []Block
}

func (b *Quote) PrintHTML(buf *bytes.Buffer) {
	buf.WriteString("<blockquote>\n")
	for _, c := range b.Blocks {
		c.PrintHTML(buf)
	}
	buf.WriteString("</blockquote>\n")
}

func (b *Quote) printMarkdown(buf *bytes.Buffer, s mdState) {
	s.prefix += "> "
	printMarkdownBlocks(b.Blocks, buf, s)
}

func trimQuote(s line) (line, bool) {
	t := s
	t.trimSpace(0, 3, false)
	if !t.trim('>') {
		return s, false
	}
	t.trimSpace(0, 1, true)
	return t, true
}

type quoteBuilder struct{}

func newQuote(p *parseState, s line) (line, bool) {
	if line, ok := trimQuote(s); ok {
		p.addBlock(new(quoteBuilder))
		return line, true
	}
	return s, false
}

func (b *quoteBuilder) extend(p *parseState, s line) (line, bool) {
	return trimQuote(s)
}

func (b *quoteBuilder) build(p buildState) Block {
	return &Quote{p.pos(), p.blocks()}
}
