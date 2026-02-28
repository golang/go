// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package markdown

import (
	"bytes"
	"fmt"
	"strings"
)

type List struct {
	Position
	Bullet rune
	Start  int
	Loose  bool
	Items  []Block // always *Item
}

type Item struct {
	Position
	Blocks []Block
	width  int
}

func (b *List) PrintHTML(buf *bytes.Buffer) {
	if b.Bullet == '.' || b.Bullet == ')' {
		buf.WriteString("<ol")
		if b.Start != 1 {
			fmt.Fprintf(buf, " start=\"%d\"", b.Start)
		}
		buf.WriteString(">\n")
	} else {
		buf.WriteString("<ul>\n")
	}
	for _, c := range b.Items {
		c.PrintHTML(buf)
	}
	if b.Bullet == '.' || b.Bullet == ')' {
		buf.WriteString("</ol>\n")
	} else {
		buf.WriteString("</ul>\n")
	}
}

func (b *List) printMarkdown(buf *bytes.Buffer, s mdState) {
	if buf.Len() > 0 && buf.Bytes()[buf.Len()-1] != '\n' {
		buf.WriteByte('\n')
	}
	s.bullet = b.Bullet
	s.num = b.Start
	for i, item := range b.Items {
		if i > 0 && b.Loose {
			buf.WriteByte('\n')
		}
		item.printMarkdown(buf, s)
		s.num++
	}
}

func (b *Item) printMarkdown(buf *bytes.Buffer, s mdState) {
	var marker string
	if s.bullet == '.' || s.bullet == ')' {
		marker = fmt.Sprintf("%d%c ", s.num, s.bullet)
	} else {
		marker = fmt.Sprintf("%c ", s.bullet)
	}
	marker = strings.Repeat(" ", b.width-len(marker)) + marker
	s.prefix1 = s.prefix + marker
	s.prefix += strings.Repeat(" ", len(marker))
	printMarkdownBlocks(b.Blocks, buf, s)
}

func (b *Item) PrintHTML(buf *bytes.Buffer) {
	buf.WriteString("<li>")
	if len(b.Blocks) > 0 {
		if _, ok := b.Blocks[0].(*Text); !ok {
			buf.WriteString("\n")
		}
	}
	for i, c := range b.Blocks {
		c.PrintHTML(buf)
		if i+1 < len(b.Blocks) {
			if _, ok := c.(*Text); ok {
				buf.WriteString("\n")
			}
		}
	}
	buf.WriteString("</li>\n")
}

type listBuilder struct {
	bullet rune
	num    int
	loose  bool
	item   *itemBuilder
	todo   func() line
}

func (b *listBuilder) build(p buildState) Block {
	blocks := p.blocks()
	pos := p.pos()

	// list can have wrong pos b/c extend dance.
	pos.EndLine = blocks[len(blocks)-1].Pos().EndLine
Loose:
	for i, c := range blocks {
		c := c.(*Item)
		if i+1 < len(blocks) {
			if blocks[i+1].Pos().StartLine-c.EndLine > 1 {
				b.loose = true
				break Loose
			}
		}
		for j, d := range c.Blocks {
			endLine := d.Pos().EndLine
			if j+1 < len(c.Blocks) {
				if c.Blocks[j+1].Pos().StartLine-endLine > 1 {
					b.loose = true
					break Loose
				}
			}
		}
	}

	if !b.loose {
		for _, c := range blocks {
			c := c.(*Item)
			for i, d := range c.Blocks {
				if p, ok := d.(*Paragraph); ok {
					c.Blocks[i] = p.Text
				}
			}
		}
	}

	return &List{
		pos,
		b.bullet,
		b.num,
		b.loose,
		p.blocks(),
	}
}

func (b *itemBuilder) build(p buildState) Block {
	b.list.item = nil
	return &Item{p.pos(), p.blocks(), b.width}
}

func (c *listBuilder) extend(p *parseState, s line) (line, bool) {
	d := c.item
	if d != nil && s.trimSpace(d.width, d.width, true) || d == nil && s.isBlank() {
		return s, true
	}
	return s, false
}

func (c *itemBuilder) extend(p *parseState, s line) (line, bool) {
	if s.isBlank() && !c.haveContent {
		return s, false
	}
	if s.isBlank() {
		// Goldmark does this and apparently commonmark.js too.
		// Not sure why it is necessary.
		return line{}, true
	}
	if !s.isBlank() {
		c.haveContent = true
	}
	return s, true
}

func newListItem(p *parseState, s line) (line, bool) {
	if list, ok := p.curB().(*listBuilder); ok && list.todo != nil {
		s = list.todo()
		list.todo = nil
		return s, true
	}
	if p.startListItem(&s) {
		return s, true
	}
	return s, false
}

func (p *parseState) startListItem(s *line) bool {
	t := *s
	n := 0
	for i := 0; i < 3; i++ {
		if !t.trimSpace(1, 1, false) {
			break
		}
		n++
	}
	bullet := t.peek()
	var num int
Switch:
	switch bullet {
	default:
		return false
	case '-', '*', '+':
		t.trim(bullet)
		n++
	case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		for j := t.i; ; j++ {
			if j >= len(t.text) {
				return false
			}
			c := t.text[j]
			if c == '.' || c == ')' {
				// success
				bullet = c
				j++
				n += j - t.i
				t.i = j
				break Switch
			}
			if c < '0' || '9' < c {
				return false
			}
			if j-t.i >= 9 {
				return false
			}
			num = num*10 + int(c) - '0'
		}

	}
	if !t.trimSpace(1, 1, true) {
		return false
	}
	n++
	tt := t
	m := 0
	for i := 0; i < 3 && tt.trimSpace(1, 1, false); i++ {
		m++
	}
	if !tt.trimSpace(1, 1, true) {
		n += m
		t = tt
	}

	// point of no return

	var list *listBuilder
	if c, ok := p.nextB().(*listBuilder); ok {
		list = c
	}
	if list == nil || list.bullet != rune(bullet) {
		// “When the first list item in a list interrupts a paragraph—that is,
		// when it starts on a line that would otherwise count as
		// paragraph continuation text—then (a) the lines Ls must
		// not begin with a blank line,
		// and (b) if the list item is ordered, the start number must be 1.”
		if list == nil && p.para() != nil && (t.isBlank() || (bullet == '.' || bullet == ')') && num != 1) {
			// Goldmark and Dingus both seem to get this wrong
			// (or the words above don't mean what we think they do).
			// when the paragraph that could be continued
			// is inside a block quote.
			// See testdata/extra.txt 117.md.
			p.corner = true
			return false
		}
		list = &listBuilder{bullet: rune(bullet), num: num}
		p.addBlock(list)
	}
	b := &itemBuilder{list: list, width: n, haveContent: !t.isBlank()}
	list.todo = func() line {
		p.addBlock(b)
		list.item = b
		return t
	}
	return true
}

// GitHub task list extension

func (p *parseState) taskList(list *List) {
	for _, item := range list.Items {
		item := item.(*Item)
		if len(item.Blocks) == 0 {
			continue
		}
		var text *Text
		switch b := item.Blocks[0].(type) {
		default:
			continue
		case *Paragraph:
			text = b.Text
		case *Text:
			text = b
		}
		if len(text.Inline) < 1 {
			continue
		}
		pl, ok := text.Inline[0].(*Plain)
		if !ok {
			continue
		}
		s := pl.Text
		if len(s) < 4 || s[0] != '[' || s[2] != ']' || (s[1] != ' ' && s[1] != 'x' && s[1] != 'X') {
			continue
		}
		if s[3] != ' ' && s[3] != '\t' {
			p.corner = true // goldmark does not require the space
			continue
		}
		text.Inline = append([]Inline{&Task{Checked: s[1] == 'x' || s[1] == 'X'},
			&Plain{Text: s[len("[x]"):]}}, text.Inline[1:]...)
	}
}

func ins(first Inline, x []Inline) []Inline {
	x = append(x, nil)
	copy(x[1:], x)
	x[0] = first
	return x
}

type Task struct {
	Checked bool
}

func (x *Task) Inline() {
}

func (x *Task) PrintHTML(buf *bytes.Buffer) {
	buf.WriteString("<input ")
	if x.Checked {
		buf.WriteString(`checked="" `)
	}
	buf.WriteString(`disabled="" type="checkbox">`)
}

func (x *Task) printMarkdown(buf *bytes.Buffer) {
	x.PrintText(buf)
}

func (x *Task) PrintText(buf *bytes.Buffer) {
	buf.WriteByte('[')
	if x.Checked {
		buf.WriteByte('x')
	} else {
		buf.WriteByte(' ')
	}
	buf.WriteByte(']')
	buf.WriteByte(' ')
}

func listCorner(list *List) bool {
	for _, item := range list.Items {
		item := item.(*Item)
		if len(item.Blocks) == 0 {
			// Goldmark mishandles what follows; see testdata/extra.txt 111.md.
			return true
		}
		switch item.Blocks[0].(type) {
		case *List, *ThematicBreak, *CodeBlock:
			// Goldmark mishandles a list with various block items inside it.
			return true
		}
	}
	return false
}
