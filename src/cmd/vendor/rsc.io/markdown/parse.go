// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package markdown

import (
	"bytes"
	"fmt"
	"reflect"
	"slices"
	"strings"
)

/*

list block itself does not appear on stack?
item does
end of item returns block,
new item continues previous block if possible?

if close leaves lines or blocks behind, panic

close(b a list item, parent)
	if b's parent's last block is list && item can be added to it, do so
	else return new list

or maybe not parent but just current list of blocks

preserve LinkRefDefs?

*/

// Block is implemented by:
//
//	CodeBLock
//	Document
//	Empty
//	HTMLBlock
//	Heading
//	Item
//	List
//	Paragraph
//	Quote
//	Text
//	ThematicBreak
type Block interface {
	Pos() Position
	PrintHTML(buf *bytes.Buffer)
	printMarkdown(buf *bytes.Buffer, s mdState)
}

type mdState struct {
	prefix  string
	prefix1 string // for first line only
	bullet  rune   // for list items
	num     int    // for numbered list items
}

type Position struct {
	StartLine int
	EndLine   int
}

func (p Position) Pos() Position {
	return p
}

type buildState interface {
	blocks() []Block
	pos() Position
	last() Block
	deleteLast()

	link(label string) *Link
	defineLink(label string, link *Link)
	newText(pos Position, text string) *Text
}

type blockBuilder interface {
	extend(p *parseState, s line) (line, bool)
	build(buildState) Block
}

type openBlock struct {
	builder blockBuilder
	inner   []Block
	pos     Position
}

type itemBuilder struct {
	list        *listBuilder
	width       int
	haveContent bool
}

func (p *parseState) last() Block {
	ob := &p.stack[len(p.stack)-1]
	return ob.inner[len(ob.inner)-1]
}

func (p *parseState) deleteLast() {
	ob := &p.stack[len(p.stack)-1]
	ob.inner = ob.inner[:len(ob.inner)-1]
}

type Text struct {
	Position
	Inline []Inline
	raw    string
}

func (b *Text) PrintHTML(buf *bytes.Buffer) {
	for _, x := range b.Inline {
		x.PrintHTML(buf)
	}
}

func (b *Text) printMarkdown(buf *bytes.Buffer, s mdState) {
	if s.prefix1 != "" {
		buf.WriteString(s.prefix1)
	} else {
		buf.WriteString(s.prefix)
	}
	var prev Inline
	for _, x := range b.Inline {
		switch prev.(type) {
		case *SoftBreak, *HardBreak:
			buf.WriteString(s.prefix)
		}
		x.printMarkdown(buf)
		prev = x
	}
	buf.WriteByte('\n')
}

type rootBuilder struct{}

func (b *rootBuilder) build(p buildState) Block {
	return &Document{p.pos(), p.blocks(), p.(*parseState).links}
}

type Document struct {
	Position
	Blocks []Block
	Links  map[string]*Link
}

// A Parser is a Markdown parser.
// The exported fields in the struct can be filled in before calling
// [Parser.Parse] in order to customize the details of the parsing process.
// A Parser is safe for concurrent use by multiple goroutines.
type Parser struct {
	// HeadingIDs determines whether the parser accepts
	// the {#hdr} syntax for an HTML id="hdr" attribute on headings.
	// For example, if HeadingIDs is true then the Markdown
	//    ## Overview {#overview}
	// will render as the HTML
	//    <h2 id="overview">Overview</h2>
	HeadingIDs bool

	// Strikethrough determines whether the parser accepts
	// ~abc~ and ~~abc~~ as strikethrough syntax, producing
	// <del>abc</del> in HTML.
	Strikethrough bool

	// TaskListItems determines whether the parser accepts
	// “task list items” as defined in GitHub Flavored Markdown.
	// When a list item begins with the plain text [ ] or [x]
	// that turns into an unchecked or checked check box.
	TaskListItems bool

	// TODO
	AutoLinkText       bool
	AutoLinkAssumeHTTP bool

	// TODO
	Table bool

	// TODO
	Emoji bool

	// TODO
	SmartDot   bool
	SmartDash  bool
	SmartQuote bool
}

type parseState struct {
	*Parser

	root      *Document
	links     map[string]*Link
	lineno    int
	stack     []openBlock
	lineDepth int

	corner bool // noticed corner case to ignore in cross-implementation testing

	// inlines
	s       string
	emitted int // s[:emitted] has been emitted into list
	list    []Inline

	// for fixup at end
	lists []*List
	texts []*Text

	backticks backtickParser
}

func (p *parseState) newText(pos Position, text string) *Text {
	b := &Text{Position: pos, raw: text}
	p.texts = append(p.texts, b)
	return b
}

func (p *parseState) blocks() []Block {
	b := &p.stack[len(p.stack)-1]
	return b.inner
}

func (p *parseState) pos() Position {
	b := &p.stack[len(p.stack)-1]
	return b.pos
}

func (p *Parser) Parse(text string) *Document {
	d, _ := p.parse(text)
	return d
}

func (p *Parser) parse(text string) (d *Document, corner bool) {
	var ps parseState
	ps.Parser = p
	if strings.Contains(text, "\x00") {
		text = strings.ReplaceAll(text, "\x00", "\uFFFD")
		ps.corner = true // goldmark does not replace NUL
	}

	ps.lineDepth = -1
	ps.addBlock(&rootBuilder{})
	for text != "" {
		var ln string
		i := strings.Index(text, "\n")
		j := strings.Index(text, "\r")
		var nl byte
		switch {
		case j >= 0 && (i < 0 || j < i): // have \r, maybe \r\n
			ln = text[:j]
			if i == j+1 {
				text = text[j+2:]
				nl = '\r' + '\n'
			} else {
				text = text[j+1:]
				nl = '\r'
			}
		case i >= 0:
			ln, text = text[:i], text[i+1:]
			nl = '\n'
		default:
			ln, text = text, ""
		}
		ps.lineno++
		ps.addLine(line{text: ln, nl: nl})
	}
	ps.trimStack(0)

	for _, t := range ps.texts {
		t.Inline = ps.inline(t.raw)
	}

	if p.TaskListItems {
		for _, list := range ps.lists {
			ps.taskList(list)
		}
	}

	return ps.root, ps.corner
}

func (p *parseState) curB() blockBuilder {
	if p.lineDepth < len(p.stack) {
		return p.stack[p.lineDepth].builder
	}
	return nil
}

func (p *parseState) nextB() blockBuilder {
	if p.lineDepth+1 < len(p.stack) {
		return p.stack[p.lineDepth+1].builder
	}
	return nil
}
func (p *parseState) trimStack(depth int) {
	if len(p.stack) < depth {
		panic("trimStack")
	}
	for len(p.stack) > depth {
		p.closeBlock()
	}
}

func (p *parseState) addBlock(c blockBuilder) {
	p.trimStack(p.lineDepth + 1)
	p.stack = append(p.stack, openBlock{})
	ob := &p.stack[len(p.stack)-1]
	ob.builder = c
	ob.pos.StartLine = p.lineno
	ob.pos.EndLine = p.lineno
}

func (p *parseState) doneBlock(b Block) {
	p.trimStack(p.lineDepth + 1)
	ob := &p.stack[len(p.stack)-1]
	ob.inner = append(ob.inner, b)
}

func (p *parseState) para() *paraBuilder {
	if b, ok := p.stack[len(p.stack)-1].builder.(*paraBuilder); ok {
		return b
	}
	return nil
}

func (p *parseState) closeBlock() Block {
	b := &p.stack[len(p.stack)-1]
	if b.builder == nil {
		println("closeBlock", len(p.stack)-1)
	}
	blk := b.builder.build(p)
	if list, ok := blk.(*List); ok {
		p.corner = p.corner || listCorner(list)
		if p.TaskListItems {
			p.lists = append(p.lists, list)
		}
	}
	p.stack = p.stack[:len(p.stack)-1]
	if len(p.stack) > 0 {
		b := &p.stack[len(p.stack)-1]
		b.inner = append(b.inner, blk)
		// _ = b
	} else {
		p.root = blk.(*Document)
	}
	return blk
}

func (p *parseState) link(label string) *Link {
	return p.links[label]
}

func (p *parseState) defineLink(label string, link *Link) {
	if p.links == nil {
		p.links = make(map[string]*Link)
	}
	p.links[label] = link
}

type line struct {
	spaces int
	i      int
	tab    int
	text   string
	nl     byte // newline character ending this line: \r or \n or zero for EOF
}

func (p *parseState) addLine(s line) {
	// Process continued prefixes.
	p.lineDepth = 0
	for ; p.lineDepth+1 < len(p.stack); p.lineDepth++ {
		old := s
		var ok bool
		s, ok = p.stack[p.lineDepth+1].builder.extend(p, s)
		if !old.isBlank() && (ok || s != old) {
			p.stack[p.lineDepth+1].pos.EndLine = p.lineno
		}
		if !ok {
			break
		}
	}

	if s.isBlank() {
		p.trimStack(p.lineDepth + 1)
		return
	}

	// Process new prefixes, if any.
Prefixes:
	// Start new block inside p.stack[depth].
	for _, fn := range news {
		if l, ok := fn(p, s); ok {
			s = l
			if s.isBlank() {
				return
			}
			p.lineDepth++
			goto Prefixes
		}
	}

	newPara(p, s)
}

func (c *rootBuilder) extend(p *parseState, s line) (line, bool) {
	panic("root extend")
}

var news = []func(*parseState, line) (line, bool){
	newQuote,
	newATXHeading,
	newSetextHeading,
	newHR,
	newListItem,
	newHTML,
	newFence,
	newPre,
}

func (s *line) peek() byte {
	if s.spaces > 0 {
		return ' '
	}
	if s.i >= len(s.text) {
		return 0
	}
	return s.text[s.i]
}

func (s *line) skipSpace() {
	s.spaces = 0
	for s.i < len(s.text) && (s.text[s.i] == ' ' || s.text[s.i] == '\t') {
		s.i++
	}
}

func (s *line) trimSpace(min, max int, eolOK bool) bool {
	t := *s
	for n := 0; n < max; n++ {
		if t.spaces > 0 {
			t.spaces--
			continue
		}
		if t.i >= len(t.text) && eolOK {
			continue
		}
		if t.i < len(t.text) {
			switch t.text[t.i] {
			case '\t':
				t.spaces = 4 - (t.i-t.tab)&3 - 1
				t.i++
				t.tab = t.i
				continue
			case ' ':
				t.i++
				continue
			}
		}
		if n >= min {
			break
		}
		return false
	}
	*s = t
	return true
}

func (s *line) trim(c byte) bool {
	if s.spaces > 0 {
		if c == ' ' {
			s.spaces--
			return true
		}
		return false
	}
	if s.i < len(s.text) && s.text[s.i] == c {
		s.i++
		return true
	}
	return false
}

func (s *line) string() string {
	switch s.spaces {
	case 0:
		return s.text[s.i:]
	case 1:
		return " " + s.text[s.i:]
	case 2:
		return "  " + s.text[s.i:]
	case 3:
		return "   " + s.text[s.i:]
	}
	panic("bad spaces")
}

func trimLeftSpaceTab(s string) string {
	i := 0
	for i < len(s) && (s[i] == ' ' || s[i] == '\t') {
		i++
	}
	return s[i:]
}

func trimRightSpaceTab(s string) string {
	j := len(s)
	for j > 0 && (s[j-1] == ' ' || s[j-1] == '\t') {
		j--
	}
	return s[:j]
}

func trimSpaceTab(s string) string {
	i := 0
	for i < len(s) && (s[i] == ' ' || s[i] == '\t') {
		i++
	}
	s = s[i:]
	j := len(s)
	for j > 0 && (s[j-1] == ' ' || s[j-1] == '\t') {
		j--
	}
	return s[:j]
}

func trimSpace(s string) string {
	i := 0
	for i < len(s) && (s[i] == ' ' || s[i] == '\t') {
		i++
	}
	s = s[i:]
	j := len(s)
	for j > 0 && (s[j-1] == ' ' || s[j-1] == '\t') {
		j--
	}
	return s[:j]
}

func trimSpaceTabNewline(s string) string {
	i := 0
	for i < len(s) && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n') {
		i++
	}
	s = s[i:]
	j := len(s)
	for j > 0 && (s[j-1] == ' ' || s[j-1] == '\t' || s[j-1] == '\n') {
		j--
	}
	return s[:j]
}

func (s *line) isBlank() bool {
	return trimLeftSpaceTab(s.text[s.i:]) == ""
}

func (s *line) eof() bool {
	return s.i >= len(s.text)
}

func (s *line) trimSpaceString() string {
	return trimLeftSpaceTab(s.text[s.i:])
}

func (s *line) trimString() string {
	return trimSpaceTab(s.text[s.i:])
}

func ToHTML(b Block) string {
	var buf bytes.Buffer
	b.PrintHTML(&buf)
	return buf.String()
}

func ToMarkdown(b Block) string {
	var buf bytes.Buffer
	b.printMarkdown(&buf, mdState{})
	s := buf.String()
	// Remove final extra newline.
	if strings.HasSuffix(s, "\n\n") {
		s = s[:len(s)-1]
	}
	return s
}

func (b *Document) PrintHTML(buf *bytes.Buffer) {
	for _, c := range b.Blocks {
		c.PrintHTML(buf)
	}
}

func (b *Document) printMarkdown(buf *bytes.Buffer, s mdState) {
	printMarkdownBlocks(b.Blocks, buf, s)
	// Print links sorted by keys for deterministic output.
	var keys []string
	for k := range b.Links {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	for _, k := range keys {
		l := b.Links[k]
		fmt.Fprintf(buf, "[%s]: %s", k, l.URL)
		printLinkTitleMarkdown(buf, l.Title, l.TitleChar)
		buf.WriteByte('\n')
	}
}

func printMarkdownBlocks(bs []Block, buf *bytes.Buffer, s mdState) {
	prevEnd := 0
	for _, b := range bs {
		// Preserve blank lines between blocks.
		if prevEnd > 0 {
			for i := prevEnd + 1; i < b.Pos().StartLine; i++ {
				buf.WriteString(trimRightSpaceTab(s.prefix))
				buf.WriteByte('\n')
			}
		}
		b.printMarkdown(buf, s)
		prevEnd = b.Pos().EndLine
		s.prefix1 = "" // item prefix only for first block
	}
}

var (
	blockType   = reflect.TypeOf(new(Block)).Elem()
	blocksType  = reflect.TypeOf(new([]Block)).Elem()
	inlinesType = reflect.TypeOf(new([]Inline)).Elem()
)

func printb(buf *bytes.Buffer, b Block, prefix string) {
	fmt.Fprintf(buf, "(%T", b)
	v := reflect.ValueOf(b)
	v = reflect.Indirect(v)
	if v.Kind() != reflect.Struct {
		fmt.Fprintf(buf, " %v", b)
	}
	t := v.Type()
	for i := 0; i < t.NumField(); i++ {
		tf := t.Field(i)
		if !tf.IsExported() {
			continue
		}
		if tf.Type == inlinesType {
			printis(buf, v.Field(i).Interface().([]Inline))
		} else if tf.Type.Kind() == reflect.Slice && tf.Type.Elem().Kind() == reflect.String {
			fmt.Fprintf(buf, " %s:%q", tf.Name, v.Field(i))
		} else if tf.Type != blocksType && !tf.Type.Implements(blockType) && tf.Type.Kind() != reflect.Slice {
			fmt.Fprintf(buf, " %s:%v", tf.Name, v.Field(i))
		}
	}

	prefix += "\t"
	for i := 0; i < t.NumField(); i++ {
		tf := t.Field(i)
		if !tf.IsExported() {
			continue
		}
		if tf.Type.Implements(blockType) {
			fmt.Fprintf(buf, "\n%s", prefix)
			printb(buf, v.Field(i).Interface().(Block), prefix)
		} else if tf.Type == blocksType {
			vf := v.Field(i)
			for i := 0; i < vf.Len(); i++ {
				fmt.Fprintf(buf, "\n%s", prefix)
				printb(buf, vf.Index(i).Interface().(Block), prefix)
			}
		} else if tf.Type.Kind() == reflect.Slice && tf.Type != inlinesType && tf.Type.Elem().Kind() != reflect.String {
			fmt.Fprintf(buf, "\n%s%s:", prefix, t.Field(i).Name)
			printslice(buf, v.Field(i), prefix)
		}
	}
	fmt.Fprintf(buf, ")")
}

func printslice(buf *bytes.Buffer, v reflect.Value, prefix string) {
	if v.Type().Elem().Kind() == reflect.Slice {
		for i := 0; i < v.Len(); i++ {
			fmt.Fprintf(buf, "\n%s#%d:", prefix, i)
			printslice(buf, v.Index(i), prefix+"\t")
		}
		return
	}
	for i := 0; i < v.Len(); i++ {
		fmt.Fprintf(buf, " ")
		printb(buf, v.Index(i).Interface().(Block), prefix+"\t")
	}
}

func printi(buf *bytes.Buffer, in Inline) {
	fmt.Fprintf(buf, "%T(", in)
	v := reflect.ValueOf(in).Elem()
	text := v.FieldByName("Text")
	if text.IsValid() {
		fmt.Fprintf(buf, "%q", text)
	}
	inner := v.FieldByName("Inner")
	if inner.IsValid() {
		printis(buf, inner.Interface().([]Inline))
	}
	buf.WriteString(")")
}

func printis(buf *bytes.Buffer, ins []Inline) {
	for _, in := range ins {
		buf.WriteByte(' ')
		printi(buf, in)
	}
}

func dump(b Block) string {
	var buf bytes.Buffer
	printb(&buf, b, "")
	return buf.String()
}
