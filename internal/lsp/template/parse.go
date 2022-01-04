// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package template contains code for dealing with templates
package template

// template files are small enough that the code reprocesses them each time
// this may be a bad choice for projects with lots of template files.

// This file contains the parsing code, some debugging printing, and
// implementations for Diagnose, Definition, HJover, References

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"regexp"
	"runtime"
	"sort"
	"text/template"
	"text/template/parse"
	"unicode/utf8"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

var (
	Left  = []byte("{{")
	Right = []byte("}}")
)

type Parsed struct {
	buf    []byte   //contents
	lines  [][]byte // needed?, other than for debugging?
	elided []int    // offsets where Left was replaced by blanks

	// tokens are matched Left-Right pairs, computed before trying to parse
	tokens []Token

	// result of parsing
	named    []*template.Template // the template and embedded templates
	ParseErr error
	symbols  []symbol
	stack    []parse.Node // used while computing symbols

	// for mapping from offsets in buf to LSP coordinates
	// See FromPosition() and LineCol()
	nls      []int // offset of newlines before each line (nls[0]==-1)
	lastnl   int   // last line seen
	check    int   // used to decide whether to use lastnl or search through nls
	nonASCII bool  // are there any non-ascii runes in buf?
}

// Token is a single {{...}}. More precisely, Left...Right
type Token struct {
	Start, End int // offset from start of template
	Multiline  bool
}

// All contains the Parse of all the template files
type All struct {
	files map[span.URI]*Parsed
}

// New returns the Parses of the snapshot's tmpl files
// (maybe cache these, but then avoiding import cycles needs code rearrangements)
func New(tmpls map[span.URI]source.VersionedFileHandle) *All {
	all := make(map[span.URI]*Parsed)
	for k, v := range tmpls {
		buf, err := v.Read()
		if err != nil { // PJW: decide what to do with these errors
			log.Printf("failed to read %s (%v)", v.URI().Filename(), err)
			continue
		}
		all[k] = parseBuffer(buf)
	}
	return &All{files: all}
}

func parseBuffer(buf []byte) *Parsed {
	ans := &Parsed{
		buf:   buf,
		check: -1,
		nls:   []int{-1},
	}
	if len(buf) == 0 {
		return ans
	}
	// how to compute allAscii...
	for _, b := range buf {
		if b >= utf8.RuneSelf {
			ans.nonASCII = true
			break
		}
	}
	if buf[len(buf)-1] != '\n' {
		ans.buf = append(buf, '\n')
	}
	for i, p := range ans.buf {
		if p == '\n' {
			ans.nls = append(ans.nls, i)
		}
	}
	ans.setTokens() // ans.buf may be a new []byte
	ans.lines = bytes.Split(ans.buf, []byte{'\n'})
	t, err := template.New("").Parse(string(ans.buf))
	if err != nil {
		funcs := make(template.FuncMap)
		for t == nil && ans.ParseErr == nil {
			// in 1.17 it may be possible to avoid getting this error
			//  template: :2: function "foo" not defined
			matches := parseErrR.FindStringSubmatch(err.Error())
			if len(matches) == 2 {
				// suppress the error by giving it a function with the right name
				funcs[matches[1]] = func() interface{} { return nil }
				t, err = template.New("").Funcs(funcs).Parse(string(ans.buf))
				continue
			}
			ans.ParseErr = err // unfixed error
			return ans
		}
	}
	ans.named = t.Templates()
	// set the symbols
	for _, t := range ans.named {
		ans.stack = append(ans.stack, t.Root)
		ans.findSymbols()
		if t.Name() != "" {
			// defining a template. The pos is just after {{define...}} (or {{block...}}?)
			at, sz := ans.FindLiteralBefore(int(t.Root.Pos))
			s := symbol{start: at, length: sz, name: t.Name(), kind: protocol.Namespace, vardef: true}
			ans.symbols = append(ans.symbols, s)
		}
	}

	sort.Slice(ans.symbols, func(i, j int) bool {
		left, right := ans.symbols[i], ans.symbols[j]
		if left.start != right.start {
			return left.start < right.start
		}
		if left.vardef != right.vardef {
			return left.vardef
		}
		return left.kind < right.kind
	})
	return ans
}

// FindLiteralBefore locates the first preceding string literal
// returning its position and length in buf
// or returns -1 if there is none. Assume "", rather than ``, for now
func (p *Parsed) FindLiteralBefore(pos int) (int, int) {
	left, right := -1, -1
	for i := pos - 1; i >= 0; i-- {
		if p.buf[i] != '"' {
			continue
		}
		if right == -1 {
			right = i
			continue
		}
		left = i
		break
	}
	if left == -1 {
		return -1, 0
	}
	return left + 1, right - left - 1
}

var (
	parseErrR = regexp.MustCompile(`template:.*function "([^"]+)" not defined`)
)

func (p *Parsed) setTokens() {
	const (
		// InRaw and InString only occur inside an action (SeenLeft)
		Start = iota
		InRaw
		InString
		SeenLeft
	)
	state := Start
	var left, oldState int
	for n := 0; n < len(p.buf); n++ {
		c := p.buf[n]
		switch state {
		case InRaw:
			if c == '`' {
				state = oldState
			}
		case InString:
			if c == '"' && !isEscaped(p.buf[:n]) {
				state = oldState
			}
		case SeenLeft:
			if c == '`' {
				oldState = state // it's SeenLeft, but a little clearer this way
				state = InRaw
				continue
			}
			if c == '"' {
				oldState = state
				state = InString
				continue
			}
			if bytes.HasPrefix(p.buf[n:], Right) {
				right := n + len(Right)
				tok := Token{Start: left,
					End:       right,
					Multiline: bytes.Contains(p.buf[left:right], []byte{'\n'}),
				}
				p.tokens = append(p.tokens, tok)
				state = Start
			}
			// If we see (unquoted) Left then the original left is probably the user
			// typing. Suppress the original left
			if bytes.HasPrefix(p.buf[n:], Left) {
				p.elideAt(left)
				left = n
				n += len(Left) - 1 // skip the rest
			}
		case Start:
			if bytes.HasPrefix(p.buf[n:], Left) {
				left = n
				state = SeenLeft
				n += len(Left) - 1 // skip the rest (avoids {{{ bug)
			}
		}
	}
	// this error occurs after typing {{ at the end of the file
	if state != Start {
		// Unclosed Left. remove the Left at left
		p.elideAt(left)
	}
}

func (p *Parsed) elideAt(left int) {
	if p.elided == nil {
		// p.buf is the same buffer that v.Read() returns, so copy it.
		// (otherwise the next time it's parsed, elided information is lost)
		b := make([]byte, len(p.buf))
		copy(b, p.buf)
		p.buf = b
	}
	for i := 0; i < len(Left); i++ {
		p.buf[left+i] = ' '
	}
	p.elided = append(p.elided, left)
}

// isEscaped reports whether the byte after buf is escaped
func isEscaped(buf []byte) bool {
	backSlashes := 0
	for j := len(buf) - 1; j >= 0 && buf[j] == '\\'; j-- {
		backSlashes++
	}
	return backSlashes%2 == 1
}

func (p *Parsed) Tokens() []Token {
	return p.tokens
}

func (p *Parsed) utf16len(buf []byte) int {
	cnt := 0
	if !p.nonASCII {
		return len(buf)
	}
	// we need a utf16len(rune), but we don't have it
	for _, r := range string(buf) {
		cnt++
		if r >= 1<<16 {
			cnt++
		}
	}
	return cnt
}

func (p *Parsed) TokenSize(t Token) (int, error) {
	if t.Multiline {
		return -1, fmt.Errorf("TokenSize called with Multiline token %#v", t)
	}
	ans := p.utf16len(p.buf[t.Start:t.End])
	return ans, nil
}

// RuneCount counts runes in a line
func (p *Parsed) RuneCount(l, s, e uint32) uint32 {
	start := p.nls[l] + 1 + int(s)
	end := int(e)
	if e == 0 || int(e) >= p.nls[l+1] {
		end = p.nls[l+1]
	}
	return uint32(utf8.RuneCount(p.buf[start:end]))
}

// LineCol converts from a 0-based byte offset to 0-based line, col. col in runes
func (p *Parsed) LineCol(x int) (uint32, uint32) {
	if x < p.check {
		p.lastnl = 0
	}
	p.check = x
	for i := p.lastnl; i < len(p.nls); i++ {
		if p.nls[i] <= x {
			continue
		}
		p.lastnl = i
		var count int
		if i > 0 && x == p.nls[i-1] { // \n
			count = 0
		} else {
			count = p.utf16len(p.buf[p.nls[i-1]+1 : x])
		}
		return uint32(i - 1), uint32(count)
	}
	if x == len(p.buf)-1 { // trailing \n
		return uint32(len(p.nls) - 1), 0
	}
	// shouldn't happen
	for i := 1; i < 4; i++ {
		_, f, l, ok := runtime.Caller(i)
		if !ok {
			break
		}
		log.Printf("%d: %s:%d", i, f, l)
	}

	msg := fmt.Errorf("LineCol off the end, %d of %d, nls=%v, %q", x, len(p.buf), p.nls, p.buf[x:])
	event.Error(context.Background(), "internal error", msg)
	return 0, 0
}

// Position produces a protocol.Position from an offset in the template
func (p *Parsed) Position(pos int) protocol.Position {
	line, col := p.LineCol(pos)
	return protocol.Position{Line: line, Character: col}
}

func (p *Parsed) Range(x, length int) protocol.Range {
	line, col := p.LineCol(x)
	ans := protocol.Range{
		Start: protocol.Position{Line: line, Character: col},
		End:   protocol.Position{Line: line, Character: col + uint32(length)},
	}
	return ans
}

// FromPosition translates a protocol.Position into an offset into the template
func (p *Parsed) FromPosition(x protocol.Position) int {
	l, c := int(x.Line), int(x.Character)
	if l >= len(p.nls) || p.nls[l]+1 >= len(p.buf) {
		// paranoia to avoid panic. return the largest offset
		return len(p.buf)
	}
	line := p.buf[p.nls[l]+1:]
	cnt := 0
	for w := range string(line) {
		if cnt >= c {
			return w + p.nls[l] + 1
		}
		cnt++
	}
	// do we get here? NO
	pos := int(x.Character) + p.nls[int(x.Line)] + 1
	event.Error(context.Background(), "internal error", fmt.Errorf("surprise %#v", x))
	return pos
}

func symAtPosition(fh source.FileHandle, loc protocol.Position) (*symbol, *Parsed, error) {
	buf, err := fh.Read()
	if err != nil {
		return nil, nil, err
	}
	p := parseBuffer(buf)
	pos := p.FromPosition(loc)
	syms := p.SymsAtPos(pos)
	if len(syms) == 0 {
		return nil, p, fmt.Errorf("no symbol found")
	}
	if len(syms) > 1 {
		log.Printf("Hover: %d syms, not 1 %v", len(syms), syms)
	}
	sym := syms[0]
	return &sym, p, nil
}

func (p *Parsed) SymsAtPos(pos int) []symbol {
	ans := []symbol{}
	for _, s := range p.symbols {
		if s.start <= pos && pos < s.start+s.length {
			ans = append(ans, s)
		}
	}
	return ans
}

type wrNode struct {
	p *Parsed
	w io.Writer
}

// WriteNode is for debugging
func (p *Parsed) WriteNode(w io.Writer, n parse.Node) {
	wr := wrNode{p: p, w: w}
	wr.writeNode(n, "")
}

func (wr wrNode) writeNode(n parse.Node, indent string) {
	if n == nil {
		return
	}
	at := func(pos parse.Pos) string {
		line, col := wr.p.LineCol(int(pos))
		return fmt.Sprintf("(%d)%v:%v", pos, line, col)
	}
	switch x := n.(type) {
	case *parse.ActionNode:
		fmt.Fprintf(wr.w, "%sActionNode at %s\n", indent, at(x.Pos))
		wr.writeNode(x.Pipe, indent+". ")
	case *parse.BoolNode:
		fmt.Fprintf(wr.w, "%sBoolNode at %s, %v\n", indent, at(x.Pos), x.True)
	case *parse.BranchNode:
		fmt.Fprintf(wr.w, "%sBranchNode at %s\n", indent, at(x.Pos))
		wr.writeNode(x.Pipe, indent+"Pipe. ")
		wr.writeNode(x.List, indent+"List. ")
		wr.writeNode(x.ElseList, indent+"Else. ")
	case *parse.ChainNode:
		fmt.Fprintf(wr.w, "%sChainNode at %s, %v\n", indent, at(x.Pos), x.Field)
	case *parse.CommandNode:
		fmt.Fprintf(wr.w, "%sCommandNode at %s, %d children\n", indent, at(x.Pos), len(x.Args))
		for _, a := range x.Args {
			wr.writeNode(a, indent+". ")
		}
	//case *parse.CommentNode: // 1.16
	case *parse.DotNode:
		fmt.Fprintf(wr.w, "%sDotNode at %s\n", indent, at(x.Pos))
	case *parse.FieldNode:
		fmt.Fprintf(wr.w, "%sFieldNode at %s, %v\n", indent, at(x.Pos), x.Ident)
	case *parse.IdentifierNode:
		fmt.Fprintf(wr.w, "%sIdentifierNode at %s, %v\n", indent, at(x.Pos), x.Ident)
	case *parse.IfNode:
		fmt.Fprintf(wr.w, "%sIfNode at %s\n", indent, at(x.Pos))
		wr.writeNode(&x.BranchNode, indent+". ")
	case *parse.ListNode:
		if x == nil {
			return // nil BranchNode.ElseList
		}
		fmt.Fprintf(wr.w, "%sListNode at %s, %d children\n", indent, at(x.Pos), len(x.Nodes))
		for _, n := range x.Nodes {
			wr.writeNode(n, indent+". ")
		}
	case *parse.NilNode:
		fmt.Fprintf(wr.w, "%sNilNode at %s\n", indent, at(x.Pos))
	case *parse.NumberNode:
		fmt.Fprintf(wr.w, "%sNumberNode at %s, %s\n", indent, at(x.Pos), x.Text)
	case *parse.PipeNode:
		if x == nil {
			return // {{template "xxx"}}
		}
		fmt.Fprintf(wr.w, "%sPipeNode at %s, %d vars, %d cmds, IsAssign:%v\n",
			indent, at(x.Pos), len(x.Decl), len(x.Cmds), x.IsAssign)
		for _, d := range x.Decl {
			wr.writeNode(d, indent+"Decl. ")
		}
		for _, c := range x.Cmds {
			wr.writeNode(c, indent+"Cmd. ")
		}
	case *parse.RangeNode:
		fmt.Fprintf(wr.w, "%sRangeNode at %s\n", indent, at(x.Pos))
		wr.writeNode(&x.BranchNode, indent+". ")
	case *parse.StringNode:
		fmt.Fprintf(wr.w, "%sStringNode at %s, %s\n", indent, at(x.Pos), x.Quoted)
	case *parse.TemplateNode:
		fmt.Fprintf(wr.w, "%sTemplateNode at %s, %s\n", indent, at(x.Pos), x.Name)
		wr.writeNode(x.Pipe, indent+". ")
	case *parse.TextNode:
		fmt.Fprintf(wr.w, "%sTextNode at %s, len %d\n", indent, at(x.Pos), len(x.Text))
	case *parse.VariableNode:
		fmt.Fprintf(wr.w, "%sVariableNode at %s, %v\n", indent, at(x.Pos), x.Ident)
	case *parse.WithNode:
		fmt.Fprintf(wr.w, "%sWithNode at %s\n", indent, at(x.Pos))
		wr.writeNode(&x.BranchNode, indent+". ")
	}
}

// short prints at most 40 bytes of node.String(), for debugging
func short(n parse.Node) (ret string) {
	defer func() {
		if x := recover(); x != nil {
			// all because of typed nils
			ret = "NIL"
		}
	}()
	s := n.String()
	if len(s) > 40 {
		return s[:40] + "..."
	}
	return s
}

var kindNames = []string{"", "File", "Module", "Namespace", "Package", "Class", "Method", "Property",
	"Field", "Constructor", "Enum", "Interface", "Function", "Variable", "Constant", "String",
	"Number", "Boolean", "Array", "Object", "Key", "Null", "EnumMember", "Struct", "Event",
	"Operator", "TypeParameter"}

func kindStr(k protocol.SymbolKind) string {
	n := int(k)
	if n < 1 || n >= len(kindNames) {
		return fmt.Sprintf("?SymbolKind %d?", n)
	}
	return kindNames[n]
}
