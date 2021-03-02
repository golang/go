// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"bytes"
	"fmt"
	"text/template/parse"
	"unicode/utf8"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

// in local coordinates, to be translated to protocol.DocumentSymbol
type symbol struct {
	start  int // for sorting
	length int // in runes (unicode code points)
	name   string
	kind   protocol.SymbolKind
	vardef bool // is this a variable definition?
	// do we care about selection range, or children?
	// no children yet, and selection range is the same as range
}

func (s symbol) String() string {
	return fmt.Sprintf("{%d,%d,%s,%s,%v}", s.start, s.length, s.name, s.kind, s.vardef)
}

// for FieldNode or VariableNode (or ChainNode?)
func (p *Parsed) fields(flds []string, x parse.Node) []symbol {
	ans := []symbol{}
	// guessing that there are no embedded blanks allowed. The doc is unclear
	lookfor := ""
	switch x.(type) {
	case *parse.FieldNode:
		for _, f := range flds {
			lookfor += "." + f // quadratic, but probably ok
		}
	case *parse.VariableNode:
		lookfor = flds[0]
		for i := 1; i < len(flds); i++ {
			lookfor += "." + flds[i]
		}
	case *parse.ChainNode: // PJW, what are these?
		for _, f := range flds {
			lookfor += "." + f // quadratic, but probably ok
		}
	default:
		panic(fmt.Sprintf("%T unexpected in fields()", x))
	}
	if len(lookfor) == 0 {
		panic(fmt.Sprintf("no strings in fields() %#v", x))
	}
	startsAt := int(x.Position())
	ix := bytes.Index(p.buf[startsAt:], []byte(lookfor)) // HasPrefix? PJW?
	if ix < 0 || ix > len(lookfor) {                     // lookfor expected to be at start (or so)
		// probably golang.go/#43388, so back up
		startsAt -= len(flds[0]) + 1
		ix = bytes.Index(p.buf[startsAt:], []byte(lookfor)) // ix might be 1? PJW
		if ix < 0 {
			return ans
		}
	}
	at := ix + startsAt
	for _, f := range flds {
		at += 1 // .
		kind := protocol.Method
		if f[0] == '$' {
			kind = protocol.Variable
		}
		sym := symbol{name: f, kind: kind, start: at, length: utf8.RuneCount([]byte(f))}
		if kind == protocol.Variable && len(p.stack) > 1 {
			if pipe, ok := p.stack[len(p.stack)-2].(*parse.PipeNode); ok {
				for _, y := range pipe.Decl {
					if x == y {
						sym.vardef = true
					}
				}
			}
		}
		ans = append(ans, sym)
		at += len(f)
	}
	return ans
}

func (p *Parsed) findSymbols() {
	if len(p.stack) == 0 {
		return
	}
	n := p.stack[len(p.stack)-1]
	pop := func() {
		p.stack = p.stack[:len(p.stack)-1]
	}
	if n == nil { // allowing nil simplifies the code
		pop()
		return
	}
	nxt := func(nd parse.Node) {
		p.stack = append(p.stack, nd)
		p.findSymbols()
	}
	switch x := n.(type) {
	case *parse.ActionNode:
		nxt(x.Pipe)
	case *parse.BoolNode:
		// need to compute the length from the value
		msg := fmt.Sprintf("%v", x.True)
		p.symbols = append(p.symbols, symbol{start: int(x.Pos), length: len(msg), kind: protocol.Boolean})
	case *parse.BranchNode:
		nxt(x.Pipe)
		nxt(x.List)
		nxt(x.ElseList)
	case *parse.ChainNode:
		p.symbols = append(p.symbols, p.fields(x.Field, x)...)
		nxt(x.Node)
	case *parse.CommandNode:
		for _, a := range x.Args {
			nxt(a)
		}
	//case *parse.CommentNode: // go 1.16
	//	log.Printf("implement %d", x.Type())
	case *parse.DotNode:
		sym := symbol{name: "dot", kind: protocol.Variable, start: int(x.Pos), length: 1}
		p.symbols = append(p.symbols, sym)
	case *parse.FieldNode:
		p.symbols = append(p.symbols, p.fields(x.Ident, x)...)
	case *parse.IdentifierNode:
		sym := symbol{name: x.Ident, kind: protocol.Function, start: int(x.Pos),
			length: utf8.RuneCount([]byte(x.Ident))}
		p.symbols = append(p.symbols, sym)
	case *parse.IfNode:
		nxt(&x.BranchNode)
	case *parse.ListNode:
		if x != nil { // wretched typed nils. Node should have an IfNil
			for _, nd := range x.Nodes {
				nxt(nd)
			}
		}
	case *parse.NilNode:
		sym := symbol{name: "nil", kind: protocol.Constant, start: int(x.Pos), length: 3}
		p.symbols = append(p.symbols, sym)
	case *parse.NumberNode:
		// no name; ascii
		p.symbols = append(p.symbols, symbol{start: int(x.Pos), length: len(x.Text), kind: protocol.Number})
	case *parse.PipeNode:
		if x == nil { // {{template "foo"}}
			return
		}
		for _, d := range x.Decl {
			nxt(d)
		}
		for _, c := range x.Cmds {
			nxt(c)
		}
	case *parse.RangeNode:
		nxt(&x.BranchNode)
	case *parse.StringNode:
		// no name
		sz := utf8.RuneCount([]byte(x.Text))
		p.symbols = append(p.symbols, symbol{start: int(x.Pos), length: sz, kind: protocol.String})
	case *parse.TemplateNode: // invoking a template
		// x.Pos points to the quote before the name
		p.symbols = append(p.symbols, symbol{name: x.Name, kind: protocol.Package, start: int(x.Pos) + 1,
			length: utf8.RuneCount([]byte(x.Name))})
		nxt(x.Pipe)
	case *parse.TextNode:
		if len(x.Text) == 1 && x.Text[0] == '\n' {
			break
		}
		// nothing to report, but build one for hover
		sz := utf8.RuneCount([]byte(x.Text))
		p.symbols = append(p.symbols, symbol{start: int(x.Pos), length: sz, kind: protocol.Constant})
	case *parse.VariableNode:
		p.symbols = append(p.symbols, p.fields(x.Ident, x)...)
	case *parse.WithNode:
		nxt(&x.BranchNode)

	}
	pop()
}

// DocumentSymbols returns a heirarchy of the symbols defined in a template file.
// (The heirarchy is flat. SymbolInformation might be better.)
func DocumentSymbols(snapshot source.Snapshot, fh source.FileHandle) ([]protocol.DocumentSymbol, error) {
	if skipTemplates(snapshot) {
		return nil, nil
	}
	buf, err := fh.Read()
	if err != nil {
		return nil, err
	}
	p := parseBuffer(buf)
	if p.ParseErr != nil {
		return nil, p.ParseErr
	}
	var ans []protocol.DocumentSymbol
	for _, s := range p.symbols {
		if s.kind == protocol.Constant {
			continue
		}
		d := kindStr(s.kind)
		if d == "Namespace" {
			d = "Template"
		}
		if s.vardef {
			d += "(def)"
		} else {
			d += "(use)"
		}
		r := p.Range(s.start, s.length)
		y := protocol.DocumentSymbol{
			Name:           s.name,
			Detail:         d,
			Kind:           s.kind,
			Range:          r,
			SelectionRange: r, // or should this be the entire {{...}}?
		}
		ans = append(ans, y)
	}
	return ans, nil
}
