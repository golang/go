// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements printing of syntax trees in source format.

package syntax

import (
	"fmt"
	"io"
	"strings"
)

// Form controls print formatting.
type Form uint

const (
	_         Form = iota // default
	LineForm              // use spaces instead of linebreaks where possible
	ShortForm             // like LineForm but print "…" for non-empty function or composite literal bodies
)

// Fprint prints node x to w in the specified form.
// It returns the number of bytes written, and whether there was an error.
func Fprint(w io.Writer, x Node, form Form) (n int, err error) {
	p := printer{
		output:     w,
		form:       form,
		linebreaks: form == 0,
	}

	defer func() {
		n = p.written
		if e := recover(); e != nil {
			err = e.(writeError).err // re-panics if it's not a writeError
		}
	}()

	p.print(x)
	p.flush(_EOF)

	return
}

// String is a convenience function that prints n in ShortForm
// and returns the printed string.
func String(n Node) string {
	var buf strings.Builder
	_, err := Fprint(&buf, n, ShortForm)
	if err != nil {
		fmt.Fprintf(&buf, "<<< ERROR: %s", err)
	}
	return buf.String()
}

type ctrlSymbol int

const (
	none ctrlSymbol = iota
	semi
	blank
	newline
	indent
	outdent
	// comment
	// eolComment
)

type whitespace struct {
	last token
	kind ctrlSymbol
	//text string // comment text (possibly ""); valid if kind == comment
}

type printer struct {
	output     io.Writer
	written    int // number of bytes written
	form       Form
	linebreaks bool // print linebreaks instead of semis

	indent  int // current indentation level
	nlcount int // number of consecutive newlines

	pending []whitespace // pending whitespace
	lastTok token        // last token (after any pending semi) processed by print
}

// write is a thin wrapper around p.output.Write
// that takes care of accounting and error handling.
func (p *printer) write(data []byte) {
	n, err := p.output.Write(data)
	p.written += n
	if err != nil {
		panic(writeError{err})
	}
}

var (
	tabBytes    = []byte("\t\t\t\t\t\t\t\t")
	newlineByte = []byte("\n")
	blankByte   = []byte(" ")
)

func (p *printer) writeBytes(data []byte) {
	if len(data) == 0 {
		panic("expected non-empty []byte")
	}
	if p.nlcount > 0 && p.indent > 0 {
		// write indentation
		n := p.indent
		for n > len(tabBytes) {
			p.write(tabBytes)
			n -= len(tabBytes)
		}
		p.write(tabBytes[:n])
	}
	p.write(data)
	p.nlcount = 0
}

func (p *printer) writeString(s string) {
	p.writeBytes([]byte(s))
}

// If impliesSemi returns true for a non-blank line's final token tok,
// a semicolon is automatically inserted. Vice versa, a semicolon may
// be omitted in those cases.
func impliesSemi(tok token) bool {
	switch tok {
	case _Name,
		_Break, _Continue, _Fallthrough, _Return,
		/*_Inc, _Dec,*/ _Rparen, _Rbrack, _Rbrace: // TODO(gri) fix this
		return true
	}
	return false
}

// TODO(gri) provide table of []byte values for all tokens to avoid repeated string conversion

func lineComment(text string) bool {
	return strings.HasPrefix(text, "//")
}

func (p *printer) addWhitespace(kind ctrlSymbol, text string) {
	p.pending = append(p.pending, whitespace{p.lastTok, kind /*text*/})
	switch kind {
	case semi:
		p.lastTok = _Semi
	case newline:
		p.lastTok = 0
		// TODO(gri) do we need to handle /*-style comments containing newlines here?
	}
}

func (p *printer) flush(next token) {
	// eliminate semis and redundant whitespace
	sawNewline := next == _EOF
	sawParen := next == _Rparen || next == _Rbrace
	for i := len(p.pending) - 1; i >= 0; i-- {
		switch p.pending[i].kind {
		case semi:
			k := semi
			if sawParen {
				sawParen = false
				k = none // eliminate semi
			} else if sawNewline && impliesSemi(p.pending[i].last) {
				sawNewline = false
				k = none // eliminate semi
			}
			p.pending[i].kind = k
		case newline:
			sawNewline = true
		case blank, indent, outdent:
			// nothing to do
		// case comment:
		// 	// A multi-line comment acts like a newline; and a ""
		// 	// comment implies by definition at least one newline.
		// 	if text := p.pending[i].text; strings.HasPrefix(text, "/*") && strings.ContainsRune(text, '\n') {
		// 		sawNewline = true
		// 	}
		// case eolComment:
		// 	// TODO(gri) act depending on sawNewline
		default:
			panic("unreachable")
		}
	}

	// print pending
	prev := none
	for i := range p.pending {
		switch p.pending[i].kind {
		case none:
			// nothing to do
		case semi:
			p.writeString(";")
			p.nlcount = 0
			prev = semi
		case blank:
			if prev != blank {
				// at most one blank
				p.writeBytes(blankByte)
				p.nlcount = 0
				prev = blank
			}
		case newline:
			const maxEmptyLines = 1
			if p.nlcount <= maxEmptyLines {
				p.write(newlineByte)
				p.nlcount++
				prev = newline
			}
		case indent:
			p.indent++
		case outdent:
			p.indent--
			if p.indent < 0 {
				panic("negative indentation")
			}
		// case comment:
		// 	if text := p.pending[i].text; text != "" {
		// 		p.writeString(text)
		// 		p.nlcount = 0
		// 		prev = comment
		// 	}
		// 	// TODO(gri) should check that line comments are always followed by newline
		default:
			panic("unreachable")
		}
	}

	p.pending = p.pending[:0] // re-use underlying array
}

func mayCombine(prev token, next byte) (b bool) {
	return // for now
	// switch prev {
	// case lexical.Int:
	// 	b = next == '.' // 1.
	// case lexical.Add:
	// 	b = next == '+' // ++
	// case lexical.Sub:
	// 	b = next == '-' // --
	// case lexical.Quo:
	// 	b = next == '*' // /*
	// case lexical.Lss:
	// 	b = next == '-' || next == '<' // <- or <<
	// case lexical.And:
	// 	b = next == '&' || next == '^' // && or &^
	// }
	// return
}

func (p *printer) print(args ...interface{}) {
	for i := 0; i < len(args); i++ {
		switch x := args[i].(type) {
		case nil:
			// we should not reach here but don't crash

		case Node:
			p.printNode(x)

		case token:
			// _Name implies an immediately following string
			// argument which is the actual value to print.
			var s string
			if x == _Name {
				i++
				if i >= len(args) {
					panic("missing string argument after _Name")
				}
				s = args[i].(string)
			} else {
				s = x.String()
			}

			// TODO(gri) This check seems at the wrong place since it doesn't
			//           take into account pending white space.
			if mayCombine(p.lastTok, s[0]) {
				panic("adjacent tokens combine without whitespace")
			}

			if x == _Semi {
				// delay printing of semi
				p.addWhitespace(semi, "")
			} else {
				p.flush(x)
				p.writeString(s)
				p.nlcount = 0
				p.lastTok = x
			}

		case Operator:
			if x != 0 {
				p.flush(_Operator)
				p.writeString(x.String())
			}

		case ctrlSymbol:
			switch x {
			case none, semi /*, comment*/ :
				panic("unreachable")
			case newline:
				// TODO(gri) need to handle mandatory newlines after a //-style comment
				if !p.linebreaks {
					x = blank
				}
			}
			p.addWhitespace(x, "")

		// case *Comment: // comments are not Nodes
		// 	p.addWhitespace(comment, x.Text)

		default:
			panic(fmt.Sprintf("unexpected argument %v (%T)", x, x))
		}
	}
}

func (p *printer) printNode(n Node) {
	// ncom := *n.Comments()
	// if ncom != nil {
	// 	// TODO(gri) in general we cannot make assumptions about whether
	// 	// a comment is a /*- or a //-style comment since the syntax
	// 	// tree may have been manipulated. Need to make sure the correct
	// 	// whitespace is emitted.
	// 	for _, c := range ncom.Alone {
	// 		p.print(c, newline)
	// 	}
	// 	for _, c := range ncom.Before {
	// 		if c.Text == "" || lineComment(c.Text) {
	// 			panic("unexpected empty line or //-style 'before' comment")
	// 		}
	// 		p.print(c, blank)
	// 	}
	// }

	p.printRawNode(n)

	// if ncom != nil && len(ncom.After) > 0 {
	// 	for i, c := range ncom.After {
	// 		if i+1 < len(ncom.After) {
	// 			if c.Text == "" || lineComment(c.Text) {
	// 				panic("unexpected empty line or //-style non-final 'after' comment")
	// 			}
	// 		}
	// 		p.print(blank, c)
	// 	}
	// 	//p.print(newline)
	// }
}

func (p *printer) printRawNode(n Node) {
	switch n := n.(type) {
	case nil:
		// we should not reach here but don't crash

	// expressions and types
	case *BadExpr:
		p.print(_Name, "<bad expr>")

	case *Name:
		p.print(_Name, n.Value) // _Name requires actual value following immediately

	case *BasicLit:
		p.print(_Name, n.Value) // _Name requires actual value following immediately

	case *FuncLit:
		p.print(n.Type, blank)
		if n.Body != nil {
			if p.form == ShortForm {
				p.print(_Lbrace)
				if len(n.Body.List) > 0 {
					p.print(_Name, "…")
				}
				p.print(_Rbrace)
			} else {
				p.print(n.Body)
			}
		}

	case *CompositeLit:
		if n.Type != nil {
			p.print(n.Type)
		}
		p.print(_Lbrace)
		if p.form == ShortForm {
			if len(n.ElemList) > 0 {
				p.print(_Name, "…")
			}
		} else {
			if n.NKeys > 0 && n.NKeys == len(n.ElemList) {
				p.printExprLines(n.ElemList)
			} else {
				p.printExprList(n.ElemList)
			}
		}
		p.print(_Rbrace)

	case *ParenExpr:
		p.print(_Lparen, n.X, _Rparen)

	case *SelectorExpr:
		p.print(n.X, _Dot, n.Sel)

	case *IndexExpr:
		p.print(n.X, _Lbrack, n.Index, _Rbrack)

	case *SliceExpr:
		p.print(n.X, _Lbrack)
		if i := n.Index[0]; i != nil {
			p.printNode(i)
		}
		p.print(_Colon)
		if j := n.Index[1]; j != nil {
			p.printNode(j)
		}
		if k := n.Index[2]; k != nil {
			p.print(_Colon, k)
		}
		p.print(_Rbrack)

	case *AssertExpr:
		p.print(n.X, _Dot, _Lparen, n.Type, _Rparen)

	case *TypeSwitchGuard:
		if n.Lhs != nil {
			p.print(n.Lhs, blank, _Define, blank)
		}
		p.print(n.X, _Dot, _Lparen, _Type, _Rparen)

	case *CallExpr:
		p.print(n.Fun, _Lparen)
		p.printExprList(n.ArgList)
		if n.HasDots {
			p.print(_DotDotDot)
		}
		p.print(_Rparen)

	case *Operation:
		if n.Y == nil {
			// unary expr
			p.print(n.Op)
			// if n.Op == lexical.Range {
			// 	p.print(blank)
			// }
			p.print(n.X)
		} else {
			// binary expr
			// TODO(gri) eventually take precedence into account
			// to control possibly missing parentheses
			p.print(n.X, blank, n.Op, blank, n.Y)
		}

	case *KeyValueExpr:
		p.print(n.Key, _Colon, blank, n.Value)

	case *ListExpr:
		p.printExprList(n.ElemList)

	case *ArrayType:
		var len interface{} = _DotDotDot
		if n.Len != nil {
			len = n.Len
		}
		p.print(_Lbrack, len, _Rbrack, n.Elem)

	case *SliceType:
		p.print(_Lbrack, _Rbrack, n.Elem)

	case *DotsType:
		p.print(_DotDotDot, n.Elem)

	case *StructType:
		p.print(_Struct)
		if len(n.FieldList) > 0 && p.linebreaks {
			p.print(blank)
		}
		p.print(_Lbrace)
		if len(n.FieldList) > 0 {
			if p.linebreaks {
				p.print(newline, indent)
				p.printFieldList(n.FieldList, n.TagList, _Semi)
				p.print(outdent, newline)
			} else {
				p.printFieldList(n.FieldList, n.TagList, _Semi)
			}
		}
		p.print(_Rbrace)

	case *FuncType:
		p.print(_Func)
		p.printSignature(n)

	case *InterfaceType:
		p.print(_Interface)
		if p.linebreaks && len(n.MethodList) > 1 {
			p.print(blank)
			p.print(_Lbrace)
			p.print(newline, indent)
			p.printMethodList(n.MethodList)
			p.print(outdent, newline)
		} else {
			p.print(_Lbrace)
			p.printMethodList(n.MethodList)
		}
		p.print(_Rbrace)

	case *MapType:
		p.print(_Map, _Lbrack, n.Key, _Rbrack, n.Value)

	case *ChanType:
		if n.Dir == RecvOnly {
			p.print(_Arrow)
		}
		p.print(_Chan)
		if n.Dir == SendOnly {
			p.print(_Arrow)
		}
		p.print(blank)
		if e, _ := n.Elem.(*ChanType); n.Dir == 0 && e != nil && e.Dir == RecvOnly {
			// don't print chan (<-chan T) as chan <-chan T
			p.print(_Lparen)
			p.print(n.Elem)
			p.print(_Rparen)
		} else {
			p.print(n.Elem)
		}

	// statements
	case *DeclStmt:
		p.printDecl(n.DeclList)

	case *EmptyStmt:
		// nothing to print

	case *LabeledStmt:
		p.print(outdent, n.Label, _Colon, indent, newline, n.Stmt)

	case *ExprStmt:
		p.print(n.X)

	case *SendStmt:
		p.print(n.Chan, blank, _Arrow, blank, n.Value)

	case *AssignStmt:
		p.print(n.Lhs)
		if n.Rhs == nil {
			// TODO(gri) This is going to break the mayCombine
			//           check once we enable that again.
			p.print(n.Op, n.Op) // ++ or --
		} else {
			p.print(blank, n.Op, _Assign, blank)
			p.print(n.Rhs)
		}

	case *CallStmt:
		p.print(n.Tok, blank, n.Call)

	case *ReturnStmt:
		p.print(_Return)
		if n.Results != nil {
			p.print(blank, n.Results)
		}

	case *BranchStmt:
		p.print(n.Tok)
		if n.Label != nil {
			p.print(blank, n.Label)
		}

	case *BlockStmt:
		p.print(_Lbrace)
		if len(n.List) > 0 {
			p.print(newline, indent)
			p.printStmtList(n.List, true)
			p.print(outdent, newline)
		}
		p.print(_Rbrace)

	case *IfStmt:
		p.print(_If, blank)
		if n.Init != nil {
			p.print(n.Init, _Semi, blank)
		}
		p.print(n.Cond, blank, n.Then)
		if n.Else != nil {
			p.print(blank, _Else, blank, n.Else)
		}

	case *SwitchStmt:
		p.print(_Switch, blank)
		if n.Init != nil {
			p.print(n.Init, _Semi, blank)
		}
		if n.Tag != nil {
			p.print(n.Tag, blank)
		}
		p.printSwitchBody(n.Body)

	case *SelectStmt:
		p.print(_Select, blank) // for now
		p.printSelectBody(n.Body)

	case *RangeClause:
		if n.Lhs != nil {
			tok := _Assign
			if n.Def {
				tok = _Define
			}
			p.print(n.Lhs, blank, tok, blank)
		}
		p.print(_Range, blank, n.X)

	case *ForStmt:
		p.print(_For, blank)
		if n.Init == nil && n.Post == nil {
			if n.Cond != nil {
				p.print(n.Cond, blank)
			}
		} else {
			if n.Init != nil {
				p.print(n.Init)
				// TODO(gri) clean this up
				if _, ok := n.Init.(*RangeClause); ok {
					p.print(blank, n.Body)
					break
				}
			}
			p.print(_Semi, blank)
			if n.Cond != nil {
				p.print(n.Cond)
			}
			p.print(_Semi, blank)
			if n.Post != nil {
				p.print(n.Post, blank)
			}
		}
		p.print(n.Body)

	case *ImportDecl:
		if n.Group == nil {
			p.print(_Import, blank)
		}
		if n.LocalPkgName != nil {
			p.print(n.LocalPkgName, blank)
		}
		p.print(n.Path)

	case *ConstDecl:
		if n.Group == nil {
			p.print(_Const, blank)
		}
		p.printNameList(n.NameList)
		if n.Type != nil {
			p.print(blank, n.Type)
		}
		if n.Values != nil {
			p.print(blank, _Assign, blank, n.Values)
		}

	case *TypeDecl:
		if n.Group == nil {
			p.print(_Type, blank)
		}
		p.print(n.Name)
		if n.TParamList != nil {
			p.printParameterList(n.TParamList, _Type)
		}
		p.print(blank)
		if n.Alias {
			p.print(_Assign, blank)
		}
		p.print(n.Type)

	case *VarDecl:
		if n.Group == nil {
			p.print(_Var, blank)
		}
		p.printNameList(n.NameList)
		if n.Type != nil {
			p.print(blank, n.Type)
		}
		if n.Values != nil {
			p.print(blank, _Assign, blank, n.Values)
		}

	case *FuncDecl:
		p.print(_Func, blank)
		if r := n.Recv; r != nil {
			p.print(_Lparen)
			if r.Name != nil {
				p.print(r.Name, blank)
			}
			p.printNode(r.Type)
			p.print(_Rparen, blank)
		}
		p.print(n.Name)
		if n.TParamList != nil {
			p.printParameterList(n.TParamList, _Func)
		}
		p.printSignature(n.Type)
		if n.Body != nil {
			p.print(blank, n.Body)
		}

	case *printGroup:
		p.print(n.Tok, blank, _Lparen)
		if len(n.Decls) > 0 {
			p.print(newline, indent)
			for _, d := range n.Decls {
				p.printNode(d)
				p.print(_Semi, newline)
			}
			p.print(outdent)
		}
		p.print(_Rparen)

	// files
	case *File:
		p.print(_Package, blank, n.PkgName)
		if len(n.DeclList) > 0 {
			p.print(_Semi, newline, newline)
			p.printDeclList(n.DeclList)
		}

	default:
		panic(fmt.Sprintf("syntax.Iterate: unexpected node type %T", n))
	}
}

func (p *printer) printFields(fields []*Field, tags []*BasicLit, i, j int) {
	if i+1 == j && fields[i].Name == nil {
		// anonymous field
		p.printNode(fields[i].Type)
	} else {
		for k, f := range fields[i:j] {
			if k > 0 {
				p.print(_Comma, blank)
			}
			p.printNode(f.Name)
		}
		p.print(blank)
		p.printNode(fields[i].Type)
	}
	if i < len(tags) && tags[i] != nil {
		p.print(blank)
		p.printNode(tags[i])
	}
}

func (p *printer) printFieldList(fields []*Field, tags []*BasicLit, sep token) {
	i0 := 0
	var typ Expr
	for i, f := range fields {
		if f.Name == nil || f.Type != typ {
			if i0 < i {
				p.printFields(fields, tags, i0, i)
				p.print(sep, newline)
				i0 = i
			}
			typ = f.Type
		}
	}
	p.printFields(fields, tags, i0, len(fields))
}

func (p *printer) printMethodList(methods []*Field) {
	for i, m := range methods {
		if i > 0 {
			p.print(_Semi, newline)
		}
		if m.Name != nil {
			p.printNode(m.Name)
			p.printSignature(m.Type.(*FuncType))
		} else {
			p.printNode(m.Type)
		}
	}
}

func (p *printer) printNameList(list []*Name) {
	for i, x := range list {
		if i > 0 {
			p.print(_Comma, blank)
		}
		p.printNode(x)
	}
}

func (p *printer) printExprList(list []Expr) {
	for i, x := range list {
		if i > 0 {
			p.print(_Comma, blank)
		}
		p.printNode(x)
	}
}

func (p *printer) printExprLines(list []Expr) {
	if len(list) > 0 {
		p.print(newline, indent)
		for _, x := range list {
			p.print(x, _Comma, newline)
		}
		p.print(outdent)
	}
}

func groupFor(d Decl) (token, *Group) {
	switch d := d.(type) {
	case *ImportDecl:
		return _Import, d.Group
	case *ConstDecl:
		return _Const, d.Group
	case *TypeDecl:
		return _Type, d.Group
	case *VarDecl:
		return _Var, d.Group
	case *FuncDecl:
		return _Func, nil
	default:
		panic("unreachable")
	}
}

type printGroup struct {
	node
	Tok   token
	Decls []Decl
}

func (p *printer) printDecl(list []Decl) {
	tok, group := groupFor(list[0])

	if group == nil {
		if len(list) != 1 {
			panic("unreachable")
		}
		p.printNode(list[0])
		return
	}

	// if _, ok := list[0].(*EmptyDecl); ok {
	// 	if len(list) != 1 {
	// 		panic("unreachable")
	// 	}
	// 	// TODO(gri) if there are comments inside the empty
	// 	// group, we may need to keep the list non-nil
	// 	list = nil
	// }

	// printGroup is here for consistent comment handling
	// (this is not yet used)
	var pg printGroup
	// *pg.Comments() = *group.Comments()
	pg.Tok = tok
	pg.Decls = list
	p.printNode(&pg)
}

func (p *printer) printDeclList(list []Decl) {
	i0 := 0
	var tok token
	var group *Group
	for i, x := range list {
		if s, g := groupFor(x); g == nil || g != group {
			if i0 < i {
				p.printDecl(list[i0:i])
				p.print(_Semi, newline)
				// print empty line between different declaration groups,
				// different kinds of declarations, or between functions
				if g != group || s != tok || s == _Func {
					p.print(newline)
				}
				i0 = i
			}
			tok, group = s, g
		}
	}
	p.printDecl(list[i0:])
}

func (p *printer) printSignature(sig *FuncType) {
	p.printParameterList(sig.ParamList, 0)
	if list := sig.ResultList; list != nil {
		p.print(blank)
		if len(list) == 1 && list[0].Name == nil {
			p.printNode(list[0].Type)
		} else {
			p.printParameterList(list, 0)
		}
	}
}

// If tok != 0 print a type parameter list: tok == _Type means
// a type parameter list for a type, tok == _Func means a type
// parameter list for a func.
func (p *printer) printParameterList(list []*Field, tok token) {
	open, close := _Lparen, _Rparen
	if tok != 0 {
		open, close = _Lbrack, _Rbrack
	}
	p.print(open)
	for i, f := range list {
		if i > 0 {
			p.print(_Comma, blank)
		}
		if f.Name != nil {
			p.printNode(f.Name)
			if i+1 < len(list) {
				f1 := list[i+1]
				if f1.Name != nil && f1.Type == f.Type {
					continue // no need to print type
				}
			}
			p.print(blank)
		}
		p.printNode(f.Type)
	}
	// A type parameter list [P T] where the name P and the type expression T syntactically
	// combine to another valid (value) expression requires a trailing comma, as in [P *T,]
	// (or an enclosing interface as in [P interface(*T)]), so that the type parameter list
	// is not parsed as an array length [P*T].
	if tok == _Type && len(list) == 1 && combinesWithName(list[0].Type) {
		p.print(_Comma)
	}
	p.print(close)
}

// combinesWithName reports whether a name followed by the expression x
// syntactically combines to another valid (value) expression. For instance
// using *T for x, "name *T" syntactically appears as the expression x*T.
// On the other hand, using  P|Q or *P|~Q for x, "name P|Q" or "name *P|~Q"
// cannot be combined into a valid (value) expression.
func combinesWithName(x Expr) bool {
	switch x := x.(type) {
	case *Operation:
		if x.Y == nil {
			// name *x.X combines to name*x.X if x.X is not a type element
			return x.Op == Mul && !isTypeElem(x.X)
		}
		// binary expressions
		return combinesWithName(x.X) && !isTypeElem(x.Y)
	case *ParenExpr:
		// Note that the parser strips parentheses in these cases
		// (see extractName, parser.typeOrNil) unless keep_parens
		// is set, so we should never reach here.
		// Do the right thing (rather than panic) for testing and
		// in case we change parser behavior.
		// See also go.dev/issues/69206.
		return !isTypeElem(x.X)
	}
	return false
}

func (p *printer) printStmtList(list []Stmt, braces bool) {
	for i, x := range list {
		p.print(x, _Semi)
		if i+1 < len(list) {
			p.print(newline)
		} else if braces {
			// Print an extra semicolon if the last statement is
			// an empty statement and we are in a braced block
			// because one semicolon is automatically removed.
			if _, ok := x.(*EmptyStmt); ok {
				p.print(x, _Semi)
			}
		}
	}
}

func (p *printer) printSwitchBody(list []*CaseClause) {
	p.print(_Lbrace)
	if len(list) > 0 {
		p.print(newline)
		for i, c := range list {
			p.printCaseClause(c, i+1 == len(list))
			p.print(newline)
		}
	}
	p.print(_Rbrace)
}

func (p *printer) printSelectBody(list []*CommClause) {
	p.print(_Lbrace)
	if len(list) > 0 {
		p.print(newline)
		for i, c := range list {
			p.printCommClause(c, i+1 == len(list))
			p.print(newline)
		}
	}
	p.print(_Rbrace)
}

func (p *printer) printCaseClause(c *CaseClause, braces bool) {
	if c.Cases != nil {
		p.print(_Case, blank, c.Cases)
	} else {
		p.print(_Default)
	}
	p.print(_Colon)
	if len(c.Body) > 0 {
		p.print(newline, indent)
		p.printStmtList(c.Body, braces)
		p.print(outdent)
	}
}

func (p *printer) printCommClause(c *CommClause, braces bool) {
	if c.Comm != nil {
		p.print(_Case, blank)
		p.print(c.Comm)
	} else {
		p.print(_Default)
	}
	p.print(_Colon)
	if len(c.Body) > 0 {
		p.print(newline, indent)
		p.printStmtList(c.Body, braces)
		p.print(outdent)
	}
}
