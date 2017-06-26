// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"cmd/internal/src"
	"fmt"
	"io"
	"strconv"
	"strings"
)

const debug = false
const trace = false

type parser struct {
	base *src.PosBase
	errh ErrorHandler
	mode Mode
	scanner

	first  error  // first error encountered
	pragma Pragma // pragma flags

	fnest  int    // function nesting level (for error handling)
	xnest  int    // expression nesting level (for complit ambiguity resolution)
	indent []byte // tracing support
}

func (p *parser) init(base *src.PosBase, r io.Reader, errh ErrorHandler, pragh PragmaHandler, mode Mode) {
	p.base = base
	p.errh = errh
	p.mode = mode
	p.scanner.init(
		r,
		// Error and pragma handlers for scanner.
		// Because the (line, col) positions passed to these
		// handlers are always at or after the current reading
		// position, it is save to use the most recent position
		// base to compute the corresponding Pos value.
		func(line, col uint, msg string) {
			p.error_at(p.pos_at(line, col), msg)
		},
		func(line, col uint, text string) {
			if strings.HasPrefix(text, "line ") {
				p.updateBase(line, col+5, text[5:])
				return
			}
			if pragh != nil {
				p.pragma |= pragh(p.pos_at(line, col), text)
			}
		},
	)

	p.first = nil
	p.pragma = 0

	p.fnest = 0
	p.xnest = 0
	p.indent = nil
}

const lineMax = 1<<24 - 1 // TODO(gri) this limit is defined for src.Pos - fix

func (p *parser) updateBase(line, col uint, text string) {
	// Want to use LastIndexByte below but it's not defined in Go1.4 and bootstrap fails.
	i := strings.LastIndex(text, ":") // look from right (Windows filenames may contain ':')
	if i < 0 {
		return // ignore (not a line directive)
	}
	nstr := text[i+1:]
	n, err := strconv.Atoi(nstr)
	if err != nil || n <= 0 || n > lineMax {
		p.error_at(p.pos_at(line, col+uint(i+1)), "invalid line number: "+nstr)
		return
	}
	p.base = src.NewLinePragmaBase(src.MakePos(p.base.Pos().Base(), line, col), text[:i], uint(n))
}

func (p *parser) got(tok token) bool {
	if p.tok == tok {
		p.next()
		return true
	}
	return false
}

func (p *parser) want(tok token) {
	if !p.got(tok) {
		p.syntax_error("expecting " + tok.String())
		p.advance()
	}
}

// ----------------------------------------------------------------------------
// Error handling

// pos_at returns the Pos value for (line, col) and the current position base.
func (p *parser) pos_at(line, col uint) src.Pos {
	return src.MakePos(p.base, line, col)
}

// error reports an error at the given position.
func (p *parser) error_at(pos src.Pos, msg string) {
	err := Error{pos, msg}
	if p.first == nil {
		p.first = err
	}
	if p.errh == nil {
		panic(p.first)
	}
	p.errh(err)
}

// syntax_error_at reports a syntax error at the given position.
func (p *parser) syntax_error_at(pos src.Pos, msg string) {
	if trace {
		defer p.trace("syntax_error (" + msg + ")")()
	}

	if p.tok == _EOF && p.first != nil {
		return // avoid meaningless follow-up errors
	}

	// add punctuation etc. as needed to msg
	switch {
	case msg == "":
		// nothing to do
	case strings.HasPrefix(msg, "in"), strings.HasPrefix(msg, "at"), strings.HasPrefix(msg, "after"):
		msg = " " + msg
	case strings.HasPrefix(msg, "expecting"):
		msg = ", " + msg
	default:
		// plain error - we don't care about current token
		p.error_at(pos, "syntax error: "+msg)
		return
	}

	// determine token string
	var tok string
	switch p.tok {
	case _Name, _Semi:
		tok = p.lit
	case _Literal:
		tok = "literal " + p.lit
	case _Operator:
		tok = p.op.String()
	case _AssignOp:
		tok = p.op.String() + "="
	case _IncOp:
		tok = p.op.String()
		tok += tok
	default:
		tok = tokstring(p.tok)
	}

	p.error_at(pos, "syntax error: unexpected "+tok+msg)
}

// Convenience methods using the current token position.
func (p *parser) pos() src.Pos            { return p.pos_at(p.line, p.col) }
func (p *parser) error(msg string)        { p.error_at(p.pos(), msg) }
func (p *parser) syntax_error(msg string) { p.syntax_error_at(p.pos(), msg) }

// The stopset contains keywords that start a statement.
// They are good synchronization points in case of syntax
// errors and (usually) shouldn't be skipped over.
const stopset uint64 = 1<<_Break |
	1<<_Const |
	1<<_Continue |
	1<<_Defer |
	1<<_Fallthrough |
	1<<_For |
	1<<_Func |
	1<<_Go |
	1<<_Goto |
	1<<_If |
	1<<_Return |
	1<<_Select |
	1<<_Switch |
	1<<_Type |
	1<<_Var

// Advance consumes tokens until it finds a token of the stopset or followlist.
// The stopset is only considered if we are inside a function (p.fnest > 0).
// The followlist is the list of valid tokens that can follow a production;
// if it is empty, exactly one token is consumed to ensure progress.
func (p *parser) advance(followlist ...token) {
	if len(followlist) == 0 {
		p.next()
		return
	}

	// compute follow set
	// (not speed critical, advance is only called in error situations)
	var followset uint64 = 1 << _EOF // never skip over EOF
	for _, tok := range followlist {
		followset |= 1 << tok
	}

	for !(contains(followset, p.tok) || p.fnest > 0 && contains(stopset, p.tok)) {
		p.next()
	}
}

func tokstring(tok token) string {
	switch tok {
	case _EOF:
		return "EOF"
	case _Comma:
		return "comma"
	case _Semi:
		return "semicolon"
	}
	return tok.String()
}

// usage: defer p.trace(msg)()
func (p *parser) trace(msg string) func() {
	fmt.Printf("%5d: %s%s (\n", p.line, p.indent, msg)
	const tab = ". "
	p.indent = append(p.indent, tab...)
	return func() {
		p.indent = p.indent[:len(p.indent)-len(tab)]
		if x := recover(); x != nil {
			panic(x) // skip print_trace
		}
		fmt.Printf("%5d: %s)\n", p.line, p.indent)
	}
}

// ----------------------------------------------------------------------------
// Package files
//
// Parse methods are annotated with matching Go productions as appropriate.
// The annotations are intended as guidelines only since a single Go grammar
// rule may be covered by multiple parse methods and vice versa.
//
// Excluding methods returning slices, parse methods named xOrNil may return
// nil; all others are expected to return a valid non-nil node.

// SourceFile = PackageClause ";" { ImportDecl ";" } { TopLevelDecl ";" } .
func (p *parser) fileOrNil() *File {
	if trace {
		defer p.trace("file")()
	}

	f := new(File)
	f.pos = p.pos()

	// PackageClause
	if !p.got(_Package) {
		p.syntax_error("package statement must be first")
		return nil
	}
	f.PkgName = p.name()
	p.want(_Semi)

	// don't bother continuing if package clause has errors
	if p.first != nil {
		return nil
	}

	// { ImportDecl ";" }
	for p.got(_Import) {
		f.DeclList = p.appendGroup(f.DeclList, p.importDecl)
		p.want(_Semi)
	}

	// { TopLevelDecl ";" }
	for p.tok != _EOF {
		switch p.tok {
		case _Const:
			p.next()
			f.DeclList = p.appendGroup(f.DeclList, p.constDecl)

		case _Type:
			p.next()
			f.DeclList = p.appendGroup(f.DeclList, p.typeDecl)

		case _Var:
			p.next()
			f.DeclList = p.appendGroup(f.DeclList, p.varDecl)

		case _Func:
			p.next()
			if d := p.funcDeclOrNil(); d != nil {
				f.DeclList = append(f.DeclList, d)
			}

		default:
			if p.tok == _Lbrace && len(f.DeclList) > 0 && isEmptyFuncDecl(f.DeclList[len(f.DeclList)-1]) {
				// opening { of function declaration on next line
				p.syntax_error("unexpected semicolon or newline before {")
			} else {
				p.syntax_error("non-declaration statement outside function body")
			}
			p.advance(_Const, _Type, _Var, _Func)
			continue
		}

		// Reset p.pragma BEFORE advancing to the next token (consuming ';')
		// since comments before may set pragmas for the next function decl.
		p.pragma = 0

		if p.tok != _EOF && !p.got(_Semi) {
			p.syntax_error("after top level declaration")
			p.advance(_Const, _Type, _Var, _Func)
		}
	}
	// p.tok == _EOF

	f.Lines = p.source.line

	return f
}

func isEmptyFuncDecl(dcl Decl) bool {
	f, ok := dcl.(*FuncDecl)
	return ok && f.Body == nil
}

// ----------------------------------------------------------------------------
// Declarations

// appendGroup(f) = f | "(" { f ";" } ")" .
func (p *parser) appendGroup(list []Decl, f func(*Group) Decl) []Decl {
	if p.got(_Lparen) {
		g := new(Group)
		for p.tok != _EOF && p.tok != _Rparen {
			list = append(list, f(g))
			if !p.osemi(_Rparen) {
				break
			}
		}
		p.want(_Rparen)
	} else {
		list = append(list, f(nil))
	}

	if debug {
		for _, d := range list {
			if d == nil {
				panic("nil list entry")
			}
		}
	}

	return list
}

// ImportSpec = [ "." | PackageName ] ImportPath .
// ImportPath = string_lit .
func (p *parser) importDecl(group *Group) Decl {
	if trace {
		defer p.trace("importDecl")()
	}

	d := new(ImportDecl)
	d.pos = p.pos()

	switch p.tok {
	case _Name:
		d.LocalPkgName = p.name()
	case _Dot:
		d.LocalPkgName = p.newName(".")
		p.next()
	}
	d.Path = p.oliteral()
	if d.Path == nil {
		p.syntax_error("missing import path")
		p.advance(_Semi, _Rparen)
		return nil
	}
	d.Group = group

	return d
}

// ConstSpec = IdentifierList [ [ Type ] "=" ExpressionList ] .
func (p *parser) constDecl(group *Group) Decl {
	if trace {
		defer p.trace("constDecl")()
	}

	d := new(ConstDecl)
	d.pos = p.pos()

	d.NameList = p.nameList(p.name())
	if p.tok != _EOF && p.tok != _Semi && p.tok != _Rparen {
		d.Type = p.typeOrNil()
		if p.got(_Assign) {
			d.Values = p.exprList()
		}
	}
	d.Group = group

	return d
}

// TypeSpec = identifier [ "=" ] Type .
func (p *parser) typeDecl(group *Group) Decl {
	if trace {
		defer p.trace("typeDecl")()
	}

	d := new(TypeDecl)
	d.pos = p.pos()

	d.Name = p.name()
	d.Alias = p.got(_Assign)
	d.Type = p.typeOrNil()
	if d.Type == nil {
		d.Type = p.bad()
		p.syntax_error("in type declaration")
		p.advance(_Semi, _Rparen)
	}
	d.Group = group
	d.Pragma = p.pragma

	return d
}

// VarSpec = IdentifierList ( Type [ "=" ExpressionList ] | "=" ExpressionList ) .
func (p *parser) varDecl(group *Group) Decl {
	if trace {
		defer p.trace("varDecl")()
	}

	d := new(VarDecl)
	d.pos = p.pos()

	d.NameList = p.nameList(p.name())
	if p.got(_Assign) {
		d.Values = p.exprList()
	} else {
		d.Type = p.type_()
		if p.got(_Assign) {
			d.Values = p.exprList()
		}
	}
	d.Group = group

	return d
}

// FunctionDecl = "func" FunctionName ( Function | Signature ) .
// FunctionName = identifier .
// Function     = Signature FunctionBody .
// MethodDecl   = "func" Receiver MethodName ( Function | Signature ) .
// Receiver     = Parameters .
func (p *parser) funcDeclOrNil() *FuncDecl {
	if trace {
		defer p.trace("funcDecl")()
	}

	f := new(FuncDecl)
	f.pos = p.pos()

	if p.tok == _Lparen {
		rcvr := p.paramList()
		switch len(rcvr) {
		case 0:
			p.error("method has no receiver")
		default:
			p.error("method has multiple receivers")
			fallthrough
		case 1:
			f.Recv = rcvr[0]
		}
	}

	if p.tok != _Name {
		p.syntax_error("expecting name or (")
		p.advance(_Lbrace, _Semi)
		return nil
	}

	// TODO(gri) check for regular functions only
	// if name.Sym.Name == "init" {
	// 	name = renameinit()
	// 	if params != nil || result != nil {
	// 		p.error("func init must have no arguments and no return values")
	// 	}
	// }

	// if localpkg.Name == "main" && name.Name == "main" {
	// 	if params != nil || result != nil {
	// 		p.error("func main must have no arguments and no return values")
	// 	}
	// }

	f.Name = p.name()
	f.Type = p.funcType()
	if p.tok == _Lbrace {
		f.Body = p.blockStmt("")
		if p.mode&CheckBranches != 0 {
			checkBranches(f.Body, p.errh)
		}
	}

	f.Pragma = p.pragma

	// TODO(gri) deal with function properties
	// if noescape && body != nil {
	// 	p.error("can only use //go:noescape with external func implementations")
	// }

	return f
}

// ----------------------------------------------------------------------------
// Expressions

func (p *parser) expr() Expr {
	if trace {
		defer p.trace("expr")()
	}

	return p.binaryExpr(0)
}

// Expression = UnaryExpr | Expression binary_op Expression .
func (p *parser) binaryExpr(prec int) Expr {
	// don't trace binaryExpr - only leads to overly nested trace output

	x := p.unaryExpr()
	for (p.tok == _Operator || p.tok == _Star) && p.prec > prec {
		t := new(Operation)
		t.pos = p.pos()
		t.Op = p.op
		t.X = x
		tprec := p.prec
		p.next()
		t.Y = p.binaryExpr(tprec)
		x = t
	}
	return x
}

// UnaryExpr = PrimaryExpr | unary_op UnaryExpr .
func (p *parser) unaryExpr() Expr {
	if trace {
		defer p.trace("unaryExpr")()
	}

	switch p.tok {
	case _Operator, _Star:
		switch p.op {
		case Mul, Add, Sub, Not, Xor:
			x := new(Operation)
			x.pos = p.pos()
			x.Op = p.op
			p.next()
			x.X = p.unaryExpr()
			return x

		case And:
			x := new(Operation)
			x.pos = p.pos()
			x.Op = And
			p.next()
			// unaryExpr may have returned a parenthesized composite literal
			// (see comment in operand) - remove parentheses if any
			x.X = unparen(p.unaryExpr())
			return x
		}

	case _Arrow:
		// receive op (<-x) or receive-only channel (<-chan E)
		pos := p.pos()
		p.next()

		// If the next token is _Chan we still don't know if it is
		// a channel (<-chan int) or a receive op (<-chan int(ch)).
		// We only know once we have found the end of the unaryExpr.

		x := p.unaryExpr()

		// There are two cases:
		//
		//   <-chan...  => <-x is a channel type
		//   <-x        => <-x is a receive operation
		//
		// In the first case, <- must be re-associated with
		// the channel type parsed already:
		//
		//   <-(chan E)   =>  (<-chan E)
		//   <-(chan<-E)  =>  (<-chan (<-E))

		if _, ok := x.(*ChanType); ok {
			// x is a channel type => re-associate <-
			dir := SendOnly
			t := x
			for dir == SendOnly {
				c, ok := t.(*ChanType)
				if !ok {
					break
				}
				dir = c.Dir
				if dir == RecvOnly {
					// t is type <-chan E but <-<-chan E is not permitted
					// (report same error as for "type _ <-<-chan E")
					p.syntax_error("unexpected <-, expecting chan")
					// already progressed, no need to advance
				}
				c.Dir = RecvOnly
				t = c.Elem
			}
			if dir == SendOnly {
				// channel dir is <- but channel element E is not a channel
				// (report same error as for "type _ <-chan<-E")
				p.syntax_error(fmt.Sprintf("unexpected %s, expecting chan", String(t)))
				// already progressed, no need to advance
			}
			return x
		}

		// x is not a channel type => we have a receive op
		o := new(Operation)
		o.pos = pos
		o.Op = Recv
		o.X = x
		return o
	}

	// TODO(mdempsky): We need parens here so we can report an
	// error for "(x) := true". It should be possible to detect
	// and reject that more efficiently though.
	return p.pexpr(true)
}

// callStmt parses call-like statements that can be preceded by 'defer' and 'go'.
func (p *parser) callStmt() *CallStmt {
	if trace {
		defer p.trace("callStmt")()
	}

	s := new(CallStmt)
	s.pos = p.pos()
	s.Tok = p.tok // _Defer or _Go
	p.next()

	x := p.pexpr(p.tok == _Lparen) // keep_parens so we can report error below
	if t := unparen(x); t != x {
		p.error(fmt.Sprintf("expression in %s must not be parenthesized", s.Tok))
		// already progressed, no need to advance
		x = t
	}

	cx, ok := x.(*CallExpr)
	if !ok {
		p.error(fmt.Sprintf("expression in %s must be function call", s.Tok))
		// already progressed, no need to advance
		cx = new(CallExpr)
		cx.pos = x.Pos()
		cx.Fun = p.bad()
	}

	s.Call = cx
	return s
}

// Operand     = Literal | OperandName | MethodExpr | "(" Expression ")" .
// Literal     = BasicLit | CompositeLit | FunctionLit .
// BasicLit    = int_lit | float_lit | imaginary_lit | rune_lit | string_lit .
// OperandName = identifier | QualifiedIdent.
func (p *parser) operand(keep_parens bool) Expr {
	if trace {
		defer p.trace("operand " + p.tok.String())()
	}

	switch p.tok {
	case _Name:
		return p.name()

	case _Literal:
		return p.oliteral()

	case _Lparen:
		pos := p.pos()
		p.next()
		p.xnest++
		x := p.expr()
		p.xnest--
		p.want(_Rparen)

		// Optimization: Record presence of ()'s only where needed
		// for error reporting. Don't bother in other cases; it is
		// just a waste of memory and time.

		// Parentheses are not permitted on lhs of := .
		// switch x.Op {
		// case ONAME, ONONAME, OPACK, OTYPE, OLITERAL, OTYPESW:
		// 	keep_parens = true
		// }

		// Parentheses are not permitted around T in a composite
		// literal T{}. If the next token is a {, assume x is a
		// composite literal type T (it may not be, { could be
		// the opening brace of a block, but we don't know yet).
		if p.tok == _Lbrace {
			keep_parens = true
		}

		// Parentheses are also not permitted around the expression
		// in a go/defer statement. In that case, operand is called
		// with keep_parens set.
		if keep_parens {
			px := new(ParenExpr)
			px.pos = pos
			px.X = x
			x = px
		}
		return x

	case _Func:
		pos := p.pos()
		p.next()
		t := p.funcType()
		if p.tok == _Lbrace {
			p.xnest++

			f := new(FuncLit)
			f.pos = pos
			f.Type = t
			f.Body = p.blockStmt("")
			if p.mode&CheckBranches != 0 {
				checkBranches(f.Body, p.errh)
			}

			p.xnest--
			return f
		}
		return t

	case _Lbrack, _Chan, _Map, _Struct, _Interface:
		return p.type_() // othertype

	default:
		x := p.bad()
		p.syntax_error("expecting expression")
		p.advance()
		return x
	}

	// Syntactically, composite literals are operands. Because a complit
	// type may be a qualified identifier which is handled by pexpr
	// (together with selector expressions), complits are parsed there
	// as well (operand is only called from pexpr).
}

// PrimaryExpr =
// 	Operand |
// 	Conversion |
// 	PrimaryExpr Selector |
// 	PrimaryExpr Index |
// 	PrimaryExpr Slice |
// 	PrimaryExpr TypeAssertion |
// 	PrimaryExpr Arguments .
//
// Selector       = "." identifier .
// Index          = "[" Expression "]" .
// Slice          = "[" ( [ Expression ] ":" [ Expression ] ) |
//                      ( [ Expression ] ":" Expression ":" Expression )
//                  "]" .
// TypeAssertion  = "." "(" Type ")" .
// Arguments      = "(" [ ( ExpressionList | Type [ "," ExpressionList ] ) [ "..." ] [ "," ] ] ")" .
func (p *parser) pexpr(keep_parens bool) Expr {
	if trace {
		defer p.trace("pexpr")()
	}

	x := p.operand(keep_parens)

loop:
	for {
		pos := p.pos()
		switch p.tok {
		case _Dot:
			p.next()
			switch p.tok {
			case _Name:
				// pexpr '.' sym
				t := new(SelectorExpr)
				t.pos = pos
				t.X = x
				t.Sel = p.name()
				x = t

			case _Lparen:
				p.next()
				if p.got(_Type) {
					t := new(TypeSwitchGuard)
					t.pos = pos
					t.X = x
					x = t
				} else {
					t := new(AssertExpr)
					t.pos = pos
					t.X = x
					t.Type = p.expr()
					x = t
				}
				p.want(_Rparen)

			default:
				p.syntax_error("expecting name or (")
				p.advance(_Semi, _Rparen)
			}

		case _Lbrack:
			p.next()
			p.xnest++

			var i Expr
			if p.tok != _Colon {
				i = p.expr()
				if p.got(_Rbrack) {
					// x[i]
					t := new(IndexExpr)
					t.pos = pos
					t.X = x
					t.Index = i
					x = t
					p.xnest--
					break
				}
			}

			// x[i:...
			t := new(SliceExpr)
			t.pos = pos
			t.X = x
			t.Index[0] = i
			p.want(_Colon)
			if p.tok != _Colon && p.tok != _Rbrack {
				// x[i:j...
				t.Index[1] = p.expr()
			}
			if p.got(_Colon) {
				t.Full = true
				// x[i:j:...]
				if t.Index[1] == nil {
					p.error("middle index required in 3-index slice")
				}
				if p.tok != _Rbrack {
					// x[i:j:k...
					t.Index[2] = p.expr()
				} else {
					p.error("final index required in 3-index slice")
				}
			}
			p.want(_Rbrack)

			x = t
			p.xnest--

		case _Lparen:
			x = p.call(x)

		case _Lbrace:
			// operand may have returned a parenthesized complit
			// type; accept it but complain if we have a complit
			t := unparen(x)
			// determine if '{' belongs to a composite literal or a block statement
			complit_ok := false
			switch t.(type) {
			case *Name, *SelectorExpr:
				if p.xnest >= 0 {
					// x is considered a composite literal type
					complit_ok = true
				}
			case *ArrayType, *SliceType, *StructType, *MapType:
				// x is a comptype
				complit_ok = true
			}
			if !complit_ok {
				break loop
			}
			if t != x {
				p.syntax_error("cannot parenthesize type in composite literal")
				// already progressed, no need to advance
			}
			n := p.complitexpr()
			n.Type = x
			x = n

		default:
			break loop
		}
	}

	return x
}

// Element = Expression | LiteralValue .
func (p *parser) bare_complitexpr() Expr {
	if trace {
		defer p.trace("bare_complitexpr")()
	}

	if p.tok == _Lbrace {
		// '{' start_complit braced_keyval_list '}'
		return p.complitexpr()
	}

	return p.expr()
}

// LiteralValue = "{" [ ElementList [ "," ] ] "}" .
func (p *parser) complitexpr() *CompositeLit {
	if trace {
		defer p.trace("complitexpr")()
	}

	x := new(CompositeLit)
	x.pos = p.pos()

	p.want(_Lbrace)
	p.xnest++

	for p.tok != _EOF && p.tok != _Rbrace {
		// value
		e := p.bare_complitexpr()
		if p.tok == _Colon {
			// key ':' value
			l := new(KeyValueExpr)
			l.pos = p.pos()
			p.next()
			l.Key = e
			l.Value = p.bare_complitexpr()
			e = l
			x.NKeys++
		}
		x.ElemList = append(x.ElemList, e)
		if !p.ocomma(_Rbrace) {
			break
		}
	}

	x.Rbrace = p.pos()
	p.xnest--
	p.want(_Rbrace)

	return x
}

// ----------------------------------------------------------------------------
// Types

func (p *parser) type_() Expr {
	if trace {
		defer p.trace("type_")()
	}

	typ := p.typeOrNil()
	if typ == nil {
		typ = p.bad()
		p.syntax_error("expecting type")
		p.advance()
	}

	return typ
}

func newIndirect(pos src.Pos, typ Expr) Expr {
	o := new(Operation)
	o.pos = pos
	o.Op = Mul
	o.X = typ
	return o
}

// typeOrNil is like type_ but it returns nil if there was no type
// instead of reporting an error.
//
// Type     = TypeName | TypeLit | "(" Type ")" .
// TypeName = identifier | QualifiedIdent .
// TypeLit  = ArrayType | StructType | PointerType | FunctionType | InterfaceType |
// 	      SliceType | MapType | Channel_Type .
func (p *parser) typeOrNil() Expr {
	if trace {
		defer p.trace("typeOrNil")()
	}

	pos := p.pos()
	switch p.tok {
	case _Star:
		// ptrtype
		p.next()
		return newIndirect(pos, p.type_())

	case _Arrow:
		// recvchantype
		p.next()
		p.want(_Chan)
		t := new(ChanType)
		t.pos = pos
		t.Dir = RecvOnly
		t.Elem = p.chanElem()
		return t

	case _Func:
		// fntype
		p.next()
		return p.funcType()

	case _Lbrack:
		// '[' oexpr ']' ntype
		// '[' _DotDotDot ']' ntype
		p.next()
		p.xnest++
		if p.got(_Rbrack) {
			// []T
			p.xnest--
			t := new(SliceType)
			t.pos = pos
			t.Elem = p.type_()
			return t
		}

		// [n]T
		t := new(ArrayType)
		t.pos = pos
		if !p.got(_DotDotDot) {
			t.Len = p.expr()
		}
		p.want(_Rbrack)
		p.xnest--
		t.Elem = p.type_()
		return t

	case _Chan:
		// _Chan non_recvchantype
		// _Chan _Comm ntype
		p.next()
		t := new(ChanType)
		t.pos = pos
		if p.got(_Arrow) {
			t.Dir = SendOnly
		}
		t.Elem = p.chanElem()
		return t

	case _Map:
		// _Map '[' ntype ']' ntype
		p.next()
		p.want(_Lbrack)
		t := new(MapType)
		t.pos = pos
		t.Key = p.type_()
		p.want(_Rbrack)
		t.Value = p.type_()
		return t

	case _Struct:
		return p.structType()

	case _Interface:
		return p.interfaceType()

	case _Name:
		return p.dotname(p.name())

	case _Lparen:
		p.next()
		t := p.type_()
		p.want(_Rparen)
		return t
	}

	return nil
}

func (p *parser) funcType() *FuncType {
	if trace {
		defer p.trace("funcType")()
	}

	typ := new(FuncType)
	typ.pos = p.pos()
	typ.ParamList = p.paramList()
	typ.ResultList = p.funcResult()

	return typ
}

func (p *parser) chanElem() Expr {
	if trace {
		defer p.trace("chanElem")()
	}

	typ := p.typeOrNil()
	if typ == nil {
		typ = p.bad()
		p.syntax_error("missing channel element type")
		// assume element type is simply absent - don't advance
	}

	return typ
}

func (p *parser) dotname(name *Name) Expr {
	if trace {
		defer p.trace("dotname")()
	}

	if p.tok == _Dot {
		s := new(SelectorExpr)
		s.pos = p.pos()
		p.next()
		s.X = name
		s.Sel = p.name()
		return s
	}
	return name
}

// StructType = "struct" "{" { FieldDecl ";" } "}" .
func (p *parser) structType() *StructType {
	if trace {
		defer p.trace("structType")()
	}

	typ := new(StructType)
	typ.pos = p.pos()

	p.want(_Struct)
	p.want(_Lbrace)
	for p.tok != _EOF && p.tok != _Rbrace {
		p.fieldDecl(typ)
		if !p.osemi(_Rbrace) {
			break
		}
	}
	p.want(_Rbrace)

	return typ
}

// InterfaceType = "interface" "{" { MethodSpec ";" } "}" .
func (p *parser) interfaceType() *InterfaceType {
	if trace {
		defer p.trace("interfaceType")()
	}

	typ := new(InterfaceType)
	typ.pos = p.pos()

	p.want(_Interface)
	p.want(_Lbrace)
	for p.tok != _EOF && p.tok != _Rbrace {
		if m := p.methodDecl(); m != nil {
			typ.MethodList = append(typ.MethodList, m)
		}
		if !p.osemi(_Rbrace) {
			break
		}
	}
	p.want(_Rbrace)

	return typ
}

// FunctionBody = Block .
func (p *parser) funcBody() []Stmt {
	if trace {
		defer p.trace("funcBody")()
	}

	p.fnest++
	body := p.stmtList()
	p.fnest--

	if body == nil {
		body = []Stmt{new(EmptyStmt)}
	}
	return body
}

// Result = Parameters | Type .
func (p *parser) funcResult() []*Field {
	if trace {
		defer p.trace("funcResult")()
	}

	if p.tok == _Lparen {
		return p.paramList()
	}

	pos := p.pos()
	if typ := p.typeOrNil(); typ != nil {
		f := new(Field)
		f.pos = pos
		f.Type = typ
		return []*Field{f}
	}

	return nil
}

func (p *parser) addField(styp *StructType, pos src.Pos, name *Name, typ Expr, tag *BasicLit) {
	if tag != nil {
		for i := len(styp.FieldList) - len(styp.TagList); i > 0; i-- {
			styp.TagList = append(styp.TagList, nil)
		}
		styp.TagList = append(styp.TagList, tag)
	}

	f := new(Field)
	f.pos = pos
	f.Name = name
	f.Type = typ
	styp.FieldList = append(styp.FieldList, f)

	if debug && tag != nil && len(styp.FieldList) != len(styp.TagList) {
		panic("inconsistent struct field list")
	}
}

// FieldDecl      = (IdentifierList Type | AnonymousField) [ Tag ] .
// AnonymousField = [ "*" ] TypeName .
// Tag            = string_lit .
func (p *parser) fieldDecl(styp *StructType) {
	if trace {
		defer p.trace("fieldDecl")()
	}

	pos := p.pos()
	switch p.tok {
	case _Name:
		name := p.name()
		if p.tok == _Dot || p.tok == _Literal || p.tok == _Semi || p.tok == _Rbrace {
			// embed oliteral
			typ := p.qualifiedName(name)
			tag := p.oliteral()
			p.addField(styp, pos, nil, typ, tag)
			return
		}

		// new_name_list ntype oliteral
		names := p.nameList(name)
		typ := p.type_()
		tag := p.oliteral()

		for _, name := range names {
			p.addField(styp, name.Pos(), name, typ, tag)
		}

	case _Lparen:
		p.next()
		if p.tok == _Star {
			// '(' '*' embed ')' oliteral
			pos := p.pos()
			p.next()
			typ := newIndirect(pos, p.qualifiedName(nil))
			p.want(_Rparen)
			tag := p.oliteral()
			p.addField(styp, pos, nil, typ, tag)
			p.syntax_error("cannot parenthesize embedded type")

		} else {
			// '(' embed ')' oliteral
			typ := p.qualifiedName(nil)
			p.want(_Rparen)
			tag := p.oliteral()
			p.addField(styp, pos, nil, typ, tag)
			p.syntax_error("cannot parenthesize embedded type")
		}

	case _Star:
		p.next()
		if p.got(_Lparen) {
			// '*' '(' embed ')' oliteral
			typ := newIndirect(pos, p.qualifiedName(nil))
			p.want(_Rparen)
			tag := p.oliteral()
			p.addField(styp, pos, nil, typ, tag)
			p.syntax_error("cannot parenthesize embedded type")

		} else {
			// '*' embed oliteral
			typ := newIndirect(pos, p.qualifiedName(nil))
			tag := p.oliteral()
			p.addField(styp, pos, nil, typ, tag)
		}

	default:
		p.syntax_error("expecting field name or embedded type")
		p.advance(_Semi, _Rbrace)
	}
}

func (p *parser) oliteral() *BasicLit {
	if p.tok == _Literal {
		b := new(BasicLit)
		b.pos = p.pos()
		b.Value = p.lit
		b.Kind = p.kind
		p.next()
		return b
	}
	return nil
}

// MethodSpec        = MethodName Signature | InterfaceTypeName .
// MethodName        = identifier .
// InterfaceTypeName = TypeName .
func (p *parser) methodDecl() *Field {
	if trace {
		defer p.trace("methodDecl")()
	}

	switch p.tok {
	case _Name:
		name := p.name()

		// accept potential name list but complain
		hasNameList := false
		for p.got(_Comma) {
			p.name()
			hasNameList = true
		}
		if hasNameList {
			p.syntax_error("name list not allowed in interface type")
			// already progressed, no need to advance
		}

		f := new(Field)
		f.pos = name.Pos()
		if p.tok != _Lparen {
			// packname
			f.Type = p.qualifiedName(name)
			return f
		}

		f.Name = name
		f.Type = p.funcType()
		return f

	case _Lparen:
		p.syntax_error("cannot parenthesize embedded type")
		f := new(Field)
		f.pos = p.pos()
		p.next()
		f.Type = p.qualifiedName(nil)
		p.want(_Rparen)
		return f

	default:
		p.syntax_error("expecting method or interface name")
		p.advance(_Semi, _Rbrace)
		return nil
	}
}

// ParameterDecl = [ IdentifierList ] [ "..." ] Type .
func (p *parser) paramDeclOrNil() *Field {
	if trace {
		defer p.trace("paramDecl")()
	}

	f := new(Field)
	f.pos = p.pos()

	switch p.tok {
	case _Name:
		f.Name = p.name()
		switch p.tok {
		case _Name, _Star, _Arrow, _Func, _Lbrack, _Chan, _Map, _Struct, _Interface, _Lparen:
			// sym name_or_type
			f.Type = p.type_()

		case _DotDotDot:
			// sym dotdotdot
			f.Type = p.dotsType()

		case _Dot:
			// name_or_type
			// from dotname
			f.Type = p.dotname(f.Name)
			f.Name = nil
		}

	case _Arrow, _Star, _Func, _Lbrack, _Chan, _Map, _Struct, _Interface, _Lparen:
		// name_or_type
		f.Type = p.type_()

	case _DotDotDot:
		// dotdotdot
		f.Type = p.dotsType()

	default:
		p.syntax_error("expecting )")
		p.advance(_Comma, _Rparen)
		return nil
	}

	return f
}

// ...Type
func (p *parser) dotsType() *DotsType {
	if trace {
		defer p.trace("dotsType")()
	}

	t := new(DotsType)
	t.pos = p.pos()

	p.want(_DotDotDot)
	t.Elem = p.typeOrNil()
	if t.Elem == nil {
		t.Elem = p.bad()
		p.syntax_error("final argument in variadic function missing type")
	}

	return t
}

// Parameters    = "(" [ ParameterList [ "," ] ] ")" .
// ParameterList = ParameterDecl { "," ParameterDecl } .
func (p *parser) paramList() (list []*Field) {
	if trace {
		defer p.trace("paramList")()
	}

	pos := p.pos()
	p.want(_Lparen)

	var named int // number of parameters that have an explicit name and type
	for p.tok != _EOF && p.tok != _Rparen {
		if par := p.paramDeclOrNil(); par != nil {
			if debug && par.Name == nil && par.Type == nil {
				panic("parameter without name or type")
			}
			if par.Name != nil && par.Type != nil {
				named++
			}
			list = append(list, par)
		}
		if !p.ocomma(_Rparen) {
			break
		}
	}

	// distribute parameter types
	if named == 0 {
		// all unnamed => found names are named types
		for _, par := range list {
			if typ := par.Name; typ != nil {
				par.Type = typ
				par.Name = nil
			}
		}
	} else if named != len(list) {
		// some named => all must be named
		ok := true
		var typ Expr
		for i := len(list) - 1; i >= 0; i-- {
			if par := list[i]; par.Type != nil {
				typ = par.Type
				if par.Name == nil {
					ok = false
					n := p.newName("_")
					n.pos = typ.Pos() // correct position
					par.Name = n
				}
			} else if typ != nil {
				par.Type = typ
			} else {
				// par.Type == nil && typ == nil => we only have a par.Name
				ok = false
				t := p.bad()
				t.pos = par.Name.Pos() // correct position
				par.Type = t
			}
		}
		if !ok {
			p.syntax_error_at(pos, "mixed named and unnamed function parameters")
		}
	}

	p.want(_Rparen)
	return
}

func (p *parser) bad() *BadExpr {
	b := new(BadExpr)
	b.pos = p.pos()
	return b
}

// ----------------------------------------------------------------------------
// Statements

// We represent x++, x-- as assignments x += ImplicitOne, x -= ImplicitOne.
// ImplicitOne should not be used elsewhere.
var ImplicitOne = &BasicLit{Value: "1"}

// SimpleStmt = EmptyStmt | ExpressionStmt | SendStmt | IncDecStmt | Assignment | ShortVarDecl .
func (p *parser) simpleStmt(lhs Expr, rangeOk bool) SimpleStmt {
	if trace {
		defer p.trace("simpleStmt")()
	}

	if rangeOk && p.tok == _Range {
		// _Range expr
		if debug && lhs != nil {
			panic("invalid call of simpleStmt")
		}
		return p.newRangeClause(nil, false)
	}

	if lhs == nil {
		lhs = p.exprList()
	}

	if _, ok := lhs.(*ListExpr); !ok && p.tok != _Assign && p.tok != _Define {
		// expr
		pos := p.pos()
		switch p.tok {
		case _AssignOp:
			// lhs op= rhs
			op := p.op
			p.next()
			return p.newAssignStmt(pos, op, lhs, p.expr())

		case _IncOp:
			// lhs++ or lhs--
			op := p.op
			p.next()
			return p.newAssignStmt(pos, op, lhs, ImplicitOne)

		case _Arrow:
			// lhs <- rhs
			s := new(SendStmt)
			s.pos = pos
			p.next()
			s.Chan = lhs
			s.Value = p.expr()
			return s

		default:
			// expr
			s := new(ExprStmt)
			s.pos = lhs.Pos()
			s.X = lhs
			return s
		}
	}

	// expr_list
	pos := p.pos()
	switch p.tok {
	case _Assign:
		p.next()

		if rangeOk && p.tok == _Range {
			// expr_list '=' _Range expr
			return p.newRangeClause(lhs, false)
		}

		// expr_list '=' expr_list
		return p.newAssignStmt(pos, 0, lhs, p.exprList())

	case _Define:
		p.next()

		if rangeOk && p.tok == _Range {
			// expr_list ':=' range expr
			return p.newRangeClause(lhs, true)
		}

		// expr_list ':=' expr_list
		rhs := p.exprList()

		if x, ok := rhs.(*TypeSwitchGuard); ok {
			switch lhs := lhs.(type) {
			case *Name:
				x.Lhs = lhs
			case *ListExpr:
				p.error_at(lhs.Pos(), fmt.Sprintf("cannot assign 1 value to %d variables", len(lhs.ElemList)))
				// make the best of what we have
				if lhs, ok := lhs.ElemList[0].(*Name); ok {
					x.Lhs = lhs
				}
			default:
				p.error_at(lhs.Pos(), fmt.Sprintf("invalid variable name %s in type switch", String(lhs)))
			}
			s := new(ExprStmt)
			s.pos = x.Pos()
			s.X = x
			return s
		}

		as := p.newAssignStmt(pos, Def, lhs, rhs)
		return as

	default:
		p.syntax_error("expecting := or = or comma")
		p.advance(_Semi, _Rbrace)
		// make the best of what we have
		if x, ok := lhs.(*ListExpr); ok {
			lhs = x.ElemList[0]
		}
		s := new(ExprStmt)
		s.pos = lhs.Pos()
		s.X = lhs
		return s
	}
}

func (p *parser) newRangeClause(lhs Expr, def bool) *RangeClause {
	r := new(RangeClause)
	r.pos = p.pos()
	p.next() // consume _Range
	r.Lhs = lhs
	r.Def = def
	r.X = p.expr()
	return r
}

func (p *parser) newAssignStmt(pos src.Pos, op Operator, lhs, rhs Expr) *AssignStmt {
	a := new(AssignStmt)
	a.pos = pos
	a.Op = op
	a.Lhs = lhs
	a.Rhs = rhs
	return a
}

func (p *parser) labeledStmtOrNil(label *Name) Stmt {
	if trace {
		defer p.trace("labeledStmt")()
	}

	s := new(LabeledStmt)
	s.pos = p.pos()
	s.Label = label

	p.want(_Colon)

	if p.tok == _Rbrace {
		// We expect a statement (incl. an empty statement), which must be
		// terminated by a semicolon. Because semicolons may be omitted before
		// an _Rbrace, seeing an _Rbrace implies an empty statement.
		e := new(EmptyStmt)
		e.pos = p.pos()
		s.Stmt = e
		return s
	}

	s.Stmt = p.stmtOrNil()
	if s.Stmt != nil {
		return s
	}

	// report error at line of ':' token
	p.syntax_error_at(s.pos, "missing statement after label")
	// we are already at the end of the labeled statement - no need to advance
	return nil // avoids follow-on errors (see e.g., fixedbugs/bug274.go)
}

func (p *parser) blockStmt(context string) *BlockStmt {
	if trace {
		defer p.trace("blockStmt")()
	}

	s := new(BlockStmt)
	s.pos = p.pos()

	if !p.got(_Lbrace) {
		p.syntax_error("expecting { after " + context)
		p.advance(_Name, _Rbrace)
		// TODO(gri) may be better to return here than to continue (#19663)
	}

	s.List = p.stmtList()
	s.Rbrace = p.pos()
	p.want(_Rbrace)

	return s
}

func (p *parser) declStmt(f func(*Group) Decl) *DeclStmt {
	if trace {
		defer p.trace("declStmt")()
	}

	s := new(DeclStmt)
	s.pos = p.pos()

	p.next() // _Const, _Type, or _Var
	s.DeclList = p.appendGroup(nil, f)

	return s
}

func (p *parser) forStmt() Stmt {
	if trace {
		defer p.trace("forStmt")()
	}

	s := new(ForStmt)
	s.pos = p.pos()

	s.Init, s.Cond, s.Post = p.header(_For)
	s.Body = p.blockStmt("for clause")

	return s
}

// TODO(gri) This function is now so heavily influenced by the keyword that
//           it may not make sense anymore to combine all three cases. It
//           may be simpler to just split it up for each statement kind.
func (p *parser) header(keyword token) (init SimpleStmt, cond Expr, post SimpleStmt) {
	p.want(keyword)

	if p.tok == _Lbrace {
		if keyword == _If {
			p.syntax_error("missing condition in if statement")
		}
		return
	}
	// p.tok != _Lbrace

	outer := p.xnest
	p.xnest = -1

	if p.tok != _Semi {
		// accept potential varDecl but complain
		if p.got(_Var) {
			p.syntax_error(fmt.Sprintf("var declaration not allowed in %s initializer", keyword.String()))
		}
		init = p.simpleStmt(nil, keyword == _For)
		// If we have a range clause, we are done (can only happen for keyword == _For).
		if _, ok := init.(*RangeClause); ok {
			p.xnest = outer
			return
		}
	}

	var condStmt SimpleStmt
	var semi struct {
		pos src.Pos
		lit string // valid if pos.IsKnown()
	}
	if p.tok == _Semi {
		semi.pos = p.pos()
		semi.lit = p.lit
		p.next()
		if keyword == _For {
			if p.tok != _Semi {
				if p.tok == _Lbrace {
					p.syntax_error("expecting for loop condition")
					goto done
				}
				condStmt = p.simpleStmt(nil, false)
			}
			p.want(_Semi)
			if p.tok != _Lbrace {
				post = p.simpleStmt(nil, false)
				if a, _ := post.(*AssignStmt); a != nil && a.Op == Def {
					p.syntax_error_at(a.Pos(), "cannot declare in post statement of for loop")
				}
			}
		} else if p.tok != _Lbrace {
			condStmt = p.simpleStmt(nil, false)
		}
	} else {
		condStmt = init
		init = nil
	}

done:
	// unpack condStmt
	switch s := condStmt.(type) {
	case nil:
		if keyword == _If && semi.pos.IsKnown() {
			if semi.lit != "semicolon" {
				p.syntax_error_at(semi.pos, fmt.Sprintf("unexpected %s, expecting { after if clause", semi.lit))
			} else {
				p.syntax_error_at(semi.pos, "missing condition in if statement")
			}
		}
	case *ExprStmt:
		cond = s.X
	default:
		p.syntax_error(fmt.Sprintf("%s used as value", String(s)))
	}

	p.xnest = outer
	return
}

func (p *parser) ifStmt() *IfStmt {
	if trace {
		defer p.trace("ifStmt")()
	}

	s := new(IfStmt)
	s.pos = p.pos()

	s.Init, s.Cond, _ = p.header(_If)
	s.Then = p.blockStmt("if clause")

	if p.got(_Else) {
		switch p.tok {
		case _If:
			s.Else = p.ifStmt()
		case _Lbrace:
			s.Else = p.blockStmt("")
		default:
			p.syntax_error("else must be followed by if or statement block")
			p.advance(_Name, _Rbrace)
		}
	}

	return s
}

func (p *parser) switchStmt() *SwitchStmt {
	if trace {
		defer p.trace("switchStmt")()
	}

	s := new(SwitchStmt)
	s.pos = p.pos()

	s.Init, s.Tag, _ = p.header(_Switch)

	if !p.got(_Lbrace) {
		p.syntax_error("missing { after switch clause")
		p.advance(_Case, _Default, _Rbrace)
	}
	for p.tok != _EOF && p.tok != _Rbrace {
		s.Body = append(s.Body, p.caseClause())
	}
	s.Rbrace = p.pos()
	p.want(_Rbrace)

	return s
}

func (p *parser) selectStmt() *SelectStmt {
	if trace {
		defer p.trace("selectStmt")()
	}

	s := new(SelectStmt)
	s.pos = p.pos()

	p.want(_Select)
	if !p.got(_Lbrace) {
		p.syntax_error("missing { after select clause")
		p.advance(_Case, _Default, _Rbrace)
	}
	for p.tok != _EOF && p.tok != _Rbrace {
		s.Body = append(s.Body, p.commClause())
	}
	s.Rbrace = p.pos()
	p.want(_Rbrace)

	return s
}

func (p *parser) caseClause() *CaseClause {
	if trace {
		defer p.trace("caseClause")()
	}

	c := new(CaseClause)
	c.pos = p.pos()

	switch p.tok {
	case _Case:
		p.next()
		c.Cases = p.exprList()

	case _Default:
		p.next()

	default:
		p.syntax_error("expecting case or default or }")
		p.advance(_Colon, _Case, _Default, _Rbrace)
	}

	c.Colon = p.pos()
	p.want(_Colon)
	c.Body = p.stmtList()

	return c
}

func (p *parser) commClause() *CommClause {
	if trace {
		defer p.trace("commClause")()
	}

	c := new(CommClause)
	c.pos = p.pos()

	switch p.tok {
	case _Case:
		p.next()
		c.Comm = p.simpleStmt(nil, false)

		// The syntax restricts the possible simple statements here to:
		//
		//     lhs <- x (send statement)
		//     <-x
		//     lhs = <-x
		//     lhs := <-x
		//
		// All these (and more) are recognized by simpleStmt and invalid
		// syntax trees are flagged later, during type checking.
		// TODO(gri) eventually may want to restrict valid syntax trees
		// here.

	case _Default:
		p.next()

	default:
		p.syntax_error("expecting case or default or }")
		p.advance(_Colon, _Case, _Default, _Rbrace)
	}

	c.Colon = p.pos()
	p.want(_Colon)
	c.Body = p.stmtList()

	return c
}

// Statement =
// 	Declaration | LabeledStmt | SimpleStmt |
// 	GoStmt | ReturnStmt | BreakStmt | ContinueStmt | GotoStmt |
// 	FallthroughStmt | Block | IfStmt | SwitchStmt | SelectStmt | ForStmt |
// 	DeferStmt .
func (p *parser) stmtOrNil() Stmt {
	if trace {
		defer p.trace("stmt " + p.tok.String())()
	}

	// Most statements (assignments) start with an identifier;
	// look for it first before doing anything more expensive.
	if p.tok == _Name {
		lhs := p.exprList()
		if label, ok := lhs.(*Name); ok && p.tok == _Colon {
			return p.labeledStmtOrNil(label)
		}
		return p.simpleStmt(lhs, false)
	}

	switch p.tok {
	case _Lbrace:
		return p.blockStmt("")

	case _Var:
		return p.declStmt(p.varDecl)

	case _Const:
		return p.declStmt(p.constDecl)

	case _Type:
		return p.declStmt(p.typeDecl)

	case _Operator, _Star:
		switch p.op {
		case Add, Sub, Mul, And, Xor, Not:
			return p.simpleStmt(nil, false) // unary operators
		}

	case _Literal, _Func, _Lparen, // operands
		_Lbrack, _Struct, _Map, _Chan, _Interface, // composite types
		_Arrow: // receive operator
		return p.simpleStmt(nil, false)

	case _For:
		return p.forStmt()

	case _Switch:
		return p.switchStmt()

	case _Select:
		return p.selectStmt()

	case _If:
		return p.ifStmt()

	case _Fallthrough:
		s := new(BranchStmt)
		s.pos = p.pos()
		p.next()
		s.Tok = _Fallthrough
		return s

	case _Break, _Continue:
		s := new(BranchStmt)
		s.pos = p.pos()
		s.Tok = p.tok
		p.next()
		if p.tok == _Name {
			s.Label = p.name()
		}
		return s

	case _Go, _Defer:
		return p.callStmt()

	case _Goto:
		s := new(BranchStmt)
		s.pos = p.pos()
		s.Tok = _Goto
		p.next()
		s.Label = p.name()
		return s

	case _Return:
		s := new(ReturnStmt)
		s.pos = p.pos()
		p.next()
		if p.tok != _Semi && p.tok != _Rbrace {
			s.Results = p.exprList()
		}
		return s

	case _Semi:
		s := new(EmptyStmt)
		s.pos = p.pos()
		return s
	}

	return nil
}

// StatementList = { Statement ";" } .
func (p *parser) stmtList() (l []Stmt) {
	if trace {
		defer p.trace("stmtList")()
	}

	for p.tok != _EOF && p.tok != _Rbrace && p.tok != _Case && p.tok != _Default {
		s := p.stmtOrNil()
		if s == nil {
			break
		}
		l = append(l, s)
		// customized version of osemi:
		// ';' is optional before a closing ')' or '}'
		if p.tok == _Rparen || p.tok == _Rbrace {
			continue
		}
		if !p.got(_Semi) {
			p.syntax_error("at end of statement")
			p.advance(_Semi, _Rbrace)
		}
	}
	return
}

// Arguments = "(" [ ( ExpressionList | Type [ "," ExpressionList ] ) [ "..." ] [ "," ] ] ")" .
func (p *parser) call(fun Expr) *CallExpr {
	if trace {
		defer p.trace("call")()
	}

	// call or conversion
	// convtype '(' expr ocomma ')'
	c := new(CallExpr)
	c.pos = p.pos()
	c.Fun = fun

	p.want(_Lparen)
	p.xnest++

	for p.tok != _EOF && p.tok != _Rparen {
		c.ArgList = append(c.ArgList, p.expr())
		c.HasDots = p.got(_DotDotDot)
		if !p.ocomma(_Rparen) || c.HasDots {
			break
		}
	}

	p.xnest--
	p.want(_Rparen)

	return c
}

// ----------------------------------------------------------------------------
// Common productions

func (p *parser) newName(value string) *Name {
	n := new(Name)
	n.pos = p.pos()
	n.Value = value
	return n
}

func (p *parser) name() *Name {
	// no tracing to avoid overly verbose output

	if p.tok == _Name {
		n := p.newName(p.lit)
		p.next()
		return n
	}

	n := p.newName("_")
	p.syntax_error("expecting name")
	p.advance()
	return n
}

// IdentifierList = identifier { "," identifier } .
// The first name must be provided.
func (p *parser) nameList(first *Name) []*Name {
	if trace {
		defer p.trace("nameList")()
	}

	if debug && first == nil {
		panic("first name not provided")
	}

	l := []*Name{first}
	for p.got(_Comma) {
		l = append(l, p.name())
	}

	return l
}

// The first name may be provided, or nil.
func (p *parser) qualifiedName(name *Name) Expr {
	if trace {
		defer p.trace("qualifiedName")()
	}

	switch {
	case name != nil:
		// name is provided
	case p.tok == _Name:
		name = p.name()
	default:
		name = p.newName("_")
		p.syntax_error("expecting name")
		p.advance(_Dot, _Semi, _Rbrace)
	}

	return p.dotname(name)
}

// ExpressionList = Expression { "," Expression } .
func (p *parser) exprList() Expr {
	if trace {
		defer p.trace("exprList")()
	}

	x := p.expr()
	if p.got(_Comma) {
		list := []Expr{x, p.expr()}
		for p.got(_Comma) {
			list = append(list, p.expr())
		}
		t := new(ListExpr)
		t.pos = x.Pos()
		t.ElemList = list
		x = t
	}
	return x
}

// osemi parses an optional semicolon.
func (p *parser) osemi(follow token) bool {
	switch p.tok {
	case _Semi:
		p.next()
		return true

	case _Rparen, _Rbrace:
		// semicolon is optional before ) or }
		return true
	}

	p.syntax_error("expecting semicolon, newline, or " + tokstring(follow))
	p.advance(follow)
	return false
}

// ocomma parses an optional comma.
func (p *parser) ocomma(follow token) bool {
	switch p.tok {
	case _Comma:
		p.next()
		return true

	case _Rparen, _Rbrace:
		// comma is optional before ) or }
		return true
	}

	p.syntax_error("expecting comma or " + tokstring(follow))
	p.advance(follow)
	return false
}

// unparen removes all parentheses around an expression.
func unparen(x Expr) Expr {
	for {
		p, ok := x.(*ParenExpr)
		if !ok {
			break
		}
		x = p.X
	}
	return x
}
