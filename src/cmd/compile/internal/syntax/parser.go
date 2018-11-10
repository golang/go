// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"fmt"
	"io"
	"strings"
)

const debug = false
const trace = false

// The old gc parser assigned line numbers very inconsistently depending
// on when it happened to construct AST nodes. To make transitioning to the
// new AST easier, we try to mimick the behavior as much as possible.
const gcCompat = true

type parser struct {
	scanner

	fnest  int    // function nesting level (for error handling)
	xnest  int    // expression nesting level (for complit ambiguity resolution)
	indent []byte // tracing support
}

func (p *parser) init(src io.Reader, errh ErrorHandler, pragh PragmaHandler) {
	p.scanner.init(src, errh, pragh)

	p.fnest = 0
	p.xnest = 0
	p.indent = nil
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

// syntax_error reports a syntax error at the current line.
func (p *parser) syntax_error(msg string) {
	p.syntax_error_at(p.pos, p.line, msg)
}

// Like syntax_error, but reports error at given line rather than current lexer line.
func (p *parser) syntax_error_at(pos, line int, msg string) {
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
		p.error_at(pos, line, "syntax error: "+msg)
		return
	}

	// determine token string
	var tok string
	switch p.tok {
	case _Name:
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

	p.error_at(pos, line, "syntax error: unexpected "+tok+msg)
}

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
	// TODO(gri) the args are constants - do as constant expressions?
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
		return "semicolon or newline"
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

// SourceFile = PackageClause ";" { ImportDecl ";" } { TopLevelDecl ";" } .
func (p *parser) file() *File {
	if trace {
		defer p.trace("file")()
	}

	f := new(File)
	f.init(p)

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
			f.DeclList = append(f.DeclList, p.funcDecl())

		default:
			if p.tok == _Lbrace && len(f.DeclList) > 0 && emptyFuncDecl(f.DeclList[len(f.DeclList)-1]) {
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

func emptyFuncDecl(dcl Decl) bool {
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
		return list
	}

	return append(list, f(nil))
}

func (p *parser) importDecl(group *Group) Decl {
	if trace {
		defer p.trace("importDecl")()
	}

	d := new(ImportDecl)
	d.init(p)

	switch p.tok {
	case _Name:
		d.LocalPkgName = p.name()
	case _Dot:
		n := new(Name)
		n.init(p)
		n.Value = "."
		d.LocalPkgName = n
		p.next()
	}
	if p.tok == _Literal && (gcCompat || p.kind == StringLit) {
		d.Path = p.oliteral()
	} else {
		p.syntax_error("missing import path; require quoted string")
		p.advance(_Semi, _Rparen)
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
	d.init(p)

	d.NameList = p.nameList(p.name())
	if p.tok != _EOF && p.tok != _Semi && p.tok != _Rparen {
		d.Type = p.tryType()
		if p.got(_Assign) {
			d.Values = p.exprList()
		}
	}
	d.Group = group

	return d
}

// TypeSpec = identifier Type .
func (p *parser) typeDecl(group *Group) Decl {
	if trace {
		defer p.trace("typeDecl")()
	}

	d := new(TypeDecl)
	d.init(p)

	d.Name = p.name()
	d.Type = p.tryType()
	if d.Type == nil {
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
	d.init(p)

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
	if gcCompat {
		d.init(p)
	}

	return d
}

// FunctionDecl = "func" FunctionName ( Function | Signature ) .
// FunctionName = identifier .
// Function     = Signature FunctionBody .
// MethodDecl   = "func" Receiver MethodName ( Function | Signature ) .
// Receiver     = Parameters .
func (p *parser) funcDecl() *FuncDecl {
	if trace {
		defer p.trace("funcDecl")()
	}

	f := new(FuncDecl)
	f.init(p)

	badRecv := false
	if p.tok == _Lparen {
		rcvr := p.paramList()
		switch len(rcvr) {
		case 0:
			p.error("method has no receiver")
			badRecv = true
		case 1:
			f.Recv = rcvr[0]
		default:
			p.error("method has multiple receivers")
			badRecv = true
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
	if gcCompat {
		f.node = f.Type.node
	}
	f.Body = p.funcBody()

	f.Pragma = p.pragma
	f.EndLine = uint32(p.line)

	// TODO(gri) deal with function properties
	// if noescape && body != nil {
	// 	p.error("can only use //go:noescape with external func implementations")
	// }

	if badRecv {
		return nil // TODO(gri) better solution
	}
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
		t.init(p)
		t.Op = p.op
		t.X = x
		tprec := p.prec
		p.next()
		t.Y = p.binaryExpr(tprec)
		if gcCompat {
			t.init(p)
		}
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
			x.init(p)
			x.Op = p.op
			p.next()
			x.X = p.unaryExpr()
			if gcCompat {
				x.init(p)
			}
			return x

		case And:
			p.next()
			x := new(Operation)
			x.init(p)
			x.Op = And
			// unaryExpr may have returned a parenthesized composite literal
			// (see comment in operand) - remove parentheses if any
			x.X = unparen(p.unaryExpr())
			return x
		}

	case _Arrow:
		// receive op (<-x) or receive-only channel (<-chan E)
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
		return &Operation{Op: Recv, X: x}
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
	s.init(p)
	s.Tok = p.tok
	p.next()

	x := p.pexpr(p.tok == _Lparen) // keep_parens so we can report error below
	switch x := x.(type) {
	case *CallExpr:
		s.Call = x
		if gcCompat {
			s.node = x.node
		}
	case *ParenExpr:
		p.error(fmt.Sprintf("expression in %s must not be parenthesized", s.Tok))
		// already progressed, no need to advance
	default:
		p.error(fmt.Sprintf("expression in %s must be function call", s.Tok))
		// already progressed, no need to advance
	}

	return s // TODO(gri) should we return nil in case of failure?
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
		p.next()
		p.xnest++
		x := p.expr() // expr_or_type
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
			x = &ParenExpr{X: x}
		}
		return x

	case _Func:
		p.next()
		t := p.funcType()
		if p.tok == _Lbrace {
			p.fnest++
			p.xnest++
			f := new(FuncLit)
			f.init(p)
			f.Type = t
			f.Body = p.funcBody()
			f.EndLine = uint32(p.line)
			p.xnest--
			p.fnest--
			return f
		}
		return t

	case _Lbrack, _Chan, _Map, _Struct, _Interface:
		return p.type_() // othertype

	case _Lbrace:
		// common case: p.header is missing simpleStmt before { in if, for, switch
		p.syntax_error("missing operand")
		// '{' will be consumed in pexpr - no need to consume it here
		return nil

	default:
		p.syntax_error("expecting expression")
		p.advance()
		return nil
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
		switch p.tok {
		case _Dot:
			p.next()
			switch p.tok {
			case _Name:
				// pexpr '.' sym
				t := new(SelectorExpr)
				t.init(p)
				t.X = x
				t.Sel = p.name()
				x = t

			case _Lparen:
				p.next()
				if p.got(_Type) {
					t := new(TypeSwitchGuard)
					t.init(p)
					t.X = x
					x = t
				} else {
					t := new(AssertExpr)
					t.init(p)
					t.X = x
					t.Type = p.expr()
					x = t
				}
				p.want(_Rparen)

			default:
				p.syntax_error("expecting name or (")
				p.advance(_Semi, _Rparen)
			}
			if gcCompat {
				x.init(p)
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
					t.init(p)
					t.X = x
					t.Index = i
					x = t
					p.xnest--
					break
				}
			}

			// x[i:...
			t := new(SliceExpr)
			t.init(p)
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
			// determine if '{' belongs to a complit or a compound_stmt
			complit_ok := false
			switch t.(type) {
			case *Name, *SelectorExpr:
				if p.xnest >= 0 {
					// x is considered a comptype
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
	x.init(p)

	p.want(_Lbrace)
	p.xnest++

	for p.tok != _EOF && p.tok != _Rbrace {
		// value
		e := p.bare_complitexpr()
		if p.got(_Colon) {
			// key ':' value
			l := new(KeyValueExpr)
			l.init(p)
			l.Key = e
			l.Value = p.bare_complitexpr()
			if gcCompat {
				l.init(p)
			}
			e = l
			x.NKeys++
		}
		x.ElemList = append(x.ElemList, e)
		if !p.ocomma(_Rbrace) {
			break
		}
	}

	x.EndLine = uint32(p.line)
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

	if typ := p.tryType(); typ != nil {
		return typ
	}

	p.syntax_error("")
	p.advance()
	return nil
}

func indirect(typ Expr) Expr {
	return &Operation{Op: Mul, X: typ}
}

// tryType is like type_ but it returns nil if there was no type
// instead of reporting an error.
//
// Type     = TypeName | TypeLit | "(" Type ")" .
// TypeName = identifier | QualifiedIdent .
// TypeLit  = ArrayType | StructType | PointerType | FunctionType | InterfaceType |
// 	      SliceType | MapType | Channel_Type .
func (p *parser) tryType() Expr {
	if trace {
		defer p.trace("tryType")()
	}

	switch p.tok {
	case _Star:
		// ptrtype
		p.next()
		return indirect(p.type_())

	case _Arrow:
		// recvchantype
		p.next()
		p.want(_Chan)
		t := new(ChanType)
		t.init(p)
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
			t.init(p)
			t.Elem = p.type_()
			return t
		}

		// [n]T
		t := new(ArrayType)
		t.init(p)
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
		t.init(p)
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
		t.init(p)
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
	typ.init(p)
	typ.ParamList = p.paramList()
	typ.ResultList = p.funcResult()
	if gcCompat {
		typ.init(p)
	}
	return typ
}

func (p *parser) chanElem() Expr {
	if trace {
		defer p.trace("chanElem")()
	}

	if typ := p.tryType(); typ != nil {
		return typ
	}

	p.syntax_error("missing channel element type")
	// assume element type is simply absent - don't advance
	return nil
}

func (p *parser) dotname(name *Name) Expr {
	if trace {
		defer p.trace("dotname")()
	}

	if p.got(_Dot) {
		s := new(SelectorExpr)
		s.init(p)
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
	typ.init(p)

	p.want(_Struct)
	p.want(_Lbrace)
	for p.tok != _EOF && p.tok != _Rbrace {
		p.fieldDecl(typ)
		if !p.osemi(_Rbrace) {
			break
		}
	}
	if gcCompat {
		typ.init(p)
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
	typ.init(p)

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
	if gcCompat {
		typ.init(p)
	}
	p.want(_Rbrace)

	return typ
}

// FunctionBody = Block .
func (p *parser) funcBody() []Stmt {
	if trace {
		defer p.trace("funcBody")()
	}

	if p.got(_Lbrace) {
		p.fnest++
		body := p.stmtList()
		p.fnest--
		p.want(_Rbrace)
		if body == nil {
			body = []Stmt{new(EmptyStmt)}
		}
		return body
	}

	return nil
}

// Result = Parameters | Type .
func (p *parser) funcResult() []*Field {
	if trace {
		defer p.trace("funcResult")()
	}

	if p.tok == _Lparen {
		return p.paramList()
	}

	if result := p.tryType(); result != nil {
		f := new(Field)
		f.init(p)
		f.Type = result
		return []*Field{f}
	}

	return nil
}

func (p *parser) addField(styp *StructType, name *Name, typ Expr, tag *BasicLit) {
	if tag != nil {
		for i := len(styp.FieldList) - len(styp.TagList); i > 0; i-- {
			styp.TagList = append(styp.TagList, nil)
		}
		styp.TagList = append(styp.TagList, tag)
	}

	f := new(Field)
	f.init(p)
	f.Name = name
	f.Type = typ
	styp.FieldList = append(styp.FieldList, f)

	if gcCompat && name != nil {
		f.node = name.node
	}

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

	var name *Name
	switch p.tok {
	case _Name:
		name = p.name()
		if p.tok == _Dot || p.tok == _Literal || p.tok == _Semi || p.tok == _Rbrace {
			// embed oliteral
			typ := p.qualifiedName(name)
			tag := p.oliteral()
			p.addField(styp, nil, typ, tag)
			return
		}

		// new_name_list ntype oliteral
		names := p.nameList(name)
		typ := p.type_()
		tag := p.oliteral()

		for _, name := range names {
			p.addField(styp, name, typ, tag)
		}

	case _Lparen:
		p.next()
		if p.tok == _Star {
			// '(' '*' embed ')' oliteral
			p.next()
			typ := indirect(p.qualifiedName(nil))
			p.want(_Rparen)
			tag := p.oliteral()
			p.addField(styp, nil, typ, tag)
			p.error("cannot parenthesize embedded type")

		} else {
			// '(' embed ')' oliteral
			typ := p.qualifiedName(nil)
			p.want(_Rparen)
			tag := p.oliteral()
			p.addField(styp, nil, typ, tag)
			p.error("cannot parenthesize embedded type")
		}

	case _Star:
		p.next()
		if p.got(_Lparen) {
			// '*' '(' embed ')' oliteral
			typ := indirect(p.qualifiedName(nil))
			p.want(_Rparen)
			tag := p.oliteral()
			p.addField(styp, nil, typ, tag)
			p.error("cannot parenthesize embedded type")

		} else {
			// '*' embed oliteral
			typ := indirect(p.qualifiedName(nil))
			tag := p.oliteral()
			p.addField(styp, nil, typ, tag)
		}

	default:
		p.syntax_error("expecting field name or embedded type")
		p.advance(_Semi, _Rbrace)
	}
}

func (p *parser) oliteral() *BasicLit {
	if p.tok == _Literal {
		b := new(BasicLit)
		b.init(p)
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
		f.init(p)
		if p.tok != _Lparen {
			// packname
			f.Type = p.qualifiedName(name)
			return f
		}

		f.Name = name
		f.Type = p.funcType()
		return f

	case _Lparen:
		p.next()
		f := new(Field)
		f.init(p)
		f.Type = p.qualifiedName(nil)
		p.want(_Rparen)
		p.error("cannot parenthesize embedded type")
		return f

	default:
		p.syntax_error("")
		p.advance(_Semi, _Rbrace)
		return nil
	}
}

// ParameterDecl = [ IdentifierList ] [ "..." ] Type .
func (p *parser) paramDecl() *Field {
	if trace {
		defer p.trace("paramDecl")()
	}

	f := new(Field)
	f.init(p)

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
	t.init(p)

	p.want(_DotDotDot)
	t.Elem = p.tryType()
	if t.Elem == nil {
		p.error("final argument in variadic function missing type")
	}

	return t
}

// Parameters    = "(" [ ParameterList [ "," ] ] ")" .
// ParameterList = ParameterDecl { "," ParameterDecl } .
func (p *parser) paramList() (list []*Field) {
	if trace {
		defer p.trace("paramList")()
	}

	p.want(_Lparen)

	var named int // number of parameters that have an explicit name and type
	for p.tok != _EOF && p.tok != _Rparen {
		if par := p.paramDecl(); par != nil {
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
		var typ Expr
		for i := len(list) - 1; i >= 0; i-- {
			if par := list[i]; par.Type != nil {
				typ = par.Type
				if par.Name == nil {
					typ = nil // error
				}
			} else {
				par.Type = typ
			}
			if typ == nil {
				p.syntax_error("mixed named and unnamed function parameters")
				break
			}
		}
	}

	p.want(_Rparen)
	return
}

// ----------------------------------------------------------------------------
// Statements

// We represent x++, x-- as assignments x += ImplicitOne, x -= ImplicitOne.
// ImplicitOne should not be used elsewhere.
var ImplicitOne = &BasicLit{Value: "1"}

// SimpleStmt = EmptyStmt | ExpressionStmt | SendStmt | IncDecStmt | Assignment | ShortVarDecl .
//
// simpleStmt may return missing_stmt if labelOk is set.
func (p *parser) simpleStmt(lhs Expr, rangeOk bool) SimpleStmt {
	if trace {
		defer p.trace("simpleStmt")()
	}

	if rangeOk && p.got(_Range) {
		// _Range expr
		if debug && lhs != nil {
			panic("invalid call of simpleStmt")
		}
		return p.rangeClause(nil, false)
	}

	if lhs == nil {
		lhs = p.exprList()
	}

	if _, ok := lhs.(*ListExpr); !ok && p.tok != _Assign && p.tok != _Define {
		// expr
		switch p.tok {
		case _AssignOp:
			// lhs op= rhs
			op := p.op
			p.next()
			return p.newAssignStmt(op, lhs, p.expr())

		case _IncOp:
			// lhs++ or lhs--
			op := p.op
			p.next()
			return p.newAssignStmt(op, lhs, ImplicitOne)

		case _Arrow:
			// lhs <- rhs
			p.next()
			s := new(SendStmt)
			s.init(p)
			s.Chan = lhs
			s.Value = p.expr()
			if gcCompat {
				s.init(p)
			}
			return s

		default:
			// expr
			return &ExprStmt{X: lhs}
		}
	}

	// expr_list
	switch p.tok {
	case _Assign:
		p.next()

		if rangeOk && p.got(_Range) {
			// expr_list '=' _Range expr
			return p.rangeClause(lhs, false)
		}

		// expr_list '=' expr_list
		return p.newAssignStmt(0, lhs, p.exprList())

	case _Define:
		var n node
		n.init(p)
		p.next()

		if rangeOk && p.got(_Range) {
			// expr_list ':=' range expr
			return p.rangeClause(lhs, true)
		}

		// expr_list ':=' expr_list
		rhs := p.exprList()

		if x, ok := rhs.(*TypeSwitchGuard); ok {
			switch lhs := lhs.(type) {
			case *Name:
				x.Lhs = lhs
			case *ListExpr:
				p.error(fmt.Sprintf("argument count mismatch: %d = %d", len(lhs.ElemList), 1))
			default:
				// TODO(mdempsky): Have Expr types implement Stringer?
				p.error(fmt.Sprintf("invalid variable name %s in type switch", lhs))
			}
			return &ExprStmt{X: x}
		}

		as := p.newAssignStmt(Def, lhs, rhs)
		if gcCompat {
			as.node = n
		}
		return as

	default:
		p.syntax_error("expecting := or = or comma")
		p.advance(_Semi, _Rbrace)
		return nil
	}
}

func (p *parser) rangeClause(lhs Expr, def bool) *RangeClause {
	r := new(RangeClause)
	r.init(p)
	r.Lhs = lhs
	r.Def = def
	r.X = p.expr()
	if gcCompat {
		r.init(p)
	}
	return r
}

func (p *parser) newAssignStmt(op Operator, lhs, rhs Expr) *AssignStmt {
	a := new(AssignStmt)
	a.init(p)
	a.Op = op
	a.Lhs = lhs
	a.Rhs = rhs
	return a
}

func (p *parser) labeledStmt(label *Name) Stmt {
	if trace {
		defer p.trace("labeledStmt")()
	}

	s := new(LabeledStmt)
	s.init(p)
	s.Label = label

	p.want(_Colon)

	if p.tok != _Rbrace && p.tok != _EOF {
		s.Stmt = p.stmt()
		if s.Stmt == missing_stmt {
			// report error at line of ':' token
			p.syntax_error_at(int(label.pos), int(label.line), "missing statement after label")
			// we are already at the end of the labeled statement - no need to advance
			return missing_stmt
		}
	}

	return s
}

func (p *parser) blockStmt() *BlockStmt {
	if trace {
		defer p.trace("blockStmt")()
	}

	s := new(BlockStmt)
	s.init(p)
	p.want(_Lbrace)
	s.Body = p.stmtList()
	p.want(_Rbrace)

	return s
}

func (p *parser) declStmt(f func(*Group) Decl) *DeclStmt {
	if trace {
		defer p.trace("declStmt")()
	}

	s := new(DeclStmt)
	s.init(p)

	p.next() // _Const, _Type, or _Var
	s.DeclList = p.appendGroup(nil, f)

	return s
}

func (p *parser) forStmt() Stmt {
	if trace {
		defer p.trace("forStmt")()
	}

	s := new(ForStmt)
	s.init(p)

	p.want(_For)
	s.Init, s.Cond, s.Post = p.header(true)
	if gcCompat {
		s.init(p)
	}
	s.Body = p.stmtBody("for clause")

	return s
}

// stmtBody parses if and for statement bodies.
func (p *parser) stmtBody(context string) []Stmt {
	if trace {
		defer p.trace("stmtBody")()
	}

	if !p.got(_Lbrace) {
		p.syntax_error("missing { after " + context)
		p.advance(_Name, _Rbrace)
	}

	body := p.stmtList()
	p.want(_Rbrace)

	return body
}

var dummyCond = &Name{Value: "false"}

func (p *parser) header(forStmt bool) (init SimpleStmt, cond Expr, post SimpleStmt) {
	if p.tok == _Lbrace {
		return
	}

	outer := p.xnest
	p.xnest = -1

	if p.tok != _Semi {
		// accept potential varDecl but complain
		if forStmt && p.got(_Var) {
			p.error("var declaration not allowed in for initializer")
		}
		init = p.simpleStmt(nil, forStmt)
		// If we have a range clause, we are done.
		if _, ok := init.(*RangeClause); ok {
			p.xnest = outer
			return
		}
	}

	var condStmt SimpleStmt
	if p.got(_Semi) {
		if forStmt {
			if p.tok != _Semi {
				condStmt = p.simpleStmt(nil, false)
			}
			p.want(_Semi)
			if p.tok != _Lbrace {
				post = p.simpleStmt(nil, false)
			}
		} else if p.tok != _Lbrace {
			condStmt = p.simpleStmt(nil, false)
		}
	} else {
		condStmt = init
		init = nil
	}

	// unpack condStmt
	switch s := condStmt.(type) {
	case nil:
		// nothing to do
	case *ExprStmt:
		cond = s.X
	default:
		p.syntax_error(fmt.Sprintf("%s used as value", String(s)))
		cond = dummyCond // avoid follow-up error for if statements
	}

	p.xnest = outer
	return
}

func (p *parser) ifStmt() *IfStmt {
	if trace {
		defer p.trace("ifStmt")()
	}

	s := new(IfStmt)
	s.init(p)

	p.want(_If)
	s.Init, s.Cond, _ = p.header(false)
	if s.Cond == nil {
		p.error("missing condition in if statement")
	}

	if gcCompat {
		s.init(p)
	}

	s.Then = p.stmtBody("if clause")

	if p.got(_Else) {
		switch p.tok {
		case _If:
			s.Else = p.ifStmt()
		case _Lbrace:
			s.Else = p.blockStmt()
		default:
			p.error("else must be followed by if or statement block")
			p.advance(_Name, _Rbrace)
		}
	}

	return s
}

func (p *parser) switchStmt() *SwitchStmt {
	if trace {
		defer p.trace("switchStmt")()
	}

	p.want(_Switch)
	s := new(SwitchStmt)
	s.init(p)

	s.Init, s.Tag, _ = p.header(false)

	if !p.got(_Lbrace) {
		p.syntax_error("missing { after switch clause")
		p.advance(_Case, _Default, _Rbrace)
	}
	for p.tok != _EOF && p.tok != _Rbrace {
		s.Body = append(s.Body, p.caseClause())
	}
	p.want(_Rbrace)

	return s
}

func (p *parser) selectStmt() *SelectStmt {
	if trace {
		defer p.trace("selectStmt")()
	}

	p.want(_Select)
	s := new(SelectStmt)
	s.init(p)

	if !p.got(_Lbrace) {
		p.syntax_error("missing { after select clause")
		p.advance(_Case, _Default, _Rbrace)
	}
	for p.tok != _EOF && p.tok != _Rbrace {
		s.Body = append(s.Body, p.commClause())
	}
	p.want(_Rbrace)

	return s
}

func (p *parser) caseClause() *CaseClause {
	if trace {
		defer p.trace("caseClause")()
	}

	c := new(CaseClause)
	c.init(p)

	switch p.tok {
	case _Case:
		p.next()
		c.Cases = p.exprList()

	case _Default:
		p.next()

	default:
		p.syntax_error("expecting case or default or }")
		p.advance(_Case, _Default, _Rbrace)
	}

	if gcCompat {
		c.init(p)
	}
	p.want(_Colon)
	c.Body = p.stmtList()

	return c
}

func (p *parser) commClause() *CommClause {
	if trace {
		defer p.trace("commClause")()
	}

	c := new(CommClause)
	c.init(p)

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
		p.advance(_Case, _Default, _Rbrace)
	}

	if gcCompat {
		c.init(p)
	}
	p.want(_Colon)
	c.Body = p.stmtList()

	return c
}

// TODO(gri) find a better solution
var missing_stmt Stmt = new(EmptyStmt) // = nod(OXXX, nil, nil)

// Statement =
// 	Declaration | LabeledStmt | SimpleStmt |
// 	GoStmt | ReturnStmt | BreakStmt | ContinueStmt | GotoStmt |
// 	FallthroughStmt | Block | IfStmt | SwitchStmt | SelectStmt | ForStmt |
// 	DeferStmt .
//
// stmt may return missing_stmt.
func (p *parser) stmt() Stmt {
	if trace {
		defer p.trace("stmt " + p.tok.String())()
	}

	// Most statements (assignments) start with an identifier;
	// look for it first before doing anything more expensive.
	if p.tok == _Name {
		lhs := p.exprList()
		if label, ok := lhs.(*Name); ok && p.tok == _Colon {
			return p.labeledStmt(label)
		}
		return p.simpleStmt(lhs, false)
	}

	switch p.tok {
	case _Lbrace:
		return p.blockStmt()

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
		p.next()
		s := new(BranchStmt)
		s.init(p)
		s.Tok = _Fallthrough
		return s
		// // will be converted to OFALL
		// stmt := nod(OXFALL, nil, nil)
		// stmt.Xoffset = int64(block)
		// return stmt

	case _Break, _Continue:
		tok := p.tok
		p.next()
		s := new(BranchStmt)
		s.init(p)
		s.Tok = tok
		if p.tok == _Name {
			s.Label = p.name()
		}
		return s

	case _Go, _Defer:
		return p.callStmt()

	case _Goto:
		p.next()
		s := new(BranchStmt)
		s.init(p)
		s.Tok = _Goto
		s.Label = p.name()
		return s
		// stmt := nod(OGOTO, p.new_name(p.name()), nil)
		// stmt.Sym = dclstack // context, for goto restrictions
		// return stmt

	case _Return:
		p.next()
		s := new(ReturnStmt)
		s.init(p)
		if p.tok != _Semi && p.tok != _Rbrace {
			s.Results = p.exprList()
		}
		if gcCompat {
			s.init(p)
		}
		return s

	case _Semi:
		s := new(EmptyStmt)
		s.init(p)
		return s
	}

	return missing_stmt
}

// StatementList = { Statement ";" } .
func (p *parser) stmtList() (l []Stmt) {
	if trace {
		defer p.trace("stmtList")()
	}

	for p.tok != _EOF && p.tok != _Rbrace && p.tok != _Case && p.tok != _Default {
		s := p.stmt()
		if s == missing_stmt {
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
	c.init(p)
	c.Fun = fun

	p.want(_Lparen)
	p.xnest++

	for p.tok != _EOF && p.tok != _Rparen {
		c.ArgList = append(c.ArgList, p.expr()) // expr_or_type
		c.HasDots = p.got(_DotDotDot)
		if !p.ocomma(_Rparen) || c.HasDots {
			break
		}
	}

	p.xnest--
	if gcCompat {
		c.init(p)
	}
	p.want(_Rparen)

	return c
}

// ----------------------------------------------------------------------------
// Common productions

func (p *parser) name() *Name {
	// no tracing to avoid overly verbose output

	n := new(Name)
	n.init(p)

	if p.tok == _Name {
		n.Value = p.lit
		p.next()
	} else {
		n.Value = "_"
		p.syntax_error("expecting name")
		p.advance()
	}

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
		name = new(Name)
		name.init(p)
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
		t.init(p) // TODO(gri) what is the correct thing here?
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
