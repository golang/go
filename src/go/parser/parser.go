// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package parser implements a parser for Go source files.
//
// The [ParseFile] function reads file input from a string, []byte, or
// io.Reader, and produces an [ast.File] representing the complete
// abstract syntax tree of the file.
//
// The [ParseExprFrom] function reads a single source-level expression and
// produces an [ast.Expr], the syntax tree of the expression.
//
// The parser accepts a larger language than is syntactically permitted by
// the Go spec, for simplicity, and for improved robustness in the presence
// of syntax errors. For instance, in method declarations, the receiver is
// treated like an ordinary parameter list and thus may contain multiple
// entries where the spec permits exactly one. Consequently, the corresponding
// field in the AST (ast.FuncDecl.Recv) field is not restricted to one entry.
//
// Applications that need to parse one or more complete packages of Go
// source code may find it more convenient not to interact directly
// with the parser but instead to use the Load function in package
// [golang.org/x/tools/go/packages].
package parser

import (
	"fmt"
	"go/ast"
	"go/build/constraint"
	"go/scanner"
	"go/token"
	"strings"
)

// The parser structure holds the parser's internal state.
type parser struct {
	file    *token.File
	errors  scanner.ErrorList
	scanner scanner.Scanner

	// Tracing/debugging
	mode   Mode // parsing mode
	trace  bool // == (mode&Trace != 0)
	indent int  // indentation used for tracing output

	// Comments
	comments    []*ast.CommentGroup
	leadComment *ast.CommentGroup // last lead comment
	lineComment *ast.CommentGroup // last line comment
	top         bool              // in top of file (before package clause)
	goVersion   string            // minimum Go version found in //go:build comment

	// Next token
	pos token.Pos   // token position
	tok token.Token // one token look-ahead
	lit string      // token literal

	// Error recovery
	// (used to limit the number of calls to parser.advance
	// w/o making scanning progress - avoids potential endless
	// loops across multiple parser functions during error recovery)
	syncPos token.Pos // last synchronization position
	syncCnt int       // number of parser.advance calls without progress

	// Non-syntactic parser control
	exprLev int  // < 0: in control clause, >= 0: in expression
	inRhs   bool // if set, the parser is parsing a rhs expression

	imports []*ast.ImportSpec // list of imports

	// nestLev is used to track and limit the recursion depth
	// during parsing.
	nestLev int
}

func (p *parser) init(file *token.File, src []byte, mode Mode) {
	p.file = file
	eh := func(pos token.Position, msg string) { p.errors.Add(pos, msg) }
	p.scanner.Init(p.file, src, eh, scanner.ScanComments)

	p.top = true
	p.mode = mode
	p.trace = mode&Trace != 0 // for convenience (p.trace is used frequently)
	p.next()
}

// ----------------------------------------------------------------------------
// Parsing support

func (p *parser) printTrace(a ...any) {
	const dots = ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . "
	const n = len(dots)
	pos := p.file.Position(p.pos)
	fmt.Printf("%5d:%3d: ", pos.Line, pos.Column)
	i := 2 * p.indent
	for i > n {
		fmt.Print(dots)
		i -= n
	}
	// i <= n
	fmt.Print(dots[0:i])
	fmt.Println(a...)
}

func trace(p *parser, msg string) *parser {
	p.printTrace(msg, "(")
	p.indent++
	return p
}

// Usage pattern: defer un(trace(p, "..."))
func un(p *parser) {
	p.indent--
	p.printTrace(")")
}

// maxNestLev is the deepest we're willing to recurse during parsing
const maxNestLev int = 1e5

func incNestLev(p *parser) *parser {
	p.nestLev++
	if p.nestLev > maxNestLev {
		p.error(p.pos, "exceeded max nesting depth")
		panic(bailout{})
	}
	return p
}

// decNestLev is used to track nesting depth during parsing to prevent stack exhaustion.
// It is used along with incNestLev in a similar fashion to how un and trace are used.
func decNestLev(p *parser) {
	p.nestLev--
}

// Advance to the next token.
func (p *parser) next0() {
	// Because of one-token look-ahead, print the previous token
	// when tracing as it provides a more readable output. The
	// very first token (!p.pos.IsValid()) is not initialized
	// (it is token.ILLEGAL), so don't print it.
	if p.trace && p.pos.IsValid() {
		s := p.tok.String()
		switch {
		case p.tok.IsLiteral():
			p.printTrace(s, p.lit)
		case p.tok.IsOperator(), p.tok.IsKeyword():
			p.printTrace("\"" + s + "\"")
		default:
			p.printTrace(s)
		}
	}

	for {
		p.pos, p.tok, p.lit = p.scanner.Scan()
		if p.tok == token.COMMENT {
			if p.top && strings.HasPrefix(p.lit, "//go:build") {
				if x, err := constraint.Parse(p.lit); err == nil {
					p.goVersion = constraint.GoVersion(x)
				}
			}
			if p.mode&ParseComments == 0 {
				continue
			}
		} else {
			// Found a non-comment; top of file is over.
			p.top = false
		}
		break
	}
}

// lineFor returns the line of pos, ignoring line directive adjustments.
func (p *parser) lineFor(pos token.Pos) int {
	return p.file.PositionFor(pos, false).Line
}

// Consume a comment and return it and the line on which it ends.
func (p *parser) consumeComment() (comment *ast.Comment, endline int) {
	// /*-style comments may end on a different line than where they start.
	// Scan the comment for '\n' chars and adjust endline accordingly.
	endline = p.lineFor(p.pos)
	if p.lit[1] == '*' {
		// don't use range here - no need to decode Unicode code points
		for i := 0; i < len(p.lit); i++ {
			if p.lit[i] == '\n' {
				endline++
			}
		}
	}

	comment = &ast.Comment{Slash: p.pos, Text: p.lit}
	p.next0()

	return
}

// Consume a group of adjacent comments, add it to the parser's
// comments list, and return it together with the line at which
// the last comment in the group ends. A non-comment token or n
// empty lines terminate a comment group.
func (p *parser) consumeCommentGroup(n int) (comments *ast.CommentGroup, endline int) {
	var list []*ast.Comment
	endline = p.lineFor(p.pos)
	for p.tok == token.COMMENT && p.lineFor(p.pos) <= endline+n {
		var comment *ast.Comment
		comment, endline = p.consumeComment()
		list = append(list, comment)
	}

	// add comment group to the comments list
	comments = &ast.CommentGroup{List: list}
	p.comments = append(p.comments, comments)

	return
}

// Advance to the next non-comment token. In the process, collect
// any comment groups encountered, and remember the last lead and
// line comments.
//
// A lead comment is a comment group that starts and ends in a
// line without any other tokens and that is followed by a non-comment
// token on the line immediately after the comment group.
//
// A line comment is a comment group that follows a non-comment
// token on the same line, and that has no tokens after it on the line
// where it ends.
//
// Lead and line comments may be considered documentation that is
// stored in the AST.
func (p *parser) next() {
	p.leadComment = nil
	p.lineComment = nil
	prev := p.pos
	p.next0()

	if p.tok == token.COMMENT {
		var comment *ast.CommentGroup
		var endline int

		if p.lineFor(p.pos) == p.lineFor(prev) {
			// The comment is on same line as the previous token; it
			// cannot be a lead comment but may be a line comment.
			comment, endline = p.consumeCommentGroup(0)
			if p.lineFor(p.pos) != endline || p.tok == token.SEMICOLON || p.tok == token.EOF {
				// The next token is on a different line, thus
				// the last comment group is a line comment.
				p.lineComment = comment
			}
		}

		// consume successor comments, if any
		endline = -1
		for p.tok == token.COMMENT {
			comment, endline = p.consumeCommentGroup(1)
		}

		if endline+1 == p.lineFor(p.pos) {
			// The next token is following on the line immediately after the
			// comment group, thus the last comment group is a lead comment.
			p.leadComment = comment
		}
	}
}

// A bailout panic is raised to indicate early termination. pos and msg are
// only populated when bailing out of object resolution.
type bailout struct {
	pos token.Pos
	msg string
}

func (p *parser) error(pos token.Pos, msg string) {
	if p.trace {
		defer un(trace(p, "error: "+msg))
	}

	epos := p.file.Position(pos)

	// If AllErrors is not set, discard errors reported on the same line
	// as the last recorded error and stop parsing if there are more than
	// 10 errors.
	if p.mode&AllErrors == 0 {
		n := len(p.errors)
		if n > 0 && p.errors[n-1].Pos.Line == epos.Line {
			return // discard - likely a spurious error
		}
		if n > 10 {
			panic(bailout{})
		}
	}

	p.errors.Add(epos, msg)
}

func (p *parser) errorExpected(pos token.Pos, msg string) {
	msg = "expected " + msg
	if pos == p.pos {
		// the error happened at the current position;
		// make the error message more specific
		switch {
		case p.tok == token.SEMICOLON && p.lit == "\n":
			msg += ", found newline"
		case p.tok.IsLiteral():
			// print 123 rather than 'INT', etc.
			msg += ", found " + p.lit
		default:
			msg += ", found '" + p.tok.String() + "'"
		}
	}
	p.error(pos, msg)
}

func (p *parser) expect(tok token.Token) token.Pos {
	pos := p.pos
	if p.tok != tok {
		p.errorExpected(pos, "'"+tok.String()+"'")
	}
	p.next() // make progress
	return pos
}

// expect2 is like expect, but it returns an invalid position
// if the expected token is not found.
func (p *parser) expect2(tok token.Token) (pos token.Pos) {
	if p.tok == tok {
		pos = p.pos
	} else {
		p.errorExpected(p.pos, "'"+tok.String()+"'")
	}
	p.next() // make progress
	return
}

// expectClosing is like expect but provides a better error message
// for the common case of a missing comma before a newline.
func (p *parser) expectClosing(tok token.Token, context string) token.Pos {
	if p.tok != tok && p.tok == token.SEMICOLON && p.lit == "\n" {
		p.error(p.pos, "missing ',' before newline in "+context)
		p.next()
	}
	return p.expect(tok)
}

// expectSemi consumes a semicolon and returns the applicable line comment.
func (p *parser) expectSemi() (comment *ast.CommentGroup) {
	// semicolon is optional before a closing ')' or '}'
	if p.tok != token.RPAREN && p.tok != token.RBRACE {
		switch p.tok {
		case token.COMMA:
			// permit a ',' instead of a ';' but complain
			p.errorExpected(p.pos, "';'")
			fallthrough
		case token.SEMICOLON:
			if p.lit == ";" {
				// explicit semicolon
				p.next()
				comment = p.lineComment // use following comments
			} else {
				// artificial semicolon
				comment = p.lineComment // use preceding comments
				p.next()
			}
			return comment
		default:
			p.errorExpected(p.pos, "';'")
			p.advance(stmtStart)
		}
	}
	return nil
}

func (p *parser) atComma(context string, follow token.Token) bool {
	if p.tok == token.COMMA {
		return true
	}
	if p.tok != follow {
		msg := "missing ','"
		if p.tok == token.SEMICOLON && p.lit == "\n" {
			msg += " before newline"
		}
		p.error(p.pos, msg+" in "+context)
		return true // "insert" comma and continue
	}
	return false
}

func assert(cond bool, msg string) {
	if !cond {
		panic("go/parser internal error: " + msg)
	}
}

// advance consumes tokens until the current token p.tok
// is in the 'to' set, or token.EOF. For error recovery.
func (p *parser) advance(to map[token.Token]bool) {
	for ; p.tok != token.EOF; p.next() {
		if to[p.tok] {
			// Return only if parser made some progress since last
			// sync or if it has not reached 10 advance calls without
			// progress. Otherwise consume at least one token to
			// avoid an endless parser loop (it is possible that
			// both parseOperand and parseStmt call advance and
			// correctly do not advance, thus the need for the
			// invocation limit p.syncCnt).
			if p.pos == p.syncPos && p.syncCnt < 10 {
				p.syncCnt++
				return
			}
			if p.pos > p.syncPos {
				p.syncPos = p.pos
				p.syncCnt = 0
				return
			}
			// Reaching here indicates a parser bug, likely an
			// incorrect token list in this function, but it only
			// leads to skipping of possibly correct code if a
			// previous error is present, and thus is preferred
			// over a non-terminating parse.
		}
	}
}

var stmtStart = map[token.Token]bool{
	token.BREAK:       true,
	token.CONST:       true,
	token.CONTINUE:    true,
	token.DEFER:       true,
	token.FALLTHROUGH: true,
	token.FOR:         true,
	token.GO:          true,
	token.GOTO:        true,
	token.IF:          true,
	token.RETURN:      true,
	token.SELECT:      true,
	token.SWITCH:      true,
	token.TYPE:        true,
	token.VAR:         true,
}

var declStart = map[token.Token]bool{
	token.IMPORT: true,
	token.CONST:  true,
	token.TYPE:   true,
	token.VAR:    true,
}

var exprEnd = map[token.Token]bool{
	token.COMMA:     true,
	token.COLON:     true,
	token.SEMICOLON: true,
	token.RPAREN:    true,
	token.RBRACK:    true,
	token.RBRACE:    true,
}

// safePos returns a valid file position for a given position: If pos
// is valid to begin with, safePos returns pos. If pos is out-of-range,
// safePos returns the EOF position.
//
// This is hack to work around "artificial" end positions in the AST which
// are computed by adding 1 to (presumably valid) token positions. If the
// token positions are invalid due to parse errors, the resulting end position
// may be past the file's EOF position, which would lead to panics if used
// later on.
func (p *parser) safePos(pos token.Pos) (res token.Pos) {
	defer func() {
		if recover() != nil {
			res = token.Pos(p.file.Base() + p.file.Size()) // EOF position
		}
	}()
	_ = p.file.Offset(pos) // trigger a panic if position is out-of-range
	return pos
}

// ----------------------------------------------------------------------------
// Identifiers

func (p *parser) parseIdent() *ast.Ident {
	pos := p.pos
	name := "_"
	if p.tok == token.IDENT {
		name = p.lit
		p.next()
	} else {
		p.expect(token.IDENT) // use expect() error handling
	}
	return &ast.Ident{NamePos: pos, Name: name}
}

func (p *parser) parseIdentList() (list []*ast.Ident) {
	if p.trace {
		defer un(trace(p, "IdentList"))
	}

	list = append(list, p.parseIdent())
	for p.tok == token.COMMA {
		p.next()
		list = append(list, p.parseIdent())
	}

	return
}

// ----------------------------------------------------------------------------
// Common productions

// If lhs is set, result list elements which are identifiers are not resolved.
func (p *parser) parseExprList() (list []ast.Expr) {
	if p.trace {
		defer un(trace(p, "ExpressionList"))
	}

	list = append(list, p.parseExpr())
	for p.tok == token.COMMA {
		p.next()
		list = append(list, p.parseExpr())
	}

	return
}

func (p *parser) parseList(inRhs bool) []ast.Expr {
	old := p.inRhs
	p.inRhs = inRhs
	list := p.parseExprList()
	p.inRhs = old
	return list
}

// ----------------------------------------------------------------------------
// Types

func (p *parser) parseType() ast.Expr {
	if p.trace {
		defer un(trace(p, "Type"))
	}

	typ := p.tryIdentOrType()

	if typ == nil {
		pos := p.pos
		p.errorExpected(pos, "type")
		p.advance(exprEnd)
		return &ast.BadExpr{From: pos, To: p.pos}
	}

	return typ
}

func (p *parser) parseQualifiedIdent(ident *ast.Ident) ast.Expr {
	if p.trace {
		defer un(trace(p, "QualifiedIdent"))
	}

	typ := p.parseTypeName(ident)
	if p.tok == token.LBRACK {
		typ = p.parseTypeInstance(typ)
	}

	return typ
}

// If the result is an identifier, it is not resolved.
func (p *parser) parseTypeName(ident *ast.Ident) ast.Expr {
	if p.trace {
		defer un(trace(p, "TypeName"))
	}

	if ident == nil {
		ident = p.parseIdent()
	}

	if p.tok == token.PERIOD {
		// ident is a package name
		p.next()
		sel := p.parseIdent()
		return &ast.SelectorExpr{X: ident, Sel: sel}
	}

	return ident
}

// "[" has already been consumed, and lbrack is its position.
// If len != nil it is the already consumed array length.
func (p *parser) parseArrayType(lbrack token.Pos, len ast.Expr) *ast.ArrayType {
	if p.trace {
		defer un(trace(p, "ArrayType"))
	}

	if len == nil {
		p.exprLev++
		// always permit ellipsis for more fault-tolerant parsing
		if p.tok == token.ELLIPSIS {
			len = &ast.Ellipsis{Ellipsis: p.pos}
			p.next()
		} else if p.tok != token.RBRACK {
			len = p.parseRhs()
		}
		p.exprLev--
	}
	if p.tok == token.COMMA {
		// Trailing commas are accepted in type parameter
		// lists but not in array type declarations.
		// Accept for better error handling but complain.
		p.error(p.pos, "unexpected comma; expecting ]")
		p.next()
	}
	p.expect(token.RBRACK)
	elt := p.parseType()
	return &ast.ArrayType{Lbrack: lbrack, Len: len, Elt: elt}
}

func (p *parser) parseArrayFieldOrTypeInstance(x *ast.Ident) (*ast.Ident, ast.Expr) {
	if p.trace {
		defer un(trace(p, "ArrayFieldOrTypeInstance"))
	}

	lbrack := p.expect(token.LBRACK)
	trailingComma := token.NoPos // if valid, the position of a trailing comma preceding the ']'
	var args []ast.Expr
	if p.tok != token.RBRACK {
		p.exprLev++
		args = append(args, p.parseRhs())
		for p.tok == token.COMMA {
			comma := p.pos
			p.next()
			if p.tok == token.RBRACK {
				trailingComma = comma
				break
			}
			args = append(args, p.parseRhs())
		}
		p.exprLev--
	}
	rbrack := p.expect(token.RBRACK)

	if len(args) == 0 {
		// x []E
		elt := p.parseType()
		return x, &ast.ArrayType{Lbrack: lbrack, Elt: elt}
	}

	// x [P]E or x[P]
	if len(args) == 1 {
		elt := p.tryIdentOrType()
		if elt != nil {
			// x [P]E
			if trailingComma.IsValid() {
				// Trailing commas are invalid in array type fields.
				p.error(trailingComma, "unexpected comma; expecting ]")
			}
			return x, &ast.ArrayType{Lbrack: lbrack, Len: args[0], Elt: elt}
		}
	}

	// x[P], x[P1, P2], ...
	return nil, packIndexExpr(x, lbrack, args, rbrack)
}

func (p *parser) parseFieldDecl() *ast.Field {
	if p.trace {
		defer un(trace(p, "FieldDecl"))
	}

	doc := p.leadComment

	var names []*ast.Ident
	var typ ast.Expr
	switch p.tok {
	case token.IDENT:
		name := p.parseIdent()
		if p.tok == token.PERIOD || p.tok == token.STRING || p.tok == token.SEMICOLON || p.tok == token.RBRACE {
			// embedded type
			typ = name
			if p.tok == token.PERIOD {
				typ = p.parseQualifiedIdent(name)
			}
		} else {
			// name1, name2, ... T
			names = []*ast.Ident{name}
			for p.tok == token.COMMA {
				p.next()
				names = append(names, p.parseIdent())
			}
			// Careful dance: We don't know if we have an embedded instantiated
			// type T[P1, P2, ...] or a field T of array type []E or [P]E.
			if len(names) == 1 && p.tok == token.LBRACK {
				name, typ = p.parseArrayFieldOrTypeInstance(name)
				if name == nil {
					names = nil
				}
			} else {
				// T P
				typ = p.parseType()
			}
		}
	case token.MUL:
		star := p.pos
		p.next()
		if p.tok == token.LPAREN {
			// *(T)
			p.error(p.pos, "cannot parenthesize embedded type")
			p.next()
			typ = p.parseQualifiedIdent(nil)
			// expect closing ')' but no need to complain if missing
			if p.tok == token.RPAREN {
				p.next()
			}
		} else {
			// *T
			typ = p.parseQualifiedIdent(nil)
		}
		typ = &ast.StarExpr{Star: star, X: typ}

	case token.LPAREN:
		p.error(p.pos, "cannot parenthesize embedded type")
		p.next()
		if p.tok == token.MUL {
			// (*T)
			star := p.pos
			p.next()
			typ = &ast.StarExpr{Star: star, X: p.parseQualifiedIdent(nil)}
		} else {
			// (T)
			typ = p.parseQualifiedIdent(nil)
		}
		// expect closing ')' but no need to complain if missing
		if p.tok == token.RPAREN {
			p.next()
		}

	default:
		pos := p.pos
		p.errorExpected(pos, "field name or embedded type")
		p.advance(exprEnd)
		typ = &ast.BadExpr{From: pos, To: p.pos}
	}

	var tag *ast.BasicLit
	if p.tok == token.STRING {
		tag = &ast.BasicLit{ValuePos: p.pos, Kind: p.tok, Value: p.lit}
		p.next()
	}

	comment := p.expectSemi()

	field := &ast.Field{Doc: doc, Names: names, Type: typ, Tag: tag, Comment: comment}
	return field
}

func (p *parser) parseStructType() *ast.StructType {
	if p.trace {
		defer un(trace(p, "StructType"))
	}

	pos := p.expect(token.STRUCT)
	lbrace := p.expect(token.LBRACE)
	var list []*ast.Field
	for p.tok == token.IDENT || p.tok == token.MUL || p.tok == token.LPAREN {
		// a field declaration cannot start with a '(' but we accept
		// it here for more robust parsing and better error messages
		// (parseFieldDecl will check and complain if necessary)
		list = append(list, p.parseFieldDecl())
	}
	rbrace := p.expect(token.RBRACE)

	return &ast.StructType{
		Struct: pos,
		Fields: &ast.FieldList{
			Opening: lbrace,
			List:    list,
			Closing: rbrace,
		},
	}
}

func (p *parser) parsePointerType() *ast.StarExpr {
	if p.trace {
		defer un(trace(p, "PointerType"))
	}

	star := p.expect(token.MUL)
	base := p.parseType()

	return &ast.StarExpr{Star: star, X: base}
}

func (p *parser) parseDotsType() *ast.Ellipsis {
	if p.trace {
		defer un(trace(p, "DotsType"))
	}

	pos := p.expect(token.ELLIPSIS)
	elt := p.parseType()

	return &ast.Ellipsis{Ellipsis: pos, Elt: elt}
}

type field struct {
	name *ast.Ident
	typ  ast.Expr
}

func (p *parser) parseParamDecl(name *ast.Ident, typeSetsOK bool) (f field) {
	// TODO(rFindley) refactor to be more similar to paramDeclOrNil in the syntax
	// package
	if p.trace {
		defer un(trace(p, "ParamDecl"))
	}

	ptok := p.tok
	if name != nil {
		p.tok = token.IDENT // force token.IDENT case in switch below
	} else if typeSetsOK && p.tok == token.TILDE {
		// "~" ...
		return field{nil, p.embeddedElem(nil)}
	}

	switch p.tok {
	case token.IDENT:
		// name
		if name != nil {
			f.name = name
			p.tok = ptok
		} else {
			f.name = p.parseIdent()
		}
		switch p.tok {
		case token.IDENT, token.MUL, token.ARROW, token.FUNC, token.CHAN, token.MAP, token.STRUCT, token.INTERFACE, token.LPAREN:
			// name type
			f.typ = p.parseType()

		case token.LBRACK:
			// name "[" type1, ..., typeN "]" or name "[" n "]" type
			f.name, f.typ = p.parseArrayFieldOrTypeInstance(f.name)

		case token.ELLIPSIS:
			// name "..." type
			f.typ = p.parseDotsType()
			return // don't allow ...type "|" ...

		case token.PERIOD:
			// name "." ...
			f.typ = p.parseQualifiedIdent(f.name)
			f.name = nil

		case token.TILDE:
			if typeSetsOK {
				f.typ = p.embeddedElem(nil)
				return
			}

		case token.OR:
			if typeSetsOK {
				// name "|" typeset
				f.typ = p.embeddedElem(f.name)
				f.name = nil
				return
			}
		}

	case token.MUL, token.ARROW, token.FUNC, token.LBRACK, token.CHAN, token.MAP, token.STRUCT, token.INTERFACE, token.LPAREN:
		// type
		f.typ = p.parseType()

	case token.ELLIPSIS:
		// "..." type
		// (always accepted)
		f.typ = p.parseDotsType()
		return // don't allow ...type "|" ...

	default:
		// TODO(rfindley): this is incorrect in the case of type parameter lists
		//                 (should be "']'" in that case)
		p.errorExpected(p.pos, "')'")
		p.advance(exprEnd)
	}

	// [name] type "|"
	if typeSetsOK && p.tok == token.OR && f.typ != nil {
		f.typ = p.embeddedElem(f.typ)
	}

	return
}

func (p *parser) parseParameterList(name0 *ast.Ident, typ0 ast.Expr, closing token.Token, dddok bool) (params []*ast.Field) {
	if p.trace {
		defer un(trace(p, "ParameterList"))
	}

	// Type parameters are the only parameter list closed by ']'.
	tparams := closing == token.RBRACK

	pos0 := p.pos
	if name0 != nil {
		pos0 = name0.Pos()
	} else if typ0 != nil {
		pos0 = typ0.Pos()
	}

	// Note: The code below matches the corresponding code in the syntax
	//       parser closely. Changes must be reflected in either parser.
	//       For the code to match, we use the local []field list that
	//       corresponds to []syntax.Field. At the end, the list must be
	//       converted into an []*ast.Field.

	var list []field
	var named int // number of parameters that have an explicit name and type
	var typed int // number of parameters that have an explicit type

	for name0 != nil || p.tok != closing && p.tok != token.EOF {
		var par field
		if typ0 != nil {
			if tparams {
				typ0 = p.embeddedElem(typ0)
			}
			par = field{name0, typ0}
		} else {
			par = p.parseParamDecl(name0, tparams)
		}
		name0 = nil // 1st name was consumed if present
		typ0 = nil  // 1st typ was consumed if present
		if par.name != nil || par.typ != nil {
			list = append(list, par)
			if par.name != nil && par.typ != nil {
				named++
			}
			if par.typ != nil {
				typed++
			}
		}
		if !p.atComma("parameter list", closing) {
			break
		}
		p.next()
	}

	if len(list) == 0 {
		return // not uncommon
	}

	// distribute parameter types (len(list) > 0)
	if named == 0 {
		// all unnamed => found names are type names
		for i := range list {
			par := &list[i]
			if typ := par.name; typ != nil {
				par.typ = typ
				par.name = nil
			}
		}
		if tparams {
			// This is the same error handling as below, adjusted for type parameters only.
			// See comment below for details. (go.dev/issue/64534)
			var errPos token.Pos
			var msg string
			if named == typed /* same as typed == 0 */ {
				errPos = p.pos // position error at closing ]
				msg = "missing type constraint"
			} else {
				errPos = pos0 // position at opening [ or first name
				msg = "missing type parameter name"
				if len(list) == 1 {
					msg += " or invalid array length"
				}
			}
			p.error(errPos, msg)
		}
	} else if named != len(list) {
		// some named or we're in a type parameter list => all must be named
		var errPos token.Pos // left-most error position (or invalid)
		var typ ast.Expr     // current type (from right to left)
		for i := range list {
			if par := &list[len(list)-i-1]; par.typ != nil {
				typ = par.typ
				if par.name == nil {
					errPos = typ.Pos()
					n := ast.NewIdent("_")
					n.NamePos = errPos // correct position
					par.name = n
				}
			} else if typ != nil {
				par.typ = typ
			} else {
				// par.typ == nil && typ == nil => we only have a par.name
				errPos = par.name.Pos()
				par.typ = &ast.BadExpr{From: errPos, To: p.pos}
			}
		}
		if errPos.IsValid() {
			// Not all parameters are named because named != len(list).
			// If named == typed, there must be parameters that have no types.
			// They must be at the end of the parameter list, otherwise types
			// would have been filled in by the right-to-left sweep above and
			// there would be no error.
			// If tparams is set, the parameter list is a type parameter list.
			var msg string
			if named == typed {
				errPos = p.pos // position error at closing token ) or ]
				if tparams {
					msg = "missing type constraint"
				} else {
					msg = "missing parameter type"
				}
			} else {
				if tparams {
					msg = "missing type parameter name"
					// go.dev/issue/60812
					if len(list) == 1 {
						msg += " or invalid array length"
					}
				} else {
					msg = "missing parameter name"
				}
			}
			p.error(errPos, msg)
		}
	}

	// check use of ...
	first := true // only report first occurrence
	for i, _ := range list {
		f := &list[i]
		if t, _ := f.typ.(*ast.Ellipsis); t != nil && (!dddok || i+1 < len(list)) {
			if first {
				first = false
				if dddok {
					p.error(t.Ellipsis, "can only use ... with final parameter")
				} else {
					p.error(t.Ellipsis, "invalid use of ...")
				}
			}
			// use T instead of invalid ...T
			// TODO(gri) would like to use `f.typ = t.Elt` but that causes problems
			//           with the resolver in cases of reuse of the same identifier
			f.typ = &ast.BadExpr{From: t.Pos(), To: t.End()}
		}
	}

	// Convert list to []*ast.Field.
	// If list contains types only, each type gets its own ast.Field.
	if named == 0 {
		// parameter list consists of types only
		for _, par := range list {
			assert(par.typ != nil, "nil type in unnamed parameter list")
			params = append(params, &ast.Field{Type: par.typ})
		}
		return
	}

	// If the parameter list consists of named parameters with types,
	// collect all names with the same types into a single ast.Field.
	var names []*ast.Ident
	var typ ast.Expr
	addParams := func() {
		assert(typ != nil, "nil type in named parameter list")
		field := &ast.Field{Names: names, Type: typ}
		params = append(params, field)
		names = nil
	}
	for _, par := range list {
		if par.typ != typ {
			if len(names) > 0 {
				addParams()
			}
			typ = par.typ
		}
		names = append(names, par.name)
	}
	if len(names) > 0 {
		addParams()
	}
	return
}

func (p *parser) parseTypeParameters() *ast.FieldList {
	if p.trace {
		defer un(trace(p, "TypeParameters"))
	}

	lbrack := p.expect(token.LBRACK)
	var list []*ast.Field
	if p.tok != token.RBRACK {
		list = p.parseParameterList(nil, nil, token.RBRACK, false)
	}
	rbrack := p.expect(token.RBRACK)

	if len(list) == 0 {
		p.error(rbrack, "empty type parameter list")
		return nil // avoid follow-on errors
	}

	return &ast.FieldList{Opening: lbrack, List: list, Closing: rbrack}
}

func (p *parser) parseParameters(result bool) *ast.FieldList {
	if p.trace {
		defer un(trace(p, "Parameters"))
	}

	if !result || p.tok == token.LPAREN {
		lparen := p.expect(token.LPAREN)
		var list []*ast.Field
		if p.tok != token.RPAREN {
			list = p.parseParameterList(nil, nil, token.RPAREN, !result)
		}
		rparen := p.expect(token.RPAREN)
		return &ast.FieldList{Opening: lparen, List: list, Closing: rparen}
	}

	if typ := p.tryIdentOrType(); typ != nil {
		list := make([]*ast.Field, 1)
		list[0] = &ast.Field{Type: typ}
		return &ast.FieldList{List: list}
	}

	return nil
}

func (p *parser) parseFuncType() *ast.FuncType {
	if p.trace {
		defer un(trace(p, "FuncType"))
	}

	pos := p.expect(token.FUNC)
	// accept type parameters for more tolerant parsing but complain
	if p.tok == token.LBRACK {
		tparams := p.parseTypeParameters()
		if tparams != nil {
			p.error(tparams.Opening, "function type must have no type parameters")
		}
	}
	params := p.parseParameters(false)
	results := p.parseParameters(true)

	return &ast.FuncType{Func: pos, Params: params, Results: results}
}

func (p *parser) parseMethodSpec() *ast.Field {
	if p.trace {
		defer un(trace(p, "MethodSpec"))
	}

	doc := p.leadComment
	var idents []*ast.Ident
	var typ ast.Expr
	x := p.parseTypeName(nil)
	if ident, _ := x.(*ast.Ident); ident != nil {
		switch {
		case p.tok == token.LBRACK:
			// generic method or embedded instantiated type
			lbrack := p.pos
			p.next()
			p.exprLev++
			x := p.parseExpr()
			p.exprLev--
			if name0, _ := x.(*ast.Ident); name0 != nil && p.tok != token.COMMA && p.tok != token.RBRACK {
				// generic method m[T any]
				//
				// Interface methods do not have type parameters. We parse them for a
				// better error message and improved error recovery.
				_ = p.parseParameterList(name0, nil, token.RBRACK, false)
				_ = p.expect(token.RBRACK)
				p.error(lbrack, "interface method must have no type parameters")

				// TODO(rfindley) refactor to share code with parseFuncType.
				params := p.parseParameters(false)
				results := p.parseParameters(true)
				idents = []*ast.Ident{ident}
				typ = &ast.FuncType{
					Func:    token.NoPos,
					Params:  params,
					Results: results,
				}
			} else {
				// embedded instantiated type
				// TODO(rfindley) should resolve all identifiers in x.
				list := []ast.Expr{x}
				if p.atComma("type argument list", token.RBRACK) {
					p.exprLev++
					p.next()
					for p.tok != token.RBRACK && p.tok != token.EOF {
						list = append(list, p.parseType())
						if !p.atComma("type argument list", token.RBRACK) {
							break
						}
						p.next()
					}
					p.exprLev--
				}
				rbrack := p.expectClosing(token.RBRACK, "type argument list")
				typ = packIndexExpr(ident, lbrack, list, rbrack)
			}
		case p.tok == token.LPAREN:
			// ordinary method
			// TODO(rfindley) refactor to share code with parseFuncType.
			params := p.parseParameters(false)
			results := p.parseParameters(true)
			idents = []*ast.Ident{ident}
			typ = &ast.FuncType{Func: token.NoPos, Params: params, Results: results}
		default:
			// embedded type
			typ = x
		}
	} else {
		// embedded, possibly instantiated type
		typ = x
		if p.tok == token.LBRACK {
			// embedded instantiated interface
			typ = p.parseTypeInstance(typ)
		}
	}

	// Comment is added at the callsite: the field below may joined with
	// additional type specs using '|'.
	// TODO(rfindley) this should be refactored.
	// TODO(rfindley) add more tests for comment handling.
	return &ast.Field{Doc: doc, Names: idents, Type: typ}
}

func (p *parser) embeddedElem(x ast.Expr) ast.Expr {
	if p.trace {
		defer un(trace(p, "EmbeddedElem"))
	}
	if x == nil {
		x = p.embeddedTerm()
	}
	for p.tok == token.OR {
		t := new(ast.BinaryExpr)
		t.OpPos = p.pos
		t.Op = token.OR
		p.next()
		t.X = x
		t.Y = p.embeddedTerm()
		x = t
	}
	return x
}

func (p *parser) embeddedTerm() ast.Expr {
	if p.trace {
		defer un(trace(p, "EmbeddedTerm"))
	}
	if p.tok == token.TILDE {
		t := new(ast.UnaryExpr)
		t.OpPos = p.pos
		t.Op = token.TILDE
		p.next()
		t.X = p.parseType()
		return t
	}

	t := p.tryIdentOrType()
	if t == nil {
		pos := p.pos
		p.errorExpected(pos, "~ term or type")
		p.advance(exprEnd)
		return &ast.BadExpr{From: pos, To: p.pos}
	}

	return t
}

func (p *parser) parseInterfaceType() *ast.InterfaceType {
	if p.trace {
		defer un(trace(p, "InterfaceType"))
	}

	pos := p.expect(token.INTERFACE)
	lbrace := p.expect(token.LBRACE)

	var list []*ast.Field

parseElements:
	for {
		switch {
		case p.tok == token.IDENT:
			f := p.parseMethodSpec()
			if f.Names == nil {
				f.Type = p.embeddedElem(f.Type)
			}
			f.Comment = p.expectSemi()
			list = append(list, f)
		case p.tok == token.TILDE:
			typ := p.embeddedElem(nil)
			comment := p.expectSemi()
			list = append(list, &ast.Field{Type: typ, Comment: comment})
		default:
			if t := p.tryIdentOrType(); t != nil {
				typ := p.embeddedElem(t)
				comment := p.expectSemi()
				list = append(list, &ast.Field{Type: typ, Comment: comment})
			} else {
				break parseElements
			}
		}
	}

	// TODO(rfindley): the error produced here could be improved, since we could
	// accept an identifier, 'type', or a '}' at this point.
	rbrace := p.expect(token.RBRACE)

	return &ast.InterfaceType{
		Interface: pos,
		Methods: &ast.FieldList{
			Opening: lbrace,
			List:    list,
			Closing: rbrace,
		},
	}
}

func (p *parser) parseMapType() *ast.MapType {
	if p.trace {
		defer un(trace(p, "MapType"))
	}

	pos := p.expect(token.MAP)
	p.expect(token.LBRACK)
	key := p.parseType()
	p.expect(token.RBRACK)
	value := p.parseType()

	return &ast.MapType{Map: pos, Key: key, Value: value}
}

func (p *parser) parseChanType() *ast.ChanType {
	if p.trace {
		defer un(trace(p, "ChanType"))
	}

	pos := p.pos
	dir := ast.SEND | ast.RECV
	var arrow token.Pos
	if p.tok == token.CHAN {
		p.next()
		if p.tok == token.ARROW {
			arrow = p.pos
			p.next()
			dir = ast.SEND
		}
	} else {
		arrow = p.expect(token.ARROW)
		p.expect(token.CHAN)
		dir = ast.RECV
	}
	value := p.parseType()

	return &ast.ChanType{Begin: pos, Arrow: arrow, Dir: dir, Value: value}
}

func (p *parser) parseTypeInstance(typ ast.Expr) ast.Expr {
	if p.trace {
		defer un(trace(p, "TypeInstance"))
	}

	opening := p.expect(token.LBRACK)
	p.exprLev++
	var list []ast.Expr
	for p.tok != token.RBRACK && p.tok != token.EOF {
		list = append(list, p.parseType())
		if !p.atComma("type argument list", token.RBRACK) {
			break
		}
		p.next()
	}
	p.exprLev--

	closing := p.expectClosing(token.RBRACK, "type argument list")

	if len(list) == 0 {
		p.errorExpected(closing, "type argument list")
		return &ast.IndexExpr{
			X:      typ,
			Lbrack: opening,
			Index:  &ast.BadExpr{From: opening + 1, To: closing},
			Rbrack: closing,
		}
	}

	return packIndexExpr(typ, opening, list, closing)
}

func (p *parser) tryIdentOrType() ast.Expr {
	defer decNestLev(incNestLev(p))

	switch p.tok {
	case token.IDENT:
		typ := p.parseTypeName(nil)
		if p.tok == token.LBRACK {
			typ = p.parseTypeInstance(typ)
		}
		return typ
	case token.LBRACK:
		lbrack := p.expect(token.LBRACK)
		return p.parseArrayType(lbrack, nil)
	case token.STRUCT:
		return p.parseStructType()
	case token.MUL:
		return p.parsePointerType()
	case token.FUNC:
		return p.parseFuncType()
	case token.INTERFACE:
		return p.parseInterfaceType()
	case token.MAP:
		return p.parseMapType()
	case token.CHAN, token.ARROW:
		return p.parseChanType()
	case token.LPAREN:
		lparen := p.pos
		p.next()
		typ := p.parseType()
		rparen := p.expect(token.RPAREN)
		return &ast.ParenExpr{Lparen: lparen, X: typ, Rparen: rparen}
	}

	// no type found
	return nil
}

// ----------------------------------------------------------------------------
// Blocks

func (p *parser) parseStmtList() (list []ast.Stmt) {
	if p.trace {
		defer un(trace(p, "StatementList"))
	}

	for p.tok != token.CASE && p.tok != token.DEFAULT && p.tok != token.RBRACE && p.tok != token.EOF {
		list = append(list, p.parseStmt())
	}

	return
}

func (p *parser) parseBody() *ast.BlockStmt {
	if p.trace {
		defer un(trace(p, "Body"))
	}

	lbrace := p.expect(token.LBRACE)
	list := p.parseStmtList()
	rbrace := p.expect2(token.RBRACE)

	return &ast.BlockStmt{Lbrace: lbrace, List: list, Rbrace: rbrace}
}

func (p *parser) parseBlockStmt() *ast.BlockStmt {
	if p.trace {
		defer un(trace(p, "BlockStmt"))
	}

	lbrace := p.expect(token.LBRACE)
	list := p.parseStmtList()
	rbrace := p.expect2(token.RBRACE)

	return &ast.BlockStmt{Lbrace: lbrace, List: list, Rbrace: rbrace}
}

// ----------------------------------------------------------------------------
// Expressions

func (p *parser) parseFuncTypeOrLit() ast.Expr {
	if p.trace {
		defer un(trace(p, "FuncTypeOrLit"))
	}

	typ := p.parseFuncType()
	if p.tok != token.LBRACE {
		// function type only
		return typ
	}

	p.exprLev++
	body := p.parseBody()
	p.exprLev--

	return &ast.FuncLit{Type: typ, Body: body}
}

// parseOperand may return an expression or a raw type (incl. array
// types of the form [...]T). Callers must verify the result.
func (p *parser) parseOperand() ast.Expr {
	if p.trace {
		defer un(trace(p, "Operand"))
	}

	switch p.tok {
	case token.IDENT:
		x := p.parseIdent()
		return x

	case token.INT, token.FLOAT, token.IMAG, token.CHAR, token.STRING:
		x := &ast.BasicLit{ValuePos: p.pos, Kind: p.tok, Value: p.lit}
		p.next()
		return x

	case token.LPAREN:
		lparen := p.pos
		p.next()
		p.exprLev++
		x := p.parseRhs() // types may be parenthesized: (some type)
		p.exprLev--
		rparen := p.expect(token.RPAREN)
		return &ast.ParenExpr{Lparen: lparen, X: x, Rparen: rparen}

	case token.FUNC:
		return p.parseFuncTypeOrLit()
	}

	if typ := p.tryIdentOrType(); typ != nil { // do not consume trailing type parameters
		// could be type for composite literal or conversion
		_, isIdent := typ.(*ast.Ident)
		assert(!isIdent, "type cannot be identifier")
		return typ
	}

	// we have an error
	pos := p.pos
	p.errorExpected(pos, "operand")
	p.advance(stmtStart)
	return &ast.BadExpr{From: pos, To: p.pos}
}

func (p *parser) parseSelector(x ast.Expr) ast.Expr {
	if p.trace {
		defer un(trace(p, "Selector"))
	}

	sel := p.parseIdent()

	return &ast.SelectorExpr{X: x, Sel: sel}
}

func (p *parser) parseTypeAssertion(x ast.Expr) ast.Expr {
	if p.trace {
		defer un(trace(p, "TypeAssertion"))
	}

	lparen := p.expect(token.LPAREN)
	var typ ast.Expr
	if p.tok == token.TYPE {
		// type switch: typ == nil
		p.next()
	} else {
		typ = p.parseType()
	}
	rparen := p.expect(token.RPAREN)

	return &ast.TypeAssertExpr{X: x, Type: typ, Lparen: lparen, Rparen: rparen}
}

func (p *parser) parseIndexOrSliceOrInstance(x ast.Expr) ast.Expr {
	if p.trace {
		defer un(trace(p, "parseIndexOrSliceOrInstance"))
	}

	lbrack := p.expect(token.LBRACK)
	if p.tok == token.RBRACK {
		// empty index, slice or index expressions are not permitted;
		// accept them for parsing tolerance, but complain
		p.errorExpected(p.pos, "operand")
		rbrack := p.pos
		p.next()
		return &ast.IndexExpr{
			X:      x,
			Lbrack: lbrack,
			Index:  &ast.BadExpr{From: rbrack, To: rbrack},
			Rbrack: rbrack,
		}
	}
	p.exprLev++

	const N = 3 // change the 3 to 2 to disable 3-index slices
	var args []ast.Expr
	var index [N]ast.Expr
	var colons [N - 1]token.Pos
	if p.tok != token.COLON {
		// We can't know if we have an index expression or a type instantiation;
		// so even if we see a (named) type we are not going to be in type context.
		index[0] = p.parseRhs()
	}
	ncolons := 0
	switch p.tok {
	case token.COLON:
		// slice expression
		for p.tok == token.COLON && ncolons < len(colons) {
			colons[ncolons] = p.pos
			ncolons++
			p.next()
			if p.tok != token.COLON && p.tok != token.RBRACK && p.tok != token.EOF {
				index[ncolons] = p.parseRhs()
			}
		}
	case token.COMMA:
		// instance expression
		args = append(args, index[0])
		for p.tok == token.COMMA {
			p.next()
			if p.tok != token.RBRACK && p.tok != token.EOF {
				args = append(args, p.parseType())
			}
		}
	}

	p.exprLev--
	rbrack := p.expect(token.RBRACK)

	if ncolons > 0 {
		// slice expression
		slice3 := false
		if ncolons == 2 {
			slice3 = true
			// Check presence of middle and final index here rather than during type-checking
			// to prevent erroneous programs from passing through gofmt (was go.dev/issue/7305).
			if index[1] == nil {
				p.error(colons[0], "middle index required in 3-index slice")
				index[1] = &ast.BadExpr{From: colons[0] + 1, To: colons[1]}
			}
			if index[2] == nil {
				p.error(colons[1], "final index required in 3-index slice")
				index[2] = &ast.BadExpr{From: colons[1] + 1, To: rbrack}
			}
		}
		return &ast.SliceExpr{X: x, Lbrack: lbrack, Low: index[0], High: index[1], Max: index[2], Slice3: slice3, Rbrack: rbrack}
	}

	if len(args) == 0 {
		// index expression
		return &ast.IndexExpr{X: x, Lbrack: lbrack, Index: index[0], Rbrack: rbrack}
	}

	// instance expression
	return packIndexExpr(x, lbrack, args, rbrack)
}

func (p *parser) parseCallOrConversion(fun ast.Expr) *ast.CallExpr {
	if p.trace {
		defer un(trace(p, "CallOrConversion"))
	}

	lparen := p.expect(token.LPAREN)
	p.exprLev++
	var list []ast.Expr
	var ellipsis token.Pos
	for p.tok != token.RPAREN && p.tok != token.EOF && !ellipsis.IsValid() {
		list = append(list, p.parseRhs()) // builtins may expect a type: make(some type, ...)
		if p.tok == token.ELLIPSIS {
			ellipsis = p.pos
			p.next()
		}
		if !p.atComma("argument list", token.RPAREN) {
			break
		}
		p.next()
	}
	p.exprLev--
	rparen := p.expectClosing(token.RPAREN, "argument list")

	return &ast.CallExpr{Fun: fun, Lparen: lparen, Args: list, Ellipsis: ellipsis, Rparen: rparen}
}

func (p *parser) parseValue() ast.Expr {
	if p.trace {
		defer un(trace(p, "Element"))
	}

	if p.tok == token.LBRACE {
		return p.parseLiteralValue(nil)
	}

	x := p.parseExpr()

	return x
}

func (p *parser) parseElement() ast.Expr {
	if p.trace {
		defer un(trace(p, "Element"))
	}

	x := p.parseValue()
	if p.tok == token.COLON {
		colon := p.pos
		p.next()
		x = &ast.KeyValueExpr{Key: x, Colon: colon, Value: p.parseValue()}
	}

	return x
}

func (p *parser) parseElementList() (list []ast.Expr) {
	if p.trace {
		defer un(trace(p, "ElementList"))
	}

	for p.tok != token.RBRACE && p.tok != token.EOF {
		list = append(list, p.parseElement())
		if !p.atComma("composite literal", token.RBRACE) {
			break
		}
		p.next()
	}

	return
}

func (p *parser) parseLiteralValue(typ ast.Expr) ast.Expr {
	defer decNestLev(incNestLev(p))

	if p.trace {
		defer un(trace(p, "LiteralValue"))
	}

	lbrace := p.expect(token.LBRACE)
	var elts []ast.Expr
	p.exprLev++
	if p.tok != token.RBRACE {
		elts = p.parseElementList()
	}
	p.exprLev--
	rbrace := p.expectClosing(token.RBRACE, "composite literal")
	return &ast.CompositeLit{Type: typ, Lbrace: lbrace, Elts: elts, Rbrace: rbrace}
}

func (p *parser) parsePrimaryExpr(x ast.Expr) ast.Expr {
	if p.trace {
		defer un(trace(p, "PrimaryExpr"))
	}

	if x == nil {
		x = p.parseOperand()
	}
	// We track the nesting here rather than at the entry for the function,
	// since it can iteratively produce a nested output, and we want to
	// limit how deep a structure we generate.
	var n int
	defer func() { p.nestLev -= n }()
	for n = 1; ; n++ {
		incNestLev(p)
		switch p.tok {
		case token.PERIOD:
			p.next()
			switch p.tok {
			case token.IDENT:
				x = p.parseSelector(x)
			case token.LPAREN:
				x = p.parseTypeAssertion(x)
			default:
				pos := p.pos
				p.errorExpected(pos, "selector or type assertion")
				// TODO(rFindley) The check for token.RBRACE below is a targeted fix
				//                to error recovery sufficient to make the x/tools tests to
				//                pass with the new parsing logic introduced for type
				//                parameters. Remove this once error recovery has been
				//                more generally reconsidered.
				if p.tok != token.RBRACE {
					p.next() // make progress
				}
				sel := &ast.Ident{NamePos: pos, Name: "_"}
				x = &ast.SelectorExpr{X: x, Sel: sel}
			}
		case token.LBRACK:
			x = p.parseIndexOrSliceOrInstance(x)
		case token.LPAREN:
			x = p.parseCallOrConversion(x)
		case token.LBRACE:
			// operand may have returned a parenthesized complit
			// type; accept it but complain if we have a complit
			t := ast.Unparen(x)
			// determine if '{' belongs to a composite literal or a block statement
			switch t.(type) {
			case *ast.BadExpr, *ast.Ident, *ast.SelectorExpr:
				if p.exprLev < 0 {
					return x
				}
				// x is possibly a composite literal type
			case *ast.IndexExpr, *ast.IndexListExpr:
				if p.exprLev < 0 {
					return x
				}
				// x is possibly a composite literal type
			case *ast.ArrayType, *ast.StructType, *ast.MapType:
				// x is a composite literal type
			default:
				return x
			}
			if t != x {
				p.error(t.Pos(), "cannot parenthesize type in composite literal")
				// already progressed, no need to advance
			}
			x = p.parseLiteralValue(x)
		default:
			return x
		}
	}
}

func (p *parser) parseUnaryExpr() ast.Expr {
	defer decNestLev(incNestLev(p))

	if p.trace {
		defer un(trace(p, "UnaryExpr"))
	}

	switch p.tok {
	case token.ADD, token.SUB, token.NOT, token.XOR, token.AND, token.TILDE:
		pos, op := p.pos, p.tok
		p.next()
		x := p.parseUnaryExpr()
		return &ast.UnaryExpr{OpPos: pos, Op: op, X: x}

	case token.ARROW:
		// channel type or receive expression
		arrow := p.pos
		p.next()

		// If the next token is token.CHAN we still don't know if it
		// is a channel type or a receive operation - we only know
		// once we have found the end of the unary expression. There
		// are two cases:
		//
		//   <- type  => (<-type) must be channel type
		//   <- expr  => <-(expr) is a receive from an expression
		//
		// In the first case, the arrow must be re-associated with
		// the channel type parsed already:
		//
		//   <- (chan type)    =>  (<-chan type)
		//   <- (chan<- type)  =>  (<-chan (<-type))

		x := p.parseUnaryExpr()

		// determine which case we have
		if typ, ok := x.(*ast.ChanType); ok {
			// (<-type)

			// re-associate position info and <-
			dir := ast.SEND
			for ok && dir == ast.SEND {
				if typ.Dir == ast.RECV {
					// error: (<-type) is (<-(<-chan T))
					p.errorExpected(typ.Arrow, "'chan'")
				}
				arrow, typ.Begin, typ.Arrow = typ.Arrow, arrow, arrow
				dir, typ.Dir = typ.Dir, ast.RECV
				typ, ok = typ.Value.(*ast.ChanType)
			}
			if dir == ast.SEND {
				p.errorExpected(arrow, "channel type")
			}

			return x
		}

		// <-(expr)
		return &ast.UnaryExpr{OpPos: arrow, Op: token.ARROW, X: x}

	case token.MUL:
		// pointer type or unary "*" expression
		pos := p.pos
		p.next()
		x := p.parseUnaryExpr()
		return &ast.StarExpr{Star: pos, X: x}
	}

	return p.parsePrimaryExpr(nil)
}

func (p *parser) tokPrec() (token.Token, int) {
	tok := p.tok
	if p.inRhs && tok == token.ASSIGN {
		tok = token.EQL
	}
	return tok, tok.Precedence()
}

// parseBinaryExpr parses a (possibly) binary expression.
// If x is non-nil, it is used as the left operand.
//
// TODO(rfindley): parseBinaryExpr has become overloaded. Consider refactoring.
func (p *parser) parseBinaryExpr(x ast.Expr, prec1 int) ast.Expr {
	if p.trace {
		defer un(trace(p, "BinaryExpr"))
	}

	if x == nil {
		x = p.parseUnaryExpr()
	}
	// We track the nesting here rather than at the entry for the function,
	// since it can iteratively produce a nested output, and we want to
	// limit how deep a structure we generate.
	var n int
	defer func() { p.nestLev -= n }()
	for n = 1; ; n++ {
		incNestLev(p)
		op, oprec := p.tokPrec()
		if oprec < prec1 {
			return x
		}
		pos := p.expect(op)
		y := p.parseBinaryExpr(nil, oprec+1)
		x = &ast.BinaryExpr{X: x, OpPos: pos, Op: op, Y: y}
	}
}

// The result may be a type or even a raw type ([...]int).
func (p *parser) parseExpr() ast.Expr {
	if p.trace {
		defer un(trace(p, "Expression"))
	}

	return p.parseBinaryExpr(nil, token.LowestPrec+1)
}

func (p *parser) parseRhs() ast.Expr {
	old := p.inRhs
	p.inRhs = true
	x := p.parseExpr()
	p.inRhs = old
	return x
}

// ----------------------------------------------------------------------------
// Statements

// Parsing modes for parseSimpleStmt.
const (
	basic = iota
	labelOk
	rangeOk
)

// parseSimpleStmt returns true as 2nd result if it parsed the assignment
// of a range clause (with mode == rangeOk). The returned statement is an
// assignment with a right-hand side that is a single unary expression of
// the form "range x". No guarantees are given for the left-hand side.
func (p *parser) parseSimpleStmt(mode int) (ast.Stmt, bool) {
	if p.trace {
		defer un(trace(p, "SimpleStmt"))
	}

	x := p.parseList(false)

	switch p.tok {
	case
		token.DEFINE, token.ASSIGN, token.ADD_ASSIGN,
		token.SUB_ASSIGN, token.MUL_ASSIGN, token.QUO_ASSIGN,
		token.REM_ASSIGN, token.AND_ASSIGN, token.OR_ASSIGN,
		token.XOR_ASSIGN, token.SHL_ASSIGN, token.SHR_ASSIGN, token.AND_NOT_ASSIGN:
		// assignment statement, possibly part of a range clause
		pos, tok := p.pos, p.tok
		p.next()
		var y []ast.Expr
		isRange := false
		if mode == rangeOk && p.tok == token.RANGE && (tok == token.DEFINE || tok == token.ASSIGN) {
			pos := p.pos
			p.next()
			y = []ast.Expr{&ast.UnaryExpr{OpPos: pos, Op: token.RANGE, X: p.parseRhs()}}
			isRange = true
		} else {
			y = p.parseList(true)
		}
		return &ast.AssignStmt{Lhs: x, TokPos: pos, Tok: tok, Rhs: y}, isRange
	}

	if len(x) > 1 {
		p.errorExpected(x[0].Pos(), "1 expression")
		// continue with first expression
	}

	switch p.tok {
	case token.COLON:
		// labeled statement
		colon := p.pos
		p.next()
		if label, isIdent := x[0].(*ast.Ident); mode == labelOk && isIdent {
			// Go spec: The scope of a label is the body of the function
			// in which it is declared and excludes the body of any nested
			// function.
			stmt := &ast.LabeledStmt{Label: label, Colon: colon, Stmt: p.parseStmt()}
			return stmt, false
		}
		// The label declaration typically starts at x[0].Pos(), but the label
		// declaration may be erroneous due to a token after that position (and
		// before the ':'). If SpuriousErrors is not set, the (only) error
		// reported for the line is the illegal label error instead of the token
		// before the ':' that caused the problem. Thus, use the (latest) colon
		// position for error reporting.
		p.error(colon, "illegal label declaration")
		return &ast.BadStmt{From: x[0].Pos(), To: colon + 1}, false

	case token.ARROW:
		// send statement
		arrow := p.pos
		p.next()
		y := p.parseRhs()
		return &ast.SendStmt{Chan: x[0], Arrow: arrow, Value: y}, false

	case token.INC, token.DEC:
		// increment or decrement
		s := &ast.IncDecStmt{X: x[0], TokPos: p.pos, Tok: p.tok}
		p.next()
		return s, false
	}

	// expression
	return &ast.ExprStmt{X: x[0]}, false
}

func (p *parser) parseCallExpr(callType string) *ast.CallExpr {
	x := p.parseRhs() // could be a conversion: (some type)(x)
	if t := ast.Unparen(x); t != x {
		p.error(x.Pos(), fmt.Sprintf("expression in %s must not be parenthesized", callType))
		x = t
	}
	if call, isCall := x.(*ast.CallExpr); isCall {
		return call
	}
	if _, isBad := x.(*ast.BadExpr); !isBad {
		// only report error if it's a new one
		p.error(p.safePos(x.End()), fmt.Sprintf("expression in %s must be function call", callType))
	}
	return nil
}

func (p *parser) parseGoStmt() ast.Stmt {
	if p.trace {
		defer un(trace(p, "GoStmt"))
	}

	pos := p.expect(token.GO)
	call := p.parseCallExpr("go")
	p.expectSemi()
	if call == nil {
		return &ast.BadStmt{From: pos, To: pos + 2} // len("go")
	}

	return &ast.GoStmt{Go: pos, Call: call}
}

func (p *parser) parseDeferStmt() ast.Stmt {
	if p.trace {
		defer un(trace(p, "DeferStmt"))
	}

	pos := p.expect(token.DEFER)
	call := p.parseCallExpr("defer")
	p.expectSemi()
	if call == nil {
		return &ast.BadStmt{From: pos, To: pos + 5} // len("defer")
	}

	return &ast.DeferStmt{Defer: pos, Call: call}
}

func (p *parser) parseReturnStmt() *ast.ReturnStmt {
	if p.trace {
		defer un(trace(p, "ReturnStmt"))
	}

	pos := p.pos
	p.expect(token.RETURN)
	var x []ast.Expr
	if p.tok != token.SEMICOLON && p.tok != token.RBRACE {
		x = p.parseList(true)
	}
	p.expectSemi()

	return &ast.ReturnStmt{Return: pos, Results: x}
}

func (p *parser) parseBranchStmt(tok token.Token) *ast.BranchStmt {
	if p.trace {
		defer un(trace(p, "BranchStmt"))
	}

	pos := p.expect(tok)
	var label *ast.Ident
	if tok == token.GOTO || ((tok == token.CONTINUE || tok == token.BREAK) && p.tok == token.IDENT) {
		label = p.parseIdent()
	}
	p.expectSemi()

	return &ast.BranchStmt{TokPos: pos, Tok: tok, Label: label}
}

func (p *parser) makeExpr(s ast.Stmt, want string) ast.Expr {
	if s == nil {
		return nil
	}
	if es, isExpr := s.(*ast.ExprStmt); isExpr {
		return es.X
	}
	found := "simple statement"
	if _, isAss := s.(*ast.AssignStmt); isAss {
		found = "assignment"
	}
	p.error(s.Pos(), fmt.Sprintf("expected %s, found %s (missing parentheses around composite literal?)", want, found))
	return &ast.BadExpr{From: s.Pos(), To: p.safePos(s.End())}
}

// parseIfHeader is an adjusted version of parser.header
// in cmd/compile/internal/syntax/parser.go, which has
// been tuned for better error handling.
func (p *parser) parseIfHeader() (init ast.Stmt, cond ast.Expr) {
	if p.tok == token.LBRACE {
		p.error(p.pos, "missing condition in if statement")
		cond = &ast.BadExpr{From: p.pos, To: p.pos}
		return
	}
	// p.tok != token.LBRACE

	prevLev := p.exprLev
	p.exprLev = -1

	if p.tok != token.SEMICOLON {
		// accept potential variable declaration but complain
		if p.tok == token.VAR {
			p.next()
			p.error(p.pos, "var declaration not allowed in if initializer")
		}
		init, _ = p.parseSimpleStmt(basic)
	}

	var condStmt ast.Stmt
	var semi struct {
		pos token.Pos
		lit string // ";" or "\n"; valid if pos.IsValid()
	}
	if p.tok != token.LBRACE {
		if p.tok == token.SEMICOLON {
			semi.pos = p.pos
			semi.lit = p.lit
			p.next()
		} else {
			p.expect(token.SEMICOLON)
		}
		if p.tok != token.LBRACE {
			condStmt, _ = p.parseSimpleStmt(basic)
		}
	} else {
		condStmt = init
		init = nil
	}

	if condStmt != nil {
		cond = p.makeExpr(condStmt, "boolean expression")
	} else if semi.pos.IsValid() {
		if semi.lit == "\n" {
			p.error(semi.pos, "unexpected newline, expecting { after if clause")
		} else {
			p.error(semi.pos, "missing condition in if statement")
		}
	}

	// make sure we have a valid AST
	if cond == nil {
		cond = &ast.BadExpr{From: p.pos, To: p.pos}
	}

	p.exprLev = prevLev
	return
}

func (p *parser) parseIfStmt() *ast.IfStmt {
	defer decNestLev(incNestLev(p))

	if p.trace {
		defer un(trace(p, "IfStmt"))
	}

	pos := p.expect(token.IF)

	init, cond := p.parseIfHeader()
	body := p.parseBlockStmt()

	var else_ ast.Stmt
	if p.tok == token.ELSE {
		p.next()
		switch p.tok {
		case token.IF:
			else_ = p.parseIfStmt()
		case token.LBRACE:
			else_ = p.parseBlockStmt()
			p.expectSemi()
		default:
			p.errorExpected(p.pos, "if statement or block")
			else_ = &ast.BadStmt{From: p.pos, To: p.pos}
		}
	} else {
		p.expectSemi()
	}

	return &ast.IfStmt{If: pos, Init: init, Cond: cond, Body: body, Else: else_}
}

func (p *parser) parseCaseClause() *ast.CaseClause {
	if p.trace {
		defer un(trace(p, "CaseClause"))
	}

	pos := p.pos
	var list []ast.Expr
	if p.tok == token.CASE {
		p.next()
		list = p.parseList(true)
	} else {
		p.expect(token.DEFAULT)
	}

	colon := p.expect(token.COLON)
	body := p.parseStmtList()

	return &ast.CaseClause{Case: pos, List: list, Colon: colon, Body: body}
}

func isTypeSwitchAssert(x ast.Expr) bool {
	a, ok := x.(*ast.TypeAssertExpr)
	return ok && a.Type == nil
}

func (p *parser) isTypeSwitchGuard(s ast.Stmt) bool {
	switch t := s.(type) {
	case *ast.ExprStmt:
		// x.(type)
		return isTypeSwitchAssert(t.X)
	case *ast.AssignStmt:
		// v := x.(type)
		if len(t.Lhs) == 1 && len(t.Rhs) == 1 && isTypeSwitchAssert(t.Rhs[0]) {
			switch t.Tok {
			case token.ASSIGN:
				// permit v = x.(type) but complain
				p.error(t.TokPos, "expected ':=', found '='")
				fallthrough
			case token.DEFINE:
				return true
			}
		}
	}
	return false
}

func (p *parser) parseSwitchStmt() ast.Stmt {
	if p.trace {
		defer un(trace(p, "SwitchStmt"))
	}

	pos := p.expect(token.SWITCH)

	var s1, s2 ast.Stmt
	if p.tok != token.LBRACE {
		prevLev := p.exprLev
		p.exprLev = -1
		if p.tok != token.SEMICOLON {
			s2, _ = p.parseSimpleStmt(basic)
		}
		if p.tok == token.SEMICOLON {
			p.next()
			s1 = s2
			s2 = nil
			if p.tok != token.LBRACE {
				// A TypeSwitchGuard may declare a variable in addition
				// to the variable declared in the initial SimpleStmt.
				// Introduce extra scope to avoid redeclaration errors:
				//
				//	switch t := 0; t := x.(T) { ... }
				//
				// (this code is not valid Go because the first t
				// cannot be accessed and thus is never used, the extra
				// scope is needed for the correct error message).
				//
				// If we don't have a type switch, s2 must be an expression.
				// Having the extra nested but empty scope won't affect it.
				s2, _ = p.parseSimpleStmt(basic)
			}
		}
		p.exprLev = prevLev
	}

	typeSwitch := p.isTypeSwitchGuard(s2)
	lbrace := p.expect(token.LBRACE)
	var list []ast.Stmt
	for p.tok == token.CASE || p.tok == token.DEFAULT {
		list = append(list, p.parseCaseClause())
	}
	rbrace := p.expect(token.RBRACE)
	p.expectSemi()
	body := &ast.BlockStmt{Lbrace: lbrace, List: list, Rbrace: rbrace}

	if typeSwitch {
		return &ast.TypeSwitchStmt{Switch: pos, Init: s1, Assign: s2, Body: body}
	}

	return &ast.SwitchStmt{Switch: pos, Init: s1, Tag: p.makeExpr(s2, "switch expression"), Body: body}
}

func (p *parser) parseCommClause() *ast.CommClause {
	if p.trace {
		defer un(trace(p, "CommClause"))
	}

	pos := p.pos
	var comm ast.Stmt
	if p.tok == token.CASE {
		p.next()
		lhs := p.parseList(false)
		if p.tok == token.ARROW {
			// SendStmt
			if len(lhs) > 1 {
				p.errorExpected(lhs[0].Pos(), "1 expression")
				// continue with first expression
			}
			arrow := p.pos
			p.next()
			rhs := p.parseRhs()
			comm = &ast.SendStmt{Chan: lhs[0], Arrow: arrow, Value: rhs}
		} else {
			// RecvStmt
			if tok := p.tok; tok == token.ASSIGN || tok == token.DEFINE {
				// RecvStmt with assignment
				if len(lhs) > 2 {
					p.errorExpected(lhs[0].Pos(), "1 or 2 expressions")
					// continue with first two expressions
					lhs = lhs[0:2]
				}
				pos := p.pos
				p.next()
				rhs := p.parseRhs()
				comm = &ast.AssignStmt{Lhs: lhs, TokPos: pos, Tok: tok, Rhs: []ast.Expr{rhs}}
			} else {
				// lhs must be single receive operation
				if len(lhs) > 1 {
					p.errorExpected(lhs[0].Pos(), "1 expression")
					// continue with first expression
				}
				comm = &ast.ExprStmt{X: lhs[0]}
			}
		}
	} else {
		p.expect(token.DEFAULT)
	}

	colon := p.expect(token.COLON)
	body := p.parseStmtList()

	return &ast.CommClause{Case: pos, Comm: comm, Colon: colon, Body: body}
}

func (p *parser) parseSelectStmt() *ast.SelectStmt {
	if p.trace {
		defer un(trace(p, "SelectStmt"))
	}

	pos := p.expect(token.SELECT)
	lbrace := p.expect(token.LBRACE)
	var list []ast.Stmt
	for p.tok == token.CASE || p.tok == token.DEFAULT {
		list = append(list, p.parseCommClause())
	}
	rbrace := p.expect(token.RBRACE)
	p.expectSemi()
	body := &ast.BlockStmt{Lbrace: lbrace, List: list, Rbrace: rbrace}

	return &ast.SelectStmt{Select: pos, Body: body}
}

func (p *parser) parseForStmt() ast.Stmt {
	if p.trace {
		defer un(trace(p, "ForStmt"))
	}

	pos := p.expect(token.FOR)

	var s1, s2, s3 ast.Stmt
	var isRange bool
	if p.tok != token.LBRACE {
		prevLev := p.exprLev
		p.exprLev = -1
		if p.tok != token.SEMICOLON {
			if p.tok == token.RANGE {
				// "for range x" (nil lhs in assignment)
				pos := p.pos
				p.next()
				y := []ast.Expr{&ast.UnaryExpr{OpPos: pos, Op: token.RANGE, X: p.parseRhs()}}
				s2 = &ast.AssignStmt{Rhs: y}
				isRange = true
			} else {
				s2, isRange = p.parseSimpleStmt(rangeOk)
			}
		}
		if !isRange && p.tok == token.SEMICOLON {
			p.next()
			s1 = s2
			s2 = nil
			if p.tok != token.SEMICOLON {
				s2, _ = p.parseSimpleStmt(basic)
			}
			p.expectSemi()
			if p.tok != token.LBRACE {
				s3, _ = p.parseSimpleStmt(basic)
			}
		}
		p.exprLev = prevLev
	}

	body := p.parseBlockStmt()
	p.expectSemi()

	if isRange {
		as := s2.(*ast.AssignStmt)
		// check lhs
		var key, value ast.Expr
		switch len(as.Lhs) {
		case 0:
			// nothing to do
		case 1:
			key = as.Lhs[0]
		case 2:
			key, value = as.Lhs[0], as.Lhs[1]
		default:
			p.errorExpected(as.Lhs[len(as.Lhs)-1].Pos(), "at most 2 expressions")
			return &ast.BadStmt{From: pos, To: p.safePos(body.End())}
		}
		// parseSimpleStmt returned a right-hand side that
		// is a single unary expression of the form "range x"
		x := as.Rhs[0].(*ast.UnaryExpr).X
		return &ast.RangeStmt{
			For:    pos,
			Key:    key,
			Value:  value,
			TokPos: as.TokPos,
			Tok:    as.Tok,
			Range:  as.Rhs[0].Pos(),
			X:      x,
			Body:   body,
		}
	}

	// regular for statement
	return &ast.ForStmt{
		For:  pos,
		Init: s1,
		Cond: p.makeExpr(s2, "boolean or range expression"),
		Post: s3,
		Body: body,
	}
}

func (p *parser) parseStmt() (s ast.Stmt) {
	defer decNestLev(incNestLev(p))

	if p.trace {
		defer un(trace(p, "Statement"))
	}

	switch p.tok {
	case token.CONST, token.TYPE, token.VAR:
		s = &ast.DeclStmt{Decl: p.parseDecl(stmtStart)}
	case
		// tokens that may start an expression
		token.IDENT, token.INT, token.FLOAT, token.IMAG, token.CHAR, token.STRING, token.FUNC, token.LPAREN, // operands
		token.LBRACK, token.STRUCT, token.MAP, token.CHAN, token.INTERFACE, // composite types
		token.ADD, token.SUB, token.MUL, token.AND, token.XOR, token.ARROW, token.NOT: // unary operators
		s, _ = p.parseSimpleStmt(labelOk)
		// because of the required look-ahead, labeled statements are
		// parsed by parseSimpleStmt - don't expect a semicolon after
		// them
		if _, isLabeledStmt := s.(*ast.LabeledStmt); !isLabeledStmt {
			p.expectSemi()
		}
	case token.GO:
		s = p.parseGoStmt()
	case token.DEFER:
		s = p.parseDeferStmt()
	case token.RETURN:
		s = p.parseReturnStmt()
	case token.BREAK, token.CONTINUE, token.GOTO, token.FALLTHROUGH:
		s = p.parseBranchStmt(p.tok)
	case token.LBRACE:
		s = p.parseBlockStmt()
		p.expectSemi()
	case token.IF:
		s = p.parseIfStmt()
	case token.SWITCH:
		s = p.parseSwitchStmt()
	case token.SELECT:
		s = p.parseSelectStmt()
	case token.FOR:
		s = p.parseForStmt()
	case token.SEMICOLON:
		// Is it ever possible to have an implicit semicolon
		// producing an empty statement in a valid program?
		// (handle correctly anyway)
		s = &ast.EmptyStmt{Semicolon: p.pos, Implicit: p.lit == "\n"}
		p.next()
	case token.RBRACE:
		// a semicolon may be omitted before a closing "}"
		s = &ast.EmptyStmt{Semicolon: p.pos, Implicit: true}
	default:
		// no statement found
		pos := p.pos
		p.errorExpected(pos, "statement")
		p.advance(stmtStart)
		s = &ast.BadStmt{From: pos, To: p.pos}
	}

	return
}

// ----------------------------------------------------------------------------
// Declarations

type parseSpecFunction func(doc *ast.CommentGroup, keyword token.Token, iota int) ast.Spec

func (p *parser) parseImportSpec(doc *ast.CommentGroup, _ token.Token, _ int) ast.Spec {
	if p.trace {
		defer un(trace(p, "ImportSpec"))
	}

	var ident *ast.Ident
	switch p.tok {
	case token.IDENT:
		ident = p.parseIdent()
	case token.PERIOD:
		ident = &ast.Ident{NamePos: p.pos, Name: "."}
		p.next()
	}

	pos := p.pos
	var path string
	if p.tok == token.STRING {
		path = p.lit
		p.next()
	} else if p.tok.IsLiteral() {
		p.error(pos, "import path must be a string")
		p.next()
	} else {
		p.error(pos, "missing import path")
		p.advance(exprEnd)
	}
	comment := p.expectSemi()

	// collect imports
	spec := &ast.ImportSpec{
		Doc:     doc,
		Name:    ident,
		Path:    &ast.BasicLit{ValuePos: pos, Kind: token.STRING, Value: path},
		Comment: comment,
	}
	p.imports = append(p.imports, spec)

	return spec
}

func (p *parser) parseValueSpec(doc *ast.CommentGroup, keyword token.Token, iota int) ast.Spec {
	if p.trace {
		defer un(trace(p, keyword.String()+"Spec"))
	}

	idents := p.parseIdentList()
	var typ ast.Expr
	var values []ast.Expr
	switch keyword {
	case token.CONST:
		// always permit optional type and initialization for more tolerant parsing
		if p.tok != token.EOF && p.tok != token.SEMICOLON && p.tok != token.RPAREN {
			typ = p.tryIdentOrType()
			if p.tok == token.ASSIGN {
				p.next()
				values = p.parseList(true)
			}
		}
	case token.VAR:
		if p.tok != token.ASSIGN {
			typ = p.parseType()
		}
		if p.tok == token.ASSIGN {
			p.next()
			values = p.parseList(true)
		}
	default:
		panic("unreachable")
	}
	comment := p.expectSemi()

	spec := &ast.ValueSpec{
		Doc:     doc,
		Names:   idents,
		Type:    typ,
		Values:  values,
		Comment: comment,
	}
	return spec
}

func (p *parser) parseGenericType(spec *ast.TypeSpec, openPos token.Pos, name0 *ast.Ident, typ0 ast.Expr) {
	if p.trace {
		defer un(trace(p, "parseGenericType"))
	}

	list := p.parseParameterList(name0, typ0, token.RBRACK, false)
	closePos := p.expect(token.RBRACK)
	spec.TypeParams = &ast.FieldList{Opening: openPos, List: list, Closing: closePos}
	if p.tok == token.ASSIGN {
		// type alias
		spec.Assign = p.pos
		p.next()
	}
	spec.Type = p.parseType()
}

func (p *parser) parseTypeSpec(doc *ast.CommentGroup, _ token.Token, _ int) ast.Spec {
	if p.trace {
		defer un(trace(p, "TypeSpec"))
	}

	name := p.parseIdent()
	spec := &ast.TypeSpec{Doc: doc, Name: name}

	if p.tok == token.LBRACK {
		// spec.Name "[" ...
		// array/slice type or type parameter list
		lbrack := p.pos
		p.next()
		if p.tok == token.IDENT {
			// We may have an array type or a type parameter list.
			// In either case we expect an expression x (which may
			// just be a name, or a more complex expression) which
			// we can analyze further.
			//
			// A type parameter list may have a type bound starting
			// with a "[" as in: P []E. In that case, simply parsing
			// an expression would lead to an error: P[] is invalid.
			// But since index or slice expressions are never constant
			// and thus invalid array length expressions, if the name
			// is followed by "[" it must be the start of an array or
			// slice constraint. Only if we don't see a "[" do we
			// need to parse a full expression. Notably, name <- x
			// is not a concern because name <- x is a statement and
			// not an expression.
			var x ast.Expr = p.parseIdent()
			if p.tok != token.LBRACK {
				// To parse the expression starting with name, expand
				// the call sequence we would get by passing in name
				// to parser.expr, and pass in name to parsePrimaryExpr.
				p.exprLev++
				lhs := p.parsePrimaryExpr(x)
				x = p.parseBinaryExpr(lhs, token.LowestPrec+1)
				p.exprLev--
			}
			// Analyze expression x. If we can split x into a type parameter
			// name, possibly followed by a type parameter type, we consider
			// this the start of a type parameter list, with some caveats:
			// a single name followed by "]" tilts the decision towards an
			// array declaration; a type parameter type that could also be
			// an ordinary expression but which is followed by a comma tilts
			// the decision towards a type parameter list.
			if pname, ptype := extractName(x, p.tok == token.COMMA); pname != nil && (ptype != nil || p.tok != token.RBRACK) {
				// spec.Name "[" pname ...
				// spec.Name "[" pname ptype ...
				// spec.Name "[" pname ptype "," ...
				p.parseGenericType(spec, lbrack, pname, ptype) // ptype may be nil
			} else {
				// spec.Name "[" pname "]" ...
				// spec.Name "[" x ...
				spec.Type = p.parseArrayType(lbrack, x)
			}
		} else {
			// array type
			spec.Type = p.parseArrayType(lbrack, nil)
		}
	} else {
		// no type parameters
		if p.tok == token.ASSIGN {
			// type alias
			spec.Assign = p.pos
			p.next()
		}
		spec.Type = p.parseType()
	}

	spec.Comment = p.expectSemi()

	return spec
}

// extractName splits the expression x into (name, expr) if syntactically
// x can be written as name expr. The split only happens if expr is a type
// element (per the isTypeElem predicate) or if force is set.
// If x is just a name, the result is (name, nil). If the split succeeds,
// the result is (name, expr). Otherwise the result is (nil, x).
// Examples:
//
//	x           force    name    expr
//	------------------------------------
//	P*[]int     T/F      P       *[]int
//	P*E         T        P       *E
//	P*E         F        nil     P*E
//	P([]int)    T/F      P       ([]int)
//	P(E)        T        P       (E)
//	P(E)        F        nil     P(E)
//	P*E|F|~G    T/F      P       *E|F|~G
//	P*E|F|G     T        P       *E|F|G
//	P*E|F|G     F        nil     P*E|F|G
func extractName(x ast.Expr, force bool) (*ast.Ident, ast.Expr) {
	switch x := x.(type) {
	case *ast.Ident:
		return x, nil
	case *ast.BinaryExpr:
		switch x.Op {
		case token.MUL:
			if name, _ := x.X.(*ast.Ident); name != nil && (force || isTypeElem(x.Y)) {
				// x = name *x.Y
				return name, &ast.StarExpr{Star: x.OpPos, X: x.Y}
			}
		case token.OR:
			if name, lhs := extractName(x.X, force || isTypeElem(x.Y)); name != nil && lhs != nil {
				// x = name lhs|x.Y
				op := *x
				op.X = lhs
				return name, &op
			}
		}
	case *ast.CallExpr:
		if name, _ := x.Fun.(*ast.Ident); name != nil {
			if len(x.Args) == 1 && x.Ellipsis == token.NoPos && (force || isTypeElem(x.Args[0])) {
				// x = name (x.Args[0])
				// (Note that the cmd/compile/internal/syntax parser does not care
				// about syntax tree fidelity and does not preserve parentheses here.)
				return name, &ast.ParenExpr{
					Lparen: x.Lparen,
					X:      x.Args[0],
					Rparen: x.Rparen,
				}
			}
		}
	}
	return nil, x
}

// isTypeElem reports whether x is a (possibly parenthesized) type element expression.
// The result is false if x could be a type element OR an ordinary (value) expression.
func isTypeElem(x ast.Expr) bool {
	switch x := x.(type) {
	case *ast.ArrayType, *ast.StructType, *ast.FuncType, *ast.InterfaceType, *ast.MapType, *ast.ChanType:
		return true
	case *ast.BinaryExpr:
		return isTypeElem(x.X) || isTypeElem(x.Y)
	case *ast.UnaryExpr:
		return x.Op == token.TILDE
	case *ast.ParenExpr:
		return isTypeElem(x.X)
	}
	return false
}

func (p *parser) parseGenDecl(keyword token.Token, f parseSpecFunction) *ast.GenDecl {
	if p.trace {
		defer un(trace(p, "GenDecl("+keyword.String()+")"))
	}

	doc := p.leadComment
	pos := p.expect(keyword)
	var lparen, rparen token.Pos
	var list []ast.Spec
	if p.tok == token.LPAREN {
		lparen = p.pos
		p.next()
		for iota := 0; p.tok != token.RPAREN && p.tok != token.EOF; iota++ {
			list = append(list, f(p.leadComment, keyword, iota))
		}
		rparen = p.expect(token.RPAREN)
		p.expectSemi()
	} else {
		list = append(list, f(nil, keyword, 0))
	}

	return &ast.GenDecl{
		Doc:    doc,
		TokPos: pos,
		Tok:    keyword,
		Lparen: lparen,
		Specs:  list,
		Rparen: rparen,
	}
}

func (p *parser) parseFuncDecl() *ast.FuncDecl {
	if p.trace {
		defer un(trace(p, "FunctionDecl"))
	}

	doc := p.leadComment
	pos := p.expect(token.FUNC)

	var recv *ast.FieldList
	if p.tok == token.LPAREN {
		recv = p.parseParameters(false)
	}

	ident := p.parseIdent()

	var tparams *ast.FieldList
	if p.tok == token.LBRACK {
		tparams = p.parseTypeParameters()
		if recv != nil && tparams != nil {
			// Method declarations do not have type parameters. We parse them for a
			// better error message and improved error recovery.
			p.error(tparams.Opening, "method must have no type parameters")
			tparams = nil
		}
	}
	params := p.parseParameters(false)
	results := p.parseParameters(true)

	var body *ast.BlockStmt
	switch p.tok {
	case token.LBRACE:
		body = p.parseBody()
		p.expectSemi()
	case token.SEMICOLON:
		p.next()
		if p.tok == token.LBRACE {
			// opening { of function declaration on next line
			p.error(p.pos, "unexpected semicolon or newline before {")
			body = p.parseBody()
			p.expectSemi()
		}
	default:
		p.expectSemi()
	}

	decl := &ast.FuncDecl{
		Doc:  doc,
		Recv: recv,
		Name: ident,
		Type: &ast.FuncType{
			Func:       pos,
			TypeParams: tparams,
			Params:     params,
			Results:    results,
		},
		Body: body,
	}
	return decl
}

func (p *parser) parseDecl(sync map[token.Token]bool) ast.Decl {
	if p.trace {
		defer un(trace(p, "Declaration"))
	}

	var f parseSpecFunction
	switch p.tok {
	case token.IMPORT:
		f = p.parseImportSpec

	case token.CONST, token.VAR:
		f = p.parseValueSpec

	case token.TYPE:
		f = p.parseTypeSpec

	case token.FUNC:
		return p.parseFuncDecl()

	case token.AT:
		// Decorator syntax:
		// 1. @decoratorName func foo() {...}
		// 2. @decoratorName(arg1, arg2) func foo() {...}
		// 3. @dec1 @dec2 func foo() {...} (stacked decorators)
		//
		// Transform to:
		// 1. var foo = decoratorName(func() {...})
		// 2. var foo = decoratorName(func() {...}, arg1, arg2)
		// 3. var foo = dec1(dec2(func() {...}))

		atPos := p.pos

		// Parse all decorators (support stacking)
		type decoratorInfo struct {
			name *ast.Ident
			args []ast.Expr // decorator arguments (if any)
		}
		var decorators []decoratorInfo

		for p.tok == token.AT {
			p.next() // consume '@'
			if p.tok != token.IDENT {
				p.errorExpected(p.pos, "decorator name")
				p.advance(sync)
				return &ast.BadDecl{From: p.pos, To: p.pos}
			}

			decName := p.parseIdent()
			var decArgs []ast.Expr

			// Check for decorator arguments: @decorator(arg1, arg2)
			if p.tok == token.LPAREN {
				p.next() // consume '('
				for p.tok != token.RPAREN && p.tok != token.EOF {
					decArgs = append(decArgs, p.parseExpr())
					if !p.atComma("decorator arguments", token.RPAREN) {
						break
					}
					p.next() // consume ','
				}
				p.expect(token.RPAREN)
			}

			decorators = append(decorators, decoratorInfo{
				name: decName,
				args: decArgs,
			})

			// Allow newline (semicolon) between decorators or before func
			if p.tok == token.SEMICOLON && p.lit == "\n" {
				p.next()
			}
		}

		if p.tok != token.FUNC {
			p.errorExpected(p.pos, "func after decorator")
			p.advance(sync)
			return &ast.BadDecl{From: p.pos, To: p.pos}
		}
		funcDecl := p.parseFuncDecl()

		// Create FuncLit from the function declaration
		funcLit := &ast.FuncLit{
			Type: funcDecl.Type,
			Body: funcDecl.Body,
		}

		// Apply decorators from innermost to outermost (reverse order)
		// @dec1 @dec2 func foo() => dec1(dec2(func()))
		var result ast.Expr = funcLit
		for i := len(decorators) - 1; i >= 0; i-- {
			dec := decorators[i]
			// Build argument list: (funcLit/previousResult, ...decoratorArgs)
			args := []ast.Expr{result}
			args = append(args, dec.args...)

			result = &ast.CallExpr{
				Fun:    dec.name,
				Lparen: dec.name.End(),
				Args:   args,
				Rparen: dec.name.End() + 1,
			}
		}

		// Create ValueSpec: foo = decorator(...)(func() {...})
		valueSpec := &ast.ValueSpec{
			Names:  []*ast.Ident{funcDecl.Name},
			Values: []ast.Expr{result},
		}

		// Create GenDecl: var foo = decorator(func() {...})
		return &ast.GenDecl{
			TokPos: atPos,
			Tok:    token.VAR,
			Specs:  []ast.Spec{valueSpec},
		}

	default:
		pos := p.pos
		p.errorExpected(pos, "declaration")
		p.advance(sync)
		return &ast.BadDecl{From: pos, To: p.pos}
	}

	return p.parseGenDecl(p.tok, f)
}

// ----------------------------------------------------------------------------
// Source files

func (p *parser) parseFile() *ast.File {
	if p.trace {
		defer un(trace(p, "File"))
	}

	// Don't bother parsing the rest if we had errors scanning the first token.
	// Likely not a Go source file at all.
	if p.errors.Len() != 0 {
		return nil
	}

	// package clause
	doc := p.leadComment
	pos := p.expect(token.PACKAGE)
	// Go spec: The package clause is not a declaration;
	// the package name does not appear in any scope.
	ident := p.parseIdent()
	if ident.Name == "_" && p.mode&DeclarationErrors != 0 {
		p.error(p.pos, "invalid package name _")
	}
	p.expectSemi()

	// Don't bother parsing the rest if we had errors parsing the package clause.
	// Likely not a Go source file at all.
	if p.errors.Len() != 0 {
		return nil
	}

	var decls []ast.Decl
	if p.mode&PackageClauseOnly == 0 {
		// import decls
		for p.tok == token.IMPORT {
			decls = append(decls, p.parseGenDecl(token.IMPORT, p.parseImportSpec))
		}

		if p.mode&ImportsOnly == 0 {
			// rest of package body
			prev := token.IMPORT
			for p.tok != token.EOF {
				// Continue to accept import declarations for error tolerance, but complain.
				if p.tok == token.IMPORT && prev != token.IMPORT {
					p.error(p.pos, "imports must appear before other declarations")
				}
				prev = p.tok

				decls = append(decls, p.parseDecl(declStart))
			}
		}
	}

	f := &ast.File{
		Doc:     doc,
		Package: pos,
		Name:    ident,
		Decls:   decls,
		// File{Start,End} are set by the defer in the caller.
		Imports:   p.imports,
		Comments:  p.comments,
		GoVersion: p.goVersion,
	}
	var declErr func(token.Pos, string)
	if p.mode&DeclarationErrors != 0 {
		declErr = p.error
	}
	if p.mode&SkipObjectResolution == 0 {
		resolveFile(f, p.file, declErr)
	}

	return f
}

// packIndexExpr returns an IndexExpr x[expr0] or IndexListExpr x[expr0, ...].
func packIndexExpr(x ast.Expr, lbrack token.Pos, exprs []ast.Expr, rbrack token.Pos) ast.Expr {
	switch len(exprs) {
	case 0:
		panic("internal error: packIndexExpr with empty expr slice")
	case 1:
		return &ast.IndexExpr{
			X:      x,
			Lbrack: lbrack,
			Index:  exprs[0],
			Rbrack: rbrack,
		}
	default:
		return &ast.IndexListExpr{
			X:       x,
			Lbrack:  lbrack,
			Indices: exprs,
			Rbrack:  rbrack,
		}
	}
}
