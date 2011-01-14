// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A parser for Go source files. Input may be provided in a variety of
// forms (see the various Parse* functions); the output is an abstract
// syntax tree (AST) representing the Go source. The parser is invoked
// through one of the Parse* functions.
//
package parser

import (
	"fmt"
	"go/ast"
	"go/scanner"
	"go/token"
)


// noPos is used when there is no corresponding source position for a token.
var noPos token.Position


// The mode parameter to the Parse* functions is a set of flags (or 0).
// They control the amount of source code parsed and other optional
// parser functionality.
//
const (
	PackageClauseOnly uint = 1 << iota // parsing stops after package clause
	ImportsOnly                        // parsing stops after import declarations
	ParseComments                      // parse comments and add them to AST
	Trace                              // print a trace of parsed productions
)


// The parser structure holds the parser's internal state.
type parser struct {
	file *token.File
	scanner.ErrorVector
	scanner scanner.Scanner

	// Tracing/debugging
	mode   uint // parsing mode
	trace  bool // == (mode & Trace != 0)
	indent uint // indentation used for tracing output

	// Comments
	comments    []*ast.CommentGroup
	leadComment *ast.CommentGroup // the last lead comment
	lineComment *ast.CommentGroup // the last line comment

	// Next token
	pos token.Pos   // token position
	tok token.Token // one token look-ahead
	lit []byte      // token literal

	// Non-syntactic parser control
	exprLev int // < 0: in control clause, >= 0: in expression
}


// scannerMode returns the scanner mode bits given the parser's mode bits.
func scannerMode(mode uint) uint {
	var m uint = scanner.InsertSemis
	if mode&ParseComments != 0 {
		m |= scanner.ScanComments
	}
	return m
}


func (p *parser) init(fset *token.FileSet, filename string, src []byte, mode uint) {
	p.file = fset.AddFile(filename, fset.Base(), len(src))
	p.scanner.Init(p.file, src, p, scannerMode(mode))
	p.mode = mode
	p.trace = mode&Trace != 0 // for convenience (p.trace is used frequently)
	p.next()
}


// ----------------------------------------------------------------------------
// Parsing support

func (p *parser) printTrace(a ...interface{}) {
	const dots = ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . " +
		". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . "
	const n = uint(len(dots))
	pos := p.file.Position(p.pos)
	fmt.Printf("%5d:%3d: ", pos.Line, pos.Column)
	i := 2 * p.indent
	for ; i > n; i -= n {
		fmt.Print(dots)
	}
	fmt.Print(dots[0:i])
	fmt.Println(a...)
}


func trace(p *parser, msg string) *parser {
	p.printTrace(msg, "(")
	p.indent++
	return p
}


// Usage pattern: defer un(trace(p, "..."));
func un(p *parser) {
	p.indent--
	p.printTrace(")")
}


// Advance to the next token.
func (p *parser) next0() {
	// Because of one-token look-ahead, print the previous token
	// when tracing as it provides a more readable output. The
	// very first token (!p.pos.IsValid()) is not initialized
	// (it is token.ILLEGAL), so don't print it .
	if p.trace && p.pos.IsValid() {
		s := p.tok.String()
		switch {
		case p.tok.IsLiteral():
			p.printTrace(s, string(p.lit))
		case p.tok.IsOperator(), p.tok.IsKeyword():
			p.printTrace("\"" + s + "\"")
		default:
			p.printTrace(s)
		}
	}

	p.pos, p.tok, p.lit = p.scanner.Scan()
}

// Consume a comment and return it and the line on which it ends.
func (p *parser) consumeComment() (comment *ast.Comment, endline int) {
	// /*-style comments may end on a different line than where they start.
	// Scan the comment for '\n' chars and adjust endline accordingly.
	endline = p.file.Line(p.pos)
	if p.lit[1] == '*' {
		for _, b := range p.lit {
			if b == '\n' {
				endline++
			}
		}
	}

	comment = &ast.Comment{p.pos, p.lit}
	p.next0()

	return
}


// Consume a group of adjacent comments, add it to the parser's
// comments list, and return it together with the line at which
// the last comment in the group ends. An empty line or non-comment
// token terminates a comment group.
//
func (p *parser) consumeCommentGroup() (comments *ast.CommentGroup, endline int) {
	var list []*ast.Comment
	endline = p.file.Line(p.pos)
	for p.tok == token.COMMENT && endline+1 >= p.file.Line(p.pos) {
		var comment *ast.Comment
		comment, endline = p.consumeComment()
		list = append(list, comment)
	}

	// add comment group to the comments list
	comments = &ast.CommentGroup{list}
	p.comments = append(p.comments, comments)

	return
}


// Advance to the next non-comment token. In the process, collect
// any comment groups encountered, and remember the last lead and
// and line comments.
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
//
func (p *parser) next() {
	p.leadComment = nil
	p.lineComment = nil
	line := p.file.Line(p.pos) // current line
	p.next0()

	if p.tok == token.COMMENT {
		var comment *ast.CommentGroup
		var endline int

		if p.file.Line(p.pos) == line {
			// The comment is on same line as previous token; it
			// cannot be a lead comment but may be a line comment.
			comment, endline = p.consumeCommentGroup()
			if p.file.Line(p.pos) != endline {
				// The next token is on a different line, thus
				// the last comment group is a line comment.
				p.lineComment = comment
			}
		}

		// consume successor comments, if any
		endline = -1
		for p.tok == token.COMMENT {
			comment, endline = p.consumeCommentGroup()
		}

		if endline+1 == p.file.Line(p.pos) {
			// The next token is following on the line immediately after the
			// comment group, thus the last comment group is a lead comment.
			p.leadComment = comment
		}
	}
}


func (p *parser) error(pos token.Pos, msg string) {
	p.Error(p.file.Position(pos), msg)
}


func (p *parser) errorExpected(pos token.Pos, msg string) {
	msg = "expected " + msg
	if pos == p.pos {
		// the error happened at the current position;
		// make the error message more specific
		if p.tok == token.SEMICOLON && p.lit[0] == '\n' {
			msg += ", found newline"
		} else {
			msg += ", found '" + p.tok.String() + "'"
			if p.tok.IsLiteral() {
				msg += " " + string(p.lit)
			}
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


func (p *parser) expectSemi() {
	if p.tok != token.RPAREN && p.tok != token.RBRACE {
		p.expect(token.SEMICOLON)
	}
}


// ----------------------------------------------------------------------------
// Identifiers

func (p *parser) parseIdent() *ast.Ident {
	pos := p.pos
	name := "_"
	if p.tok == token.IDENT {
		name = string(p.lit)
		p.next()
	} else {
		p.expect(token.IDENT) // use expect() error handling
	}
	return &ast.Ident{pos, name, nil}
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


// ----------------------------------------------------------------------------
// Types

func (p *parser) parseType() ast.Expr {
	if p.trace {
		defer un(trace(p, "Type"))
	}

	typ := p.tryType()

	if typ == nil {
		pos := p.pos
		p.errorExpected(pos, "type")
		p.next() // make progress
		return &ast.BadExpr{pos, p.pos}
	}

	return typ
}


func (p *parser) parseQualifiedIdent() ast.Expr {
	if p.trace {
		defer un(trace(p, "QualifiedIdent"))
	}

	var x ast.Expr = p.parseIdent()
	if p.tok == token.PERIOD {
		// first identifier is a package identifier
		p.next()
		sel := p.parseIdent()
		x = &ast.SelectorExpr{x, sel}
	}
	return x
}


func (p *parser) parseTypeName() ast.Expr {
	if p.trace {
		defer un(trace(p, "TypeName"))
	}

	return p.parseQualifiedIdent()
}


func (p *parser) parseArrayType(ellipsisOk bool) ast.Expr {
	if p.trace {
		defer un(trace(p, "ArrayType"))
	}

	lbrack := p.expect(token.LBRACK)
	var len ast.Expr
	if ellipsisOk && p.tok == token.ELLIPSIS {
		len = &ast.Ellipsis{p.pos, nil}
		p.next()
	} else if p.tok != token.RBRACK {
		len = p.parseExpr()
	}
	p.expect(token.RBRACK)
	elt := p.parseType()

	return &ast.ArrayType{lbrack, len, elt}
}


func (p *parser) makeIdentList(list []ast.Expr) []*ast.Ident {
	idents := make([]*ast.Ident, len(list))
	for i, x := range list {
		ident, isIdent := x.(*ast.Ident)
		if !isIdent {
			pos := x.(ast.Expr).Pos()
			p.errorExpected(pos, "identifier")
			ident = &ast.Ident{pos, "_", nil}
		}
		idents[i] = ident
	}
	return idents
}


func (p *parser) parseFieldDecl() *ast.Field {
	if p.trace {
		defer un(trace(p, "FieldDecl"))
	}

	doc := p.leadComment

	// fields
	list, typ := p.parseVarList(false)

	// optional tag
	var tag *ast.BasicLit
	if p.tok == token.STRING {
		tag = &ast.BasicLit{p.pos, p.tok, p.lit}
		p.next()
	}

	// analyze case
	var idents []*ast.Ident
	if typ != nil {
		// IdentifierList Type
		idents = p.makeIdentList(list)
	} else {
		// ["*"] TypeName (AnonymousField)
		typ = list[0] // we always have at least one element
		if n := len(list); n > 1 || !isTypeName(deref(typ)) {
			pos := typ.Pos()
			p.errorExpected(pos, "anonymous field")
			typ = &ast.BadExpr{pos, list[n-1].End()}
		}
	}

	p.expectSemi()

	return &ast.Field{doc, idents, typ, tag, p.lineComment}
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

	return &ast.StructType{pos, &ast.FieldList{lbrace, list, rbrace}, false}
}


func (p *parser) parsePointerType() *ast.StarExpr {
	if p.trace {
		defer un(trace(p, "PointerType"))
	}

	star := p.expect(token.MUL)
	base := p.parseType()

	return &ast.StarExpr{star, base}
}


func (p *parser) tryVarType(isParam bool) ast.Expr {
	if isParam && p.tok == token.ELLIPSIS {
		pos := p.pos
		p.next()
		typ := p.tryType() // don't use parseType so we can provide better error message
		if typ == nil {
			p.error(pos, "'...' parameter is missing type")
			typ = &ast.BadExpr{pos, p.pos}
		}
		if p.tok != token.RPAREN {
			p.error(pos, "can use '...' with last parameter type only")
		}
		return &ast.Ellipsis{pos, typ}
	}
	return p.tryType()
}


func (p *parser) parseVarType(isParam bool) ast.Expr {
	typ := p.tryVarType(isParam)
	if typ == nil {
		pos := p.pos
		p.errorExpected(pos, "type")
		p.next() // make progress
		typ = &ast.BadExpr{pos, p.pos}
	}
	return typ
}


func (p *parser) parseVarList(isParam bool) (list []ast.Expr, typ ast.Expr) {
	if p.trace {
		defer un(trace(p, "VarList"))
	}

	// a list of identifiers looks like a list of type names
	for {
		// parseVarType accepts any type (including parenthesized ones)
		// even though the syntax does not permit them here: we
		// accept them all for more robust parsing and complain
		// afterwards
		list = append(list, p.parseVarType(isParam))
		if p.tok != token.COMMA {
			break
		}
		p.next()
	}

	// if we had a list of identifiers, it must be followed by a type
	typ = p.tryVarType(isParam)

	return
}


func (p *parser) parseParameterList(ellipsisOk bool) (params []*ast.Field) {
	if p.trace {
		defer un(trace(p, "ParameterList"))
	}

	list, typ := p.parseVarList(ellipsisOk)
	if typ != nil {
		// IdentifierList Type
		idents := p.makeIdentList(list)
		params = append(params, &ast.Field{nil, idents, typ, nil, nil})
		if p.tok == token.COMMA {
			p.next()
		}

		for p.tok != token.RPAREN && p.tok != token.EOF {
			idents := p.parseIdentList()
			typ := p.parseVarType(ellipsisOk)
			params = append(params, &ast.Field{nil, idents, typ, nil, nil})
			if p.tok != token.COMMA {
				break
			}
			p.next()
		}

	} else {
		// Type { "," Type } (anonymous parameters)
		params = make([]*ast.Field, len(list))
		for i, x := range list {
			params[i] = &ast.Field{Type: x}
		}
	}

	return
}


func (p *parser) parseParameters(ellipsisOk bool) *ast.FieldList {
	if p.trace {
		defer un(trace(p, "Parameters"))
	}

	var params []*ast.Field
	lparen := p.expect(token.LPAREN)
	if p.tok != token.RPAREN {
		params = p.parseParameterList(ellipsisOk)
	}
	rparen := p.expect(token.RPAREN)

	return &ast.FieldList{lparen, params, rparen}
}


func (p *parser) parseResult() *ast.FieldList {
	if p.trace {
		defer un(trace(p, "Result"))
	}

	if p.tok == token.LPAREN {
		return p.parseParameters(false)
	}

	typ := p.tryType()
	if typ != nil {
		list := make([]*ast.Field, 1)
		list[0] = &ast.Field{Type: typ}
		return &ast.FieldList{List: list}
	}

	return nil
}


func (p *parser) parseSignature() (params, results *ast.FieldList) {
	if p.trace {
		defer un(trace(p, "Signature"))
	}

	params = p.parseParameters(true)
	results = p.parseResult()

	return
}


func (p *parser) parseFuncType() *ast.FuncType {
	if p.trace {
		defer un(trace(p, "FuncType"))
	}

	pos := p.expect(token.FUNC)
	params, results := p.parseSignature()

	return &ast.FuncType{pos, params, results}
}


func (p *parser) parseMethodSpec() *ast.Field {
	if p.trace {
		defer un(trace(p, "MethodSpec"))
	}

	doc := p.leadComment
	var idents []*ast.Ident
	var typ ast.Expr
	x := p.parseQualifiedIdent()
	if ident, isIdent := x.(*ast.Ident); isIdent && p.tok == token.LPAREN {
		// method
		idents = []*ast.Ident{ident}
		params, results := p.parseSignature()
		typ = &ast.FuncType{token.NoPos, params, results}
	} else {
		// embedded interface
		typ = x
	}
	p.expectSemi()

	return &ast.Field{doc, idents, typ, nil, p.lineComment}
}


func (p *parser) parseInterfaceType() *ast.InterfaceType {
	if p.trace {
		defer un(trace(p, "InterfaceType"))
	}

	pos := p.expect(token.INTERFACE)
	lbrace := p.expect(token.LBRACE)
	var list []*ast.Field
	for p.tok == token.IDENT {
		list = append(list, p.parseMethodSpec())
	}
	rbrace := p.expect(token.RBRACE)

	return &ast.InterfaceType{pos, &ast.FieldList{lbrace, list, rbrace}, false}
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

	return &ast.MapType{pos, key, value}
}


func (p *parser) parseChanType() *ast.ChanType {
	if p.trace {
		defer un(trace(p, "ChanType"))
	}

	pos := p.pos
	dir := ast.SEND | ast.RECV
	if p.tok == token.CHAN {
		p.next()
		if p.tok == token.ARROW {
			p.next()
			dir = ast.SEND
		}
	} else {
		p.expect(token.ARROW)
		p.expect(token.CHAN)
		dir = ast.RECV
	}
	value := p.parseType()

	return &ast.ChanType{pos, dir, value}
}


func (p *parser) tryRawType(ellipsisOk bool) ast.Expr {
	switch p.tok {
	case token.IDENT:
		return p.parseTypeName()
	case token.LBRACK:
		return p.parseArrayType(ellipsisOk)
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
		return &ast.ParenExpr{lparen, typ, rparen}
	}

	// no type found
	return nil
}


func (p *parser) tryType() ast.Expr { return p.tryRawType(false) }


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
	rbrace := p.expect(token.RBRACE)

	return &ast.BlockStmt{lbrace, list, rbrace}
}


func (p *parser) parseBlockStmt() *ast.BlockStmt {
	if p.trace {
		defer un(trace(p, "BlockStmt"))
	}

	lbrace := p.expect(token.LBRACE)
	list := p.parseStmtList()
	rbrace := p.expect(token.RBRACE)

	return &ast.BlockStmt{lbrace, list, rbrace}
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

	return &ast.FuncLit{typ, body}
}


// parseOperand may return an expression or a raw type (incl. array
// types of the form [...]T. Callers must verify the result.
//
func (p *parser) parseOperand() ast.Expr {
	if p.trace {
		defer un(trace(p, "Operand"))
	}

	switch p.tok {
	case token.IDENT:
		return p.parseIdent()

	case token.INT, token.FLOAT, token.IMAG, token.CHAR, token.STRING:
		x := &ast.BasicLit{p.pos, p.tok, p.lit}
		p.next()
		return x

	case token.LPAREN:
		lparen := p.pos
		p.next()
		p.exprLev++
		x := p.parseExpr()
		p.exprLev--
		rparen := p.expect(token.RPAREN)
		return &ast.ParenExpr{lparen, x, rparen}

	case token.FUNC:
		return p.parseFuncTypeOrLit()

	default:
		t := p.tryRawType(true) // could be type for composite literal or conversion
		if t != nil {
			return t
		}
	}

	pos := p.pos
	p.errorExpected(pos, "operand")
	p.next() // make progress
	return &ast.BadExpr{pos, p.pos}
}


func (p *parser) parseSelectorOrTypeAssertion(x ast.Expr) ast.Expr {
	if p.trace {
		defer un(trace(p, "SelectorOrTypeAssertion"))
	}

	p.expect(token.PERIOD)
	if p.tok == token.IDENT {
		// selector
		sel := p.parseIdent()
		return &ast.SelectorExpr{x, sel}
	}

	// type assertion
	p.expect(token.LPAREN)
	var typ ast.Expr
	if p.tok == token.TYPE {
		// type switch: typ == nil
		p.next()
	} else {
		typ = p.parseType()
	}
	p.expect(token.RPAREN)

	return &ast.TypeAssertExpr{x, typ}
}


func (p *parser) parseIndexOrSlice(x ast.Expr) ast.Expr {
	if p.trace {
		defer un(trace(p, "IndexOrSlice"))
	}

	lbrack := p.expect(token.LBRACK)
	p.exprLev++
	var low, high ast.Expr
	isSlice := false
	if p.tok != token.COLON {
		low = p.parseExpr()
	}
	if p.tok == token.COLON {
		isSlice = true
		p.next()
		if p.tok != token.RBRACK {
			high = p.parseExpr()
		}
	}
	p.exprLev--
	rbrack := p.expect(token.RBRACK)

	if isSlice {
		return &ast.SliceExpr{x, lbrack, low, high, rbrack}
	}
	return &ast.IndexExpr{x, lbrack, low, rbrack}
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
		list = append(list, p.parseExpr())
		if p.tok == token.ELLIPSIS {
			ellipsis = p.pos
			p.next()
		}
		if p.tok != token.COMMA {
			break
		}
		p.next()
	}
	p.exprLev--
	rparen := p.expect(token.RPAREN)

	return &ast.CallExpr{fun, lparen, list, ellipsis, rparen}
}


func (p *parser) parseElement(keyOk bool) ast.Expr {
	if p.trace {
		defer un(trace(p, "Element"))
	}

	if p.tok == token.LBRACE {
		return p.parseLiteralValue(nil)
	}

	x := p.parseExpr()
	if keyOk && p.tok == token.COLON {
		colon := p.pos
		p.next()
		x = &ast.KeyValueExpr{x, colon, p.parseElement(false)}
	}
	return x
}


func (p *parser) parseElementList() (list []ast.Expr) {
	if p.trace {
		defer un(trace(p, "ElementList"))
	}

	for p.tok != token.RBRACE && p.tok != token.EOF {
		list = append(list, p.parseElement(true))
		if p.tok != token.COMMA {
			break
		}
		p.next()
	}

	return
}


func (p *parser) parseLiteralValue(typ ast.Expr) ast.Expr {
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
	rbrace := p.expect(token.RBRACE)
	return &ast.CompositeLit{typ, lbrace, elts, rbrace}
}


// checkExpr checks that x is an expression (and not a type).
func (p *parser) checkExpr(x ast.Expr) ast.Expr {
	switch t := unparen(x).(type) {
	case *ast.BadExpr:
	case *ast.Ident:
	case *ast.BasicLit:
	case *ast.FuncLit:
	case *ast.CompositeLit:
	case *ast.ParenExpr:
		panic("unreachable")
	case *ast.SelectorExpr:
	case *ast.IndexExpr:
	case *ast.SliceExpr:
	case *ast.TypeAssertExpr:
		if t.Type == nil {
			// the form X.(type) is only allowed in type switch expressions
			p.errorExpected(x.Pos(), "expression")
			x = &ast.BadExpr{x.Pos(), x.End()}
		}
	case *ast.CallExpr:
	case *ast.StarExpr:
	case *ast.UnaryExpr:
		if t.Op == token.RANGE {
			// the range operator is only allowed at the top of a for statement
			p.errorExpected(x.Pos(), "expression")
			x = &ast.BadExpr{x.Pos(), x.End()}
		}
	case *ast.BinaryExpr:
	default:
		// all other nodes are not proper expressions
		p.errorExpected(x.Pos(), "expression")
		x = &ast.BadExpr{x.Pos(), x.End()}
	}
	return x
}


// isTypeName returns true iff x is a (qualified) TypeName.
func isTypeName(x ast.Expr) bool {
	switch t := x.(type) {
	case *ast.BadExpr:
	case *ast.Ident:
	case *ast.SelectorExpr:
		_, isIdent := t.X.(*ast.Ident)
		return isIdent
	default:
		return false // all other nodes are not type names
	}
	return true
}


// isLiteralType returns true iff x is a legal composite literal type.
func isLiteralType(x ast.Expr) bool {
	switch t := x.(type) {
	case *ast.BadExpr:
	case *ast.Ident:
	case *ast.SelectorExpr:
		_, isIdent := t.X.(*ast.Ident)
		return isIdent
	case *ast.ArrayType:
	case *ast.StructType:
	case *ast.MapType:
	default:
		return false // all other nodes are not legal composite literal types
	}
	return true
}


// If x is of the form *T, deref returns T, otherwise it returns x.
func deref(x ast.Expr) ast.Expr {
	if p, isPtr := x.(*ast.StarExpr); isPtr {
		x = p.X
	}
	return x
}


// If x is of the form (T), unparen returns unparen(T), otherwise it returns x.
func unparen(x ast.Expr) ast.Expr {
	if p, isParen := x.(*ast.ParenExpr); isParen {
		x = unparen(p.X)
	}
	return x
}


// checkExprOrType checks that x is an expression or a type
// (and not a raw type such as [...]T).
//
func (p *parser) checkExprOrType(x ast.Expr) ast.Expr {
	switch t := unparen(x).(type) {
	case *ast.ParenExpr:
		panic("unreachable")
	case *ast.UnaryExpr:
		if t.Op == token.RANGE {
			// the range operator is only allowed at the top of a for statement
			p.errorExpected(x.Pos(), "expression")
			x = &ast.BadExpr{x.Pos(), x.End()}
		}
	case *ast.ArrayType:
		if len, isEllipsis := t.Len.(*ast.Ellipsis); isEllipsis {
			p.error(len.Pos(), "expected array length, found '...'")
			x = &ast.BadExpr{x.Pos(), x.End()}
		}
	}

	// all other nodes are expressions or types
	return x
}


func (p *parser) parsePrimaryExpr() ast.Expr {
	if p.trace {
		defer un(trace(p, "PrimaryExpr"))
	}

	x := p.parseOperand()
L:
	for {
		switch p.tok {
		case token.PERIOD:
			x = p.parseSelectorOrTypeAssertion(p.checkExpr(x))
		case token.LBRACK:
			x = p.parseIndexOrSlice(p.checkExpr(x))
		case token.LPAREN:
			x = p.parseCallOrConversion(p.checkExprOrType(x))
		case token.LBRACE:
			if isLiteralType(x) && (p.exprLev >= 0 || !isTypeName(x)) {
				x = p.parseLiteralValue(x)
			} else {
				break L
			}
		default:
			break L
		}
	}

	return x
}


func (p *parser) parseUnaryExpr() ast.Expr {
	if p.trace {
		defer un(trace(p, "UnaryExpr"))
	}

	switch p.tok {
	case token.ADD, token.SUB, token.NOT, token.XOR, token.AND, token.RANGE:
		pos, op := p.pos, p.tok
		p.next()
		x := p.parseUnaryExpr()
		return &ast.UnaryExpr{pos, op, p.checkExpr(x)}

	case token.ARROW:
		// channel type or receive expression
		pos := p.pos
		p.next()
		if p.tok == token.CHAN {
			p.next()
			value := p.parseType()
			return &ast.ChanType{pos, ast.RECV, value}
		}

		x := p.parseUnaryExpr()
		return &ast.UnaryExpr{pos, token.ARROW, p.checkExpr(x)}

	case token.MUL:
		// pointer type or unary "*" expression
		pos := p.pos
		p.next()
		x := p.parseUnaryExpr()
		return &ast.StarExpr{pos, p.checkExprOrType(x)}
	}

	return p.parsePrimaryExpr()
}


func (p *parser) parseBinaryExpr(prec1 int) ast.Expr {
	if p.trace {
		defer un(trace(p, "BinaryExpr"))
	}

	x := p.parseUnaryExpr()
	for prec := p.tok.Precedence(); prec >= prec1; prec-- {
		for p.tok.Precedence() == prec {
			pos, op := p.pos, p.tok
			p.next()
			y := p.parseBinaryExpr(prec + 1)
			x = &ast.BinaryExpr{p.checkExpr(x), pos, op, p.checkExpr(y)}
		}
	}

	return x
}


// TODO(gri): parseExpr may return a type or even a raw type ([..]int) -
//            should reject when a type/raw type is obviously not allowed
func (p *parser) parseExpr() ast.Expr {
	if p.trace {
		defer un(trace(p, "Expression"))
	}

	return p.parseBinaryExpr(token.LowestPrec + 1)
}


// ----------------------------------------------------------------------------
// Statements

func (p *parser) parseSimpleStmt(labelOk bool) ast.Stmt {
	if p.trace {
		defer un(trace(p, "SimpleStmt"))
	}

	x := p.parseExprList()

	switch p.tok {
	case token.COLON:
		// labeled statement
		colon := p.pos
		p.next()
		if labelOk && len(x) == 1 {
			if label, isIdent := x[0].(*ast.Ident); isIdent {
				return &ast.LabeledStmt{label, colon, p.parseStmt()}
			}
		}
		p.error(x[0].Pos(), "illegal label declaration")
		return &ast.BadStmt{x[0].Pos(), colon + 1}

	case
		token.DEFINE, token.ASSIGN, token.ADD_ASSIGN,
		token.SUB_ASSIGN, token.MUL_ASSIGN, token.QUO_ASSIGN,
		token.REM_ASSIGN, token.AND_ASSIGN, token.OR_ASSIGN,
		token.XOR_ASSIGN, token.SHL_ASSIGN, token.SHR_ASSIGN, token.AND_NOT_ASSIGN:
		// assignment statement
		pos, tok := p.pos, p.tok
		p.next()
		y := p.parseExprList()
		return &ast.AssignStmt{x, pos, tok, y}
	}

	if len(x) > 1 {
		p.error(x[0].Pos(), "only one expression allowed")
		// continue with first expression
	}

	if p.tok == token.INC || p.tok == token.DEC {
		// increment or decrement
		s := &ast.IncDecStmt{x[0], p.pos, p.tok}
		p.next() // consume "++" or "--"
		return s
	}

	// expression
	return &ast.ExprStmt{x[0]}
}


func (p *parser) parseCallExpr() *ast.CallExpr {
	x := p.parseExpr()
	if call, isCall := x.(*ast.CallExpr); isCall {
		return call
	}
	p.errorExpected(x.Pos(), "function/method call")
	return nil
}


func (p *parser) parseGoStmt() ast.Stmt {
	if p.trace {
		defer un(trace(p, "GoStmt"))
	}

	pos := p.expect(token.GO)
	call := p.parseCallExpr()
	p.expectSemi()
	if call == nil {
		return &ast.BadStmt{pos, pos + 2} // len("go")
	}

	return &ast.GoStmt{pos, call}
}


func (p *parser) parseDeferStmt() ast.Stmt {
	if p.trace {
		defer un(trace(p, "DeferStmt"))
	}

	pos := p.expect(token.DEFER)
	call := p.parseCallExpr()
	p.expectSemi()
	if call == nil {
		return &ast.BadStmt{pos, pos + 5} // len("defer")
	}

	return &ast.DeferStmt{pos, call}
}


func (p *parser) parseReturnStmt() *ast.ReturnStmt {
	if p.trace {
		defer un(trace(p, "ReturnStmt"))
	}

	pos := p.pos
	p.expect(token.RETURN)
	var x []ast.Expr
	if p.tok != token.SEMICOLON && p.tok != token.RBRACE {
		x = p.parseExprList()
	}
	p.expectSemi()

	return &ast.ReturnStmt{pos, x}
}


func (p *parser) parseBranchStmt(tok token.Token) *ast.BranchStmt {
	if p.trace {
		defer un(trace(p, "BranchStmt"))
	}

	s := &ast.BranchStmt{p.pos, tok, nil}
	p.expect(tok)
	if tok != token.FALLTHROUGH && p.tok == token.IDENT {
		s.Label = p.parseIdent()
	}
	p.expectSemi()

	return s
}


func (p *parser) makeExpr(s ast.Stmt) ast.Expr {
	if s == nil {
		return nil
	}
	if es, isExpr := s.(*ast.ExprStmt); isExpr {
		return p.checkExpr(es.X)
	}
	p.error(s.Pos(), "expected condition, found simple statement")
	return &ast.BadExpr{s.Pos(), s.End()}
}


func (p *parser) parseControlClause(isForStmt bool) (s1, s2, s3 ast.Stmt) {
	if p.tok != token.LBRACE {
		prevLev := p.exprLev
		p.exprLev = -1

		if p.tok != token.SEMICOLON {
			s1 = p.parseSimpleStmt(false)
		}
		if p.tok == token.SEMICOLON {
			p.next()
			if p.tok != token.LBRACE && p.tok != token.SEMICOLON {
				s2 = p.parseSimpleStmt(false)
			}
			if isForStmt {
				// for statements have a 3rd section
				p.expectSemi()
				if p.tok != token.LBRACE {
					s3 = p.parseSimpleStmt(false)
				}
			}
		} else {
			s1, s2 = nil, s1
		}

		p.exprLev = prevLev
	}

	return s1, s2, s3
}


func (p *parser) parseIfStmt() *ast.IfStmt {
	if p.trace {
		defer un(trace(p, "IfStmt"))
	}

	pos := p.expect(token.IF)
	s1, s2, _ := p.parseControlClause(false)
	body := p.parseBlockStmt()
	var else_ ast.Stmt
	if p.tok == token.ELSE {
		p.next()
		else_ = p.parseStmt()
	} else {
		p.expectSemi()
	}

	return &ast.IfStmt{pos, s1, p.makeExpr(s2), body, else_}
}


func (p *parser) parseCaseClause() *ast.CaseClause {
	if p.trace {
		defer un(trace(p, "CaseClause"))
	}

	// SwitchCase
	pos := p.pos
	var x []ast.Expr
	if p.tok == token.CASE {
		p.next()
		x = p.parseExprList()
	} else {
		p.expect(token.DEFAULT)
	}

	colon := p.expect(token.COLON)
	body := p.parseStmtList()

	return &ast.CaseClause{pos, x, colon, body}
}


func (p *parser) parseTypeList() (list []ast.Expr) {
	if p.trace {
		defer un(trace(p, "TypeList"))
	}

	list = append(list, p.parseType())
	for p.tok == token.COMMA {
		p.next()
		list = append(list, p.parseType())
	}

	return
}


func (p *parser) parseTypeCaseClause() *ast.TypeCaseClause {
	if p.trace {
		defer un(trace(p, "TypeCaseClause"))
	}

	// TypeSwitchCase
	pos := p.pos
	var types []ast.Expr
	if p.tok == token.CASE {
		p.next()
		types = p.parseTypeList()
	} else {
		p.expect(token.DEFAULT)
	}

	colon := p.expect(token.COLON)
	body := p.parseStmtList()

	return &ast.TypeCaseClause{pos, types, colon, body}
}


func isExprSwitch(s ast.Stmt) bool {
	if s == nil {
		return true
	}
	if e, ok := s.(*ast.ExprStmt); ok {
		if a, ok := e.X.(*ast.TypeAssertExpr); ok {
			return a.Type != nil // regular type assertion
		}
		return true
	}
	return false
}


func (p *parser) parseSwitchStmt() ast.Stmt {
	if p.trace {
		defer un(trace(p, "SwitchStmt"))
	}

	pos := p.expect(token.SWITCH)
	s1, s2, _ := p.parseControlClause(false)

	if isExprSwitch(s2) {
		lbrace := p.expect(token.LBRACE)
		var list []ast.Stmt
		for p.tok == token.CASE || p.tok == token.DEFAULT {
			list = append(list, p.parseCaseClause())
		}
		rbrace := p.expect(token.RBRACE)
		body := &ast.BlockStmt{lbrace, list, rbrace}
		p.expectSemi()
		return &ast.SwitchStmt{pos, s1, p.makeExpr(s2), body}
	}

	// type switch
	// TODO(gri): do all the checks!
	lbrace := p.expect(token.LBRACE)
	var list []ast.Stmt
	for p.tok == token.CASE || p.tok == token.DEFAULT {
		list = append(list, p.parseTypeCaseClause())
	}
	rbrace := p.expect(token.RBRACE)
	p.expectSemi()
	body := &ast.BlockStmt{lbrace, list, rbrace}
	return &ast.TypeSwitchStmt{pos, s1, s2, body}
}


func (p *parser) parseCommClause() *ast.CommClause {
	if p.trace {
		defer un(trace(p, "CommClause"))
	}

	// CommCase
	pos := p.pos
	var tok token.Token
	var lhs, rhs ast.Expr
	if p.tok == token.CASE {
		p.next()
		if p.tok == token.ARROW {
			// RecvExpr without assignment
			rhs = p.parseExpr()
		} else {
			// SendExpr or RecvExpr
			rhs = p.parseExpr()
			if p.tok == token.ASSIGN || p.tok == token.DEFINE {
				// RecvExpr with assignment
				tok = p.tok
				p.next()
				lhs = rhs
				if p.tok == token.ARROW {
					rhs = p.parseExpr()
				} else {
					p.expect(token.ARROW) // use expect() error handling
				}
			}
			// else SendExpr
		}
	} else {
		p.expect(token.DEFAULT)
	}

	colon := p.expect(token.COLON)
	body := p.parseStmtList()

	return &ast.CommClause{pos, tok, lhs, rhs, colon, body}
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
	body := &ast.BlockStmt{lbrace, list, rbrace}

	return &ast.SelectStmt{pos, body}
}


func (p *parser) parseForStmt() ast.Stmt {
	if p.trace {
		defer un(trace(p, "ForStmt"))
	}

	pos := p.expect(token.FOR)
	s1, s2, s3 := p.parseControlClause(true)
	body := p.parseBlockStmt()
	p.expectSemi()

	if as, isAssign := s2.(*ast.AssignStmt); isAssign {
		// possibly a for statement with a range clause; check assignment operator
		if as.Tok != token.ASSIGN && as.Tok != token.DEFINE {
			p.errorExpected(as.TokPos, "'=' or ':='")
			return &ast.BadStmt{pos, body.End()}
		}
		// check lhs
		var key, value ast.Expr
		switch len(as.Lhs) {
		case 2:
			key, value = as.Lhs[0], as.Lhs[1]
		case 1:
			key = as.Lhs[0]
		default:
			p.errorExpected(as.Lhs[0].Pos(), "1 or 2 expressions")
			return &ast.BadStmt{pos, body.End()}
		}
		// check rhs
		if len(as.Rhs) != 1 {
			p.errorExpected(as.Rhs[0].Pos(), "1 expressions")
			return &ast.BadStmt{pos, body.End()}
		}
		if rhs, isUnary := as.Rhs[0].(*ast.UnaryExpr); isUnary && rhs.Op == token.RANGE {
			// rhs is range expression; check lhs
			return &ast.RangeStmt{pos, key, value, as.TokPos, as.Tok, rhs.X, body}
		} else {
			p.errorExpected(s2.Pos(), "range clause")
			return &ast.BadStmt{pos, body.End()}
		}
	} else {
		// regular for statement
		return &ast.ForStmt{pos, s1, p.makeExpr(s2), s3, body}
	}

	panic("unreachable")
}


func (p *parser) parseStmt() (s ast.Stmt) {
	if p.trace {
		defer un(trace(p, "Statement"))
	}

	switch p.tok {
	case token.CONST, token.TYPE, token.VAR:
		s = &ast.DeclStmt{p.parseDecl()}
	case
		// tokens that may start a top-level expression
		token.IDENT, token.INT, token.FLOAT, token.CHAR, token.STRING, token.FUNC, token.LPAREN, // operand
		token.LBRACK, token.STRUCT, // composite type
		token.MUL, token.AND, token.ARROW, token.ADD, token.SUB, token.XOR: // unary operators
		s = p.parseSimpleStmt(true)
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
		s = &ast.EmptyStmt{p.pos}
		p.next()
	case token.RBRACE:
		// a semicolon may be omitted before a closing "}"
		s = &ast.EmptyStmt{p.pos}
	default:
		// no statement found
		pos := p.pos
		p.errorExpected(pos, "statement")
		p.next() // make progress
		s = &ast.BadStmt{pos, p.pos}
	}

	return
}


// ----------------------------------------------------------------------------
// Declarations

type parseSpecFunction func(p *parser, doc *ast.CommentGroup) ast.Spec


func parseImportSpec(p *parser, doc *ast.CommentGroup) ast.Spec {
	if p.trace {
		defer un(trace(p, "ImportSpec"))
	}

	var ident *ast.Ident
	if p.tok == token.PERIOD {
		ident = &ast.Ident{p.pos, ".", nil}
		p.next()
	} else if p.tok == token.IDENT {
		ident = p.parseIdent()
	}

	var path *ast.BasicLit
	if p.tok == token.STRING {
		path = &ast.BasicLit{p.pos, p.tok, p.lit}
		p.next()
	} else {
		p.expect(token.STRING) // use expect() error handling
	}
	p.expectSemi()

	return &ast.ImportSpec{doc, ident, path, p.lineComment}
}


func parseConstSpec(p *parser, doc *ast.CommentGroup) ast.Spec {
	if p.trace {
		defer un(trace(p, "ConstSpec"))
	}

	idents := p.parseIdentList()
	typ := p.tryType()
	var values []ast.Expr
	if typ != nil || p.tok == token.ASSIGN {
		p.expect(token.ASSIGN)
		values = p.parseExprList()
	}
	p.expectSemi()

	return &ast.ValueSpec{doc, idents, typ, values, p.lineComment}
}


func parseTypeSpec(p *parser, doc *ast.CommentGroup) ast.Spec {
	if p.trace {
		defer un(trace(p, "TypeSpec"))
	}

	ident := p.parseIdent()
	typ := p.parseType()
	p.expectSemi()

	return &ast.TypeSpec{doc, ident, typ, p.lineComment}
}


func parseVarSpec(p *parser, doc *ast.CommentGroup) ast.Spec {
	if p.trace {
		defer un(trace(p, "VarSpec"))
	}

	idents := p.parseIdentList()
	typ := p.tryType()
	var values []ast.Expr
	if typ == nil || p.tok == token.ASSIGN {
		p.expect(token.ASSIGN)
		values = p.parseExprList()
	}
	p.expectSemi()

	return &ast.ValueSpec{doc, idents, typ, values, p.lineComment}
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
		for p.tok != token.RPAREN && p.tok != token.EOF {
			list = append(list, f(p, p.leadComment))
		}
		rparen = p.expect(token.RPAREN)
		p.expectSemi()
	} else {
		list = append(list, f(p, nil))
	}

	return &ast.GenDecl{doc, pos, keyword, lparen, list, rparen}
}


func (p *parser) parseReceiver() *ast.FieldList {
	if p.trace {
		defer un(trace(p, "Receiver"))
	}

	pos := p.pos
	par := p.parseParameters(false)

	// must have exactly one receiver
	if par.NumFields() != 1 {
		p.errorExpected(pos, "exactly one receiver")
		// TODO determine a better range for BadExpr below
		par.List = []*ast.Field{&ast.Field{Type: &ast.BadExpr{pos, pos}}}
		return par
	}

	// recv type must be of the form ["*"] identifier
	recv := par.List[0]
	base := deref(recv.Type)
	if _, isIdent := base.(*ast.Ident); !isIdent {
		p.errorExpected(base.Pos(), "(unqualified) identifier")
		par.List = []*ast.Field{&ast.Field{Type: &ast.BadExpr{recv.Pos(), recv.End()}}}
	}

	return par
}


func (p *parser) parseFuncDecl() *ast.FuncDecl {
	if p.trace {
		defer un(trace(p, "FunctionDecl"))
	}

	doc := p.leadComment
	pos := p.expect(token.FUNC)

	var recv *ast.FieldList
	if p.tok == token.LPAREN {
		recv = p.parseReceiver()
	}

	ident := p.parseIdent()
	params, results := p.parseSignature()

	var body *ast.BlockStmt
	if p.tok == token.LBRACE {
		body = p.parseBody()
	}
	p.expectSemi()

	return &ast.FuncDecl{doc, recv, ident, &ast.FuncType{pos, params, results}, body}
}


func (p *parser) parseDecl() ast.Decl {
	if p.trace {
		defer un(trace(p, "Declaration"))
	}

	var f parseSpecFunction
	switch p.tok {
	case token.CONST:
		f = parseConstSpec

	case token.TYPE:
		f = parseTypeSpec

	case token.VAR:
		f = parseVarSpec

	case token.FUNC:
		return p.parseFuncDecl()

	default:
		pos := p.pos
		p.errorExpected(pos, "declaration")
		p.next() // make progress
		decl := &ast.BadDecl{pos, p.pos}
		return decl
	}

	return p.parseGenDecl(p.tok, f)
}


func (p *parser) parseDeclList() (list []ast.Decl) {
	if p.trace {
		defer un(trace(p, "DeclList"))
	}

	for p.tok != token.EOF {
		list = append(list, p.parseDecl())
	}

	return
}


// ----------------------------------------------------------------------------
// Source files

func (p *parser) parseFile() *ast.File {
	if p.trace {
		defer un(trace(p, "File"))
	}

	// package clause
	doc := p.leadComment
	pos := p.expect(token.PACKAGE)
	ident := p.parseIdent()
	p.expectSemi()

	var decls []ast.Decl

	// Don't bother parsing the rest if we had errors already.
	// Likely not a Go source file at all.

	if p.ErrorCount() == 0 && p.mode&PackageClauseOnly == 0 {
		// import decls
		for p.tok == token.IMPORT {
			decls = append(decls, p.parseGenDecl(token.IMPORT, parseImportSpec))
		}

		if p.mode&ImportsOnly == 0 {
			// rest of package body
			for p.tok != token.EOF {
				decls = append(decls, p.parseDecl())
			}
		}
	}

	return &ast.File{doc, pos, ident, decls, p.comments}
}
