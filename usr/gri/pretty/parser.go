// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A parser for Go source text. The input is a stream of lexical tokens
// provided via the Scanner interface. The output is an abstract syntax
// tree (AST) representing the Go source. The parser is invoked by calling
// Parse.
//
package parser

import (
	"fmt";
	"vector";
	"token";
	"ast";
)


// An implementation of a Scanner must be provided to the Parser.
// The parser calls Scan() repeatedly until token.EOF is returned.
// Scan must return the current token position pos, the token value
// tok, and the corresponding token literal string lit; lit can be
// undefined/nil unless the token is a literal (i.e., tok.IsLiteral()
// is true).
//
type Scanner interface {
	Scan() (pos token.Position, tok token.Token, lit []byte);
}


// An implementation of an ErrorHandler may be provided to the parser.
// If a syntax error is encountered and a handler was installed, Error
// is called with a position and an error message. The position points
// to the beginning of the offending token.
//
type ErrorHandler interface {
	Error(pos token.Position, msg string);
}


// The following flags control optional parser functionality. A set of
// flags (or 0) must be provided as a parameter to the Parse function.
//
const (
	Trace = 1 << iota;  // print a trace of parsed productions
)


type interval struct {
	beg, end int;
}


// The parser structure holds the parser's internal state.
type parser struct {
	scanner Scanner;
	err ErrorHandler;  // nil if no handler installed
	errorCount int;

	// Tracing/debugging
	trace bool;
	indent uint;

	// Comments
	comments vector.Vector;  // list of collected, unassociated comments
	last_doc interval;  // last comments interval of consecutive comments

	// The next token
	pos token.Position;  // token position
	tok token.Token;  // one token look-ahead
	lit []byte;  // token literal

	// Non-syntactic parser control
	opt_semi bool;  // true if semicolon separator is optional in statement list
	expr_lev int;  // < 0: in control clause, >= 0: in expression
};


// When we don't have a position use nopos.
// TODO make sure we always have a position.
var nopos token.Position;


// ----------------------------------------------------------------------------
// Helper functions

func unreachable() {
	panic("unreachable");
}


// ----------------------------------------------------------------------------
// Parsing support

func (p *parser) printIndent() {
	i := p.indent;
	// reduce printing time by a factor of 2 or more
	for ; i > 10; i -= 10 {
		fmt.Printf(". . . . . . . . . . ");
	}
	for ; i > 0; i-- {
		fmt.Printf(". ");
	}
}


func trace(p *parser, msg string) *parser {
	p.printIndent();
	fmt.Printf("%s (\n", msg);
	p.indent++;
	return p;
}


func un/*trace*/(p *parser) {
	p.indent--;
	p.printIndent();
	fmt.Printf(")\n");
}


func (p *parser) next0() {
	p.pos, p.tok, p.lit = p.scanner.Scan();
	p.opt_semi = false;

	if p.trace {
		p.printIndent();
		switch p.tok {
		case token.IDENT, token.INT, token.FLOAT, token.CHAR, token.STRING:
			fmt.Printf("%d:%d: %s = %s\n", p.pos.Line, p.pos.Column, p.tok.String(), p.lit);
		case token.LPAREN:
			// don't print '(' - screws up selection in terminal window
			fmt.Printf("%d:%d: LPAREN\n", p.pos.Line, p.pos.Column);
		case token.RPAREN:
			// don't print ')' - screws up selection in terminal window
			fmt.Printf("%d:%d: RPAREN\n", p.pos.Line, p.pos.Column);
		default:
			fmt.Printf("%d:%d: %s\n", p.pos.Line, p.pos.Column, p.tok.String());
		}
	}
}


// Collect a comment in the parser's comment list and return the line
// on which the comment ends.
func (p *parser) collectComment() int {
	// For /*-style comments, the comment may end on a different line.
	// Scan the comment for '\n' chars and adjust the end line accordingly.
	// (Note that the position of the next token may be even further down
	// as there may be more whitespace lines after the comment.)
	endline := p.pos.Line;
	if p.lit[1] == '*' {
		for i, b := range p.lit {
			if b == '\n' {
				endline++;
			}
		}
	}
	p.comments.Push(&ast.Comment{p.pos, p.lit, endline});
	p.next0();
	
	return endline;
}


func (p *parser) getComments() interval {
	// group adjacent comments, an empty line terminates a group
	beg := p.comments.Len();
	endline := p.pos.Line;
	for p.tok == token.COMMENT && endline+1 >= p.pos.Line {
		endline = p.collectComment();
	}
	end := p.comments.Len();
	return interval {beg, end};
}


func (p *parser) next() {
	p.next0();
	p.last_doc = interval{0, 0};
	for p.tok == token.COMMENT {
		p.last_doc = p.getComments();
	}
}


func (p *parser) error(pos token.Position, msg string) {
	if p.err != nil {
		p.err.Error(pos, msg);
	}
	p.errorCount++;
}


func (p *parser) expect(tok token.Token) token.Position {
	if p.tok != tok {
		msg := "expected '" + tok.String() + "', found '" + p.tok.String() + "'";
		if p.tok.IsLiteral() {
			msg += " " + string(p.lit);
		}
		p.error(p.pos, msg);
	}
	pos := p.pos;
	p.next();  // make progress in any case
	return pos;
}


func (p *parser) getDoc() ast.Comments {
	doc := p.last_doc;
	n := doc.end - doc.beg;
	
	if n <= 0 || p.comments.At(doc.end - 1).(*ast.Comment).EndLine + 1 < p.pos.Line {
		// no comments or empty line between last comment and current token;
		// do not use as documentation
		return nil;
	}

	// found immediately adjacent comment interval;
	// use as documentation
	c := make(ast.Comments, n);
	for i := 0; i < n; i++ {
		c[i] = p.comments.At(doc.beg + i).(*ast.Comment);
	}

	// remove comments from the general list
	p.comments.Cut(doc.beg, doc.end);

	return c;
}


// ----------------------------------------------------------------------------
// Common productions

func (p *parser) tryType() ast.Expr;
func (p *parser) parseExpression(prec int) ast.Expr;
func (p *parser) parseStatement() ast.Stmt;
func (p *parser) parseDeclaration() ast.Decl;


func (p *parser) parseIdent() *ast.Ident {
	if p.trace {
		defer un(trace(p, "Ident"));
	}

	if p.tok == token.IDENT {
		x := &ast.Ident{p.pos, p.lit};
		p.next();
		return x;
	}
	p.expect(token.IDENT);  // use expect() error handling

	return &ast.Ident{p.pos, [0]byte{}};
}


func (p *parser) parseIdentList(x ast.Expr) []*ast.Ident {
	if p.trace {
		defer un(trace(p, "IdentList"));
	}

	list := vector.New(0);
	if x == nil {
		x = p.parseIdent();
	}
	list.Push(x);
	for p.tok == token.COMMA {
		p.next();
		list.Push(p.parseIdent());
	}

	// convert vector
	idents := make([]*ast.Ident, list.Len());
	for i := 0; i < list.Len(); i++ {
		idents[i] = list.At(i).(*ast.Ident);
	}

	return idents;
}


func (p *parser) parseExpressionList() []ast.Expr {
	if p.trace {
		defer un(trace(p, "ExpressionList"));
	}

	list := vector.New(0);
	list.Push(p.parseExpression(1));
	for p.tok == token.COMMA {
		p.next();
		list.Push(p.parseExpression(1));
	}

	// convert list
	exprs := make([]ast.Expr, list.Len());
	for i := 0; i < list.Len(); i++ {
		exprs[i] = list.At(i).(ast.Expr);
	}

	return exprs;
}


// ----------------------------------------------------------------------------
// Types

func (p *parser) parseType() ast.Expr {
	if p.trace {
		defer un(trace(p, "Type"));
	}

	typ := p.tryType();
	if typ == nil {
		p.error(p.pos, "type expected");
		typ = &ast.BadExpr{p.pos};
	}

	return typ;
}


func (p *parser) parseQualifiedIdent() ast.Expr {
	if p.trace {
		defer un(trace(p, "QualifiedIdent"));
	}

	var x ast.Expr = p.parseIdent();
	for p.tok == token.PERIOD {
		p.next();
		sel := p.parseIdent();
		x = &ast.SelectorExpr{x, sel};
	}
	return x;
}


func (p *parser) parseTypeName() ast.Expr {
	if p.trace {
		defer un(trace(p, "TypeName"));
	}

	return p.parseQualifiedIdent();
}


func (p *parser) parseArrayType() *ast.ArrayType {
	if p.trace {
		defer un(trace(p, "ArrayType"));
	}

	lbrack := p.expect(token.LBRACK);
	var len ast.Expr;
	if p.tok == token.ELLIPSIS {
		len = &ast.Ellipsis{p.pos};
		p.next();
	} else if p.tok != token.RBRACK {
		len = p.parseExpression(1);
	}
	p.expect(token.RBRACK);
	elt := p.parseType();

	return &ast.ArrayType{lbrack, len, elt};
}


func (p *parser) parseChannelType() *ast.ChannelType {
	if p.trace {
		defer un(trace(p, "ChannelType"));
	}

	pos := p.pos;
	dir := ast.SEND | ast.RECV;
	if p.tok == token.CHAN {
		p.next();
		if p.tok == token.ARROW {
			p.next();
			dir = ast.SEND;
		}
	} else {
		p.expect(token.ARROW);
		p.expect(token.CHAN);
		dir = ast.RECV;
	}
	value := p.parseType();

	return &ast.ChannelType{pos, dir, value};
}


func (p *parser) tryParameterType() ast.Expr {
	if p.tok == token.ELLIPSIS {
		x := &ast.Ellipsis{p.pos};
		p.next();
		return x;
	}
	return p.tryType();
}


func (p *parser) parseParameterType() ast.Expr {
	typ := p.tryParameterType();
	if typ == nil {
		p.error(p.pos, "type expected");
		typ = &ast.BadExpr{p.pos};
	}

	return typ;
}


func (p *parser) parseParameterDecl(ellipsis_ok bool) (*vector.Vector, ast.Expr) {
	if p.trace {
		defer un(trace(p, "ParameterDecl"));
	}

	// a list of identifiers looks like a list of type names
	list := vector.New(0);
	for {
		// TODO do not allow ()'s here
		list.Push(p.parseParameterType());
		if p.tok == token.COMMA {
			p.next();
		} else {
			break;
		}
	}

	// if we had a list of identifiers, it must be followed by a type
	typ := p.tryParameterType();

	return list, typ;
}


func (p *parser) parseParameterList(ellipsis_ok bool) []*ast.Field {
	if p.trace {
		defer un(trace(p, "ParameterList"));
	}

	list, typ := p.parseParameterDecl(false);
	if typ != nil {
		// IdentifierList Type
		// convert list of identifiers into []*Ident
		idents := make([]*ast.Ident, list.Len());
		for i := 0; i < list.Len(); i++ {
			idents[i] = list.At(i).(*ast.Ident);
		}
		list.Init(0);
		list.Push(&ast.Field{nil, idents, typ, nil});

		for p.tok == token.COMMA {
			p.next();
			idents := p.parseIdentList(nil);
			typ := p.parseParameterType();
			list.Push(&ast.Field{nil, idents, typ, nil});
		}

	} else {
		// Type { "," Type }
		// convert list of types into list of *Param
		for i := 0; i < list.Len(); i++ {
			list.Set(i, &ast.Field{nil, nil, list.At(i).(ast.Expr), nil});
		}
	}

	// convert list
	params := make([]*ast.Field, list.Len());
	for i := 0; i < list.Len(); i++ {
		params[i] = list.At(i).(*ast.Field);
	}

	return params;
}


// TODO make sure Go spec is updated
func (p *parser) parseParameters(ellipsis_ok bool) []*ast.Field {
	if p.trace {
		defer un(trace(p, "Parameters"));
	}

	var params []*ast.Field;
	p.expect(token.LPAREN);
	if p.tok != token.RPAREN {
		params = p.parseParameterList(ellipsis_ok);
	}
	p.expect(token.RPAREN);

	return params;
}


func (p *parser) parseResult() []*ast.Field {
	if p.trace {
		defer un(trace(p, "Result"));
	}

	var results []*ast.Field;
	if p.tok == token.LPAREN {
		results = p.parseParameters(false);
	} else if p.tok != token.FUNC {
		typ := p.tryType();
		if typ != nil {
			results = make([]*ast.Field, 1);
			results[0] = &ast.Field{nil, nil, typ, nil};
		}
	}

	return results;
}


// Function types
//
// (params)
// (params) type
// (params) (results)

func (p *parser) parseSignature() (params []*ast.Field, results []*ast.Field) {
	if p.trace {
		defer un(trace(p, "Signature"));
	}

	params = p.parseParameters(true);  // TODO find better solution
	results = p.parseResult();

	return params, results;
}


func (p *parser) parseFunctionType() *ast.FunctionType {
	if p.trace {
		defer un(trace(p, "FunctionType"));
	}

	pos := p.expect(token.FUNC);
	params, results := p.parseSignature();

	return &ast.FunctionType{pos, params, results};
}


func (p *parser) parseMethodSpec() *ast.Field {
	if p.trace {
		defer un(trace(p, "MethodSpec"));
	}

	doc := p.getDoc();
	var idents []*ast.Ident;
	var typ ast.Expr;
	x := p.parseQualifiedIdent();
	if tmp, is_ident := x.(*ast.Ident); is_ident && (p.tok == token.COMMA || p.tok == token.LPAREN) {
		// method(s)
		idents = p.parseIdentList(x);
		params, results := p.parseSignature();
		typ = &ast.FunctionType{nopos, params, results};
	} else {
		// embedded interface
		typ = x;
	}

	return &ast.Field{doc, idents, typ, nil};
}


func (p *parser) parseInterfaceType() *ast.InterfaceType {
	if p.trace {
		defer un(trace(p, "InterfaceType"));
	}

	pos := p.expect(token.INTERFACE);
	var lbrace, rbrace token.Position;
	var methods []*ast.Field;
	if p.tok == token.LBRACE {
		lbrace = p.pos;
		p.next();

		list := vector.New(0);
		for p.tok == token.IDENT {
			list.Push(p.parseMethodSpec());
			if p.tok != token.RBRACE {
				p.expect(token.SEMICOLON);
			}
		}

		rbrace = p.expect(token.RBRACE);
		p.opt_semi = true;

		// convert vector
		methods = make([]*ast.Field, list.Len());
		for i := list.Len() - 1; i >= 0; i-- {
			methods[i] = list.At(i).(*ast.Field);
		}
	}

	return &ast.InterfaceType{pos, lbrace, methods, rbrace};
}


func (p *parser) parseMapType() *ast.MapType {
	if p.trace {
		defer un(trace(p, "MapType"));
	}

	pos := p.expect(token.MAP);
	p.expect(token.LBRACK);
	key := p.parseType();
	p.expect(token.RBRACK);
	value := p.parseType();

	return &ast.MapType{pos, key, value};
}


func (p *parser) parseStringList(x *ast.StringLit) []*ast.StringLit

func (p *parser) parseFieldDecl() *ast.Field {
	if p.trace {
		defer un(trace(p, "FieldDecl"));
	}

	doc := p.getDoc();

	// a list of identifiers looks like a list of type names
	list := vector.New(0);
	for {
		// TODO do not allow ()'s here
		list.Push(p.parseType());
		if p.tok == token.COMMA {
			p.next();
		} else {
			break;
		}
	}

	// if we had a list of identifiers, it must be followed by a type
	typ := p.tryType();

	// optional tag
	var tag []*ast.StringLit;
	if p.tok == token.STRING {
		tag = p.parseStringList(nil);
	}

	// analyze case
	var idents []*ast.Ident;
	if typ != nil {
		// non-empty identifier list followed by a type
		idents = make([]*ast.Ident, list.Len());
		for i := 0; i < list.Len(); i++ {
			if ident, is_ident := list.At(i).(*ast.Ident); is_ident {
				idents[i] = ident;
			} else {
				p.error(list.At(i).(ast.Expr).Pos(), "identifier expected");
			}
		}
	} else {
		// anonymous field
		if list.Len() == 1 {
			// TODO should do more checks here
			typ = list.At(0).(ast.Expr);
		} else {
			p.error(p.pos, "anonymous field expected");
		}
	}

	return &ast.Field{doc, idents, typ, tag};
}


func (p *parser) parseStructType() *ast.StructType {
	if p.trace {
		defer un(trace(p, "StructType"));
	}

	pos := p.expect(token.STRUCT);
	var lbrace, rbrace token.Position;
	var fields []*ast.Field;
	if p.tok == token.LBRACE {
		lbrace = p.pos;
		p.next();

		list := vector.New(0);
		for p.tok != token.RBRACE && p.tok != token.EOF {
			list.Push(p.parseFieldDecl());
			if p.tok == token.SEMICOLON {
				p.next();
			} else {
				break;
			}
		}
		if p.tok == token.SEMICOLON {
			p.next();
		}

		rbrace = p.expect(token.RBRACE);
		p.opt_semi = true;

		// convert vector
		fields = make([]*ast.Field, list.Len());
		for i := list.Len() - 1; i >= 0; i-- {
			fields[i] = list.At(i).(*ast.Field);
		}
	}

	return &ast.StructType{pos, lbrace, fields, rbrace};
}


func (p *parser) parsePointerType() *ast.StarExpr {
	if p.trace {
		defer un(trace(p, "PointerType"));
	}

	star := p.expect(token.MUL);
	base := p.parseType();

	return &ast.StarExpr{star, base};
}


func (p *parser) tryType() ast.Expr {
	if p.trace {
		defer un(trace(p, "Type (try)"));
	}

	switch p.tok {
	case token.IDENT: return p.parseTypeName();
	case token.LBRACK: return p.parseArrayType();
	case token.CHAN, token.ARROW: return p.parseChannelType();
	case token.INTERFACE: return p.parseInterfaceType();
	case token.FUNC: return p.parseFunctionType();
	case token.MAP: return p.parseMapType();
	case token.STRUCT: return p.parseStructType();
	case token.MUL: return p.parsePointerType();
	case token.LPAREN:
		lparen := p.pos;
		p.next();
		x := p.parseType();
		rparen := p.expect(token.RPAREN);
		return &ast.ParenExpr{lparen, x, rparen};
	}

	// no type found
	return nil;
}


// ----------------------------------------------------------------------------
// Blocks

func asStmtList(list *vector.Vector) []ast.Stmt {
	stats := make([]ast.Stmt, list.Len());
	for i := 0; i < list.Len(); i++ {
		stats[i] = list.At(i).(ast.Stmt);
	}
	return stats;
}


func (p *parser) parseStatementList() []ast.Stmt {
	if p.trace {
		defer un(trace(p, "StatementList"));
	}

	list := vector.New(0);
	expect_semi := false;
	for p.tok != token.CASE && p.tok != token.DEFAULT && p.tok != token.RBRACE && p.tok != token.EOF {
		if expect_semi {
			p.expect(token.SEMICOLON);
			expect_semi = false;
		}
		list.Push(p.parseStatement());
		if p.tok == token.SEMICOLON {
			p.next();
		} else if p.opt_semi {
			p.opt_semi = false;  // "consume" optional semicolon
		} else {
			expect_semi = true;
		}
	}
	
	return asStmtList(list);
}


func (p *parser) parseBlockStmt() *ast.BlockStmt {
	if p.trace {
		defer un(trace(p, "compositeStmt"));
	}

	lbrace := p.expect(token.LBRACE);
	list := p.parseStatementList();
	rbrace := p.expect(token.RBRACE);
	p.opt_semi = true;

	return &ast.BlockStmt{lbrace, list, rbrace};
}


// ----------------------------------------------------------------------------
// Expressions

func (p *parser) parseFunctionLit() ast.Expr {
	if p.trace {
		defer un(trace(p, "FunctionLit"));
	}

	typ := p.parseFunctionType();
	p.expr_lev++;
	body := p.parseBlockStmt();
	p.expr_lev--;

	return &ast.FunctionLit{typ, body};
}


func (p *parser) parseStringList(x *ast.StringLit) []*ast.StringLit {
	if p.trace {
		defer un(trace(p, "StringList"));
	}

	list := vector.New(0);
	if x != nil {
		list.Push(x);
	}
	
	for p.tok == token.STRING {
		list.Push(&ast.StringLit{p.pos, p.lit});
		p.next();
	}

	// convert list
	strings := make([]*ast.StringLit, list.Len());
	for i := 0; i < list.Len(); i++ {
		strings[i] = list.At(i).(*ast.StringLit);
	}
	
	return strings;
}


func (p *parser) parseOperand() ast.Expr {
	if p.trace {
		defer un(trace(p, "Operand"));
	}

	switch p.tok {
	case token.IDENT:
		return p.parseIdent();

	case token.INT:
		x := &ast.IntLit{p.pos, p.lit};
		p.next();
		return x;

	case token.FLOAT:
		x := &ast.FloatLit{p.pos, p.lit};
		p.next();
		return x;

	case token.CHAR:
		x := &ast.CharLit{p.pos, p.lit};
		p.next();
		return x;

	case token.STRING:
		x := &ast.StringLit{p.pos, p.lit};
		p.next();
		if p.tok == token.STRING {
			return &ast.StringList{p.parseStringList(x)};
		}
		return x;

	case token.LPAREN:
		lparen := p.pos;
		p.next();
		p.expr_lev++;
		x := p.parseExpression(1);
		p.expr_lev--;
		rparen := p.expect(token.RPAREN);
		return &ast.ParenExpr{lparen, x, rparen};

	case token.FUNC:
		return p.parseFunctionLit();

	default:
		t := p.tryType();
		if t != nil {
			return t;
		} else {
			p.error(p.pos, "operand expected");
			p.next();  // make progress
		}
	}

	return &ast.BadExpr{p.pos};
}


func (p *parser) parseSelectorOrTypeAssertion(x ast.Expr) ast.Expr {
	if p.trace {
		defer un(trace(p, "SelectorOrTypeAssertion"));
	}

	p.expect(token.PERIOD);
	if p.tok == token.IDENT {
		// selector
		sel := p.parseIdent();
		return &ast.SelectorExpr{x, sel};
		
	} else {
		// type assertion
		p.expect(token.LPAREN);
		var typ ast.Expr;
		if p.tok == token.TYPE {
			// special case for type switch syntax
			typ = &ast.Ident{p.pos, p.lit};
			p.next();
		} else {
			typ = p.parseType();
		}
		p.expect(token.RPAREN);
		return &ast.TypeAssertExpr{x, typ};
	}

	unreachable();
	return nil;
}


func (p *parser) parseIndexOrSlice(x ast.Expr) ast.Expr {
	if p.trace {
		defer un(trace(p, "IndexOrSlice"));
	}

	p.expect(token.LBRACK);
	p.expr_lev++;
	index := p.parseExpression(1);
	p.expr_lev--;

	if p.tok == token.RBRACK {
		// index
		p.next();
		return &ast.IndexExpr{x, index};
	}
	
	// slice
	p.expect(token.COLON);
	p.expr_lev++;
	end := p.parseExpression(1);
	p.expr_lev--;
	p.expect(token.RBRACK);
	return &ast.SliceExpr{x, index, end};
}


func (p *parser) parseCall(fun ast.Expr) *ast.CallExpr {
	if p.trace {
		defer un(trace(p, "Call"));
	}

	lparen := p.expect(token.LPAREN);
	var args []ast.Expr;
	if p.tok != token.RPAREN {
		args = p.parseExpressionList();
	}
	rparen := p.expect(token.RPAREN);
	return &ast.CallExpr{fun, lparen, args, rparen};
}


func (p *parser) parseElementList() []ast.Expr {
	if p.trace {
		defer un(trace(p, "ElementList"));
	}

	list := vector.New(0);
	singles := true;
	for p.tok != token.RBRACE {
		x := p.parseExpression(0);
		if list.Len() == 0 {
			// first element determines syntax for remaining elements
			if t, is_binary := x.(*ast.BinaryExpr); is_binary && t.Op == token.COLON {
				singles = false;
			}
		} else {
			// not the first element - check syntax
			if singles {
				if t, is_binary := x.(*ast.BinaryExpr); is_binary && t.Op == token.COLON {
					p.error(t.X.Pos(), "single value expected; found pair");
				}
			} else {
				if t, is_binary := x.(*ast.BinaryExpr); !is_binary || t.Op != token.COLON {
					p.error(x.Pos(), "key:value pair expected; found single value");
				}
			}
		}

		list.Push(x);

		if p.tok == token.COMMA {
			p.next();
		} else {
			break;
		}
	}
	
	// convert list
	elts := make([]ast.Expr, list.Len());
	for i := 0; i < list.Len(); i++ {
		elts[i] = list.At(i).(ast.Expr);
	}
	
	return elts;
}


func (p *parser) parseCompositeLit(typ ast.Expr) ast.Expr {
	if p.trace {
		defer un(trace(p, "CompositeLit"));
	}

	lbrace := p.expect(token.LBRACE);
	var elts []ast.Expr;
	if p.tok != token.RBRACE {
		elts = p.parseElementList();
	}
	rbrace := p.expect(token.RBRACE);
	return &ast.CompositeLit{typ, lbrace, elts, rbrace};
}


func (p *parser) parsePrimaryExpr() ast.Expr {
	if p.trace {
		defer un(trace(p, "PrimaryExpr"));
	}

	x := p.parseOperand();
	for {
		switch p.tok {
		case token.PERIOD: x = p.parseSelectorOrTypeAssertion(x);
		case token.LBRACK: x = p.parseIndexOrSlice(x);
		case token.LPAREN: x = p.parseCall(x);
		case token.LBRACE:
			if p.expr_lev >= 0 {
				x = p.parseCompositeLit(x);
			} else {
				return x;
			}
		default:
			return x;
		}
	}

	unreachable();
	return nil;
}


func (p *parser) parseUnaryExpr() ast.Expr {
	if p.trace {
		defer un(trace(p, "UnaryExpr"));
	}

	switch p.tok {
	case token.ADD, token.SUB, token.NOT, token.XOR, token.ARROW, token.AND, token.RANGE:
		pos, tok := p.pos, p.tok;
		p.next();
		x := p.parseUnaryExpr();
		return &ast.UnaryExpr{pos, tok, x};

	case token.MUL:
		// unary "*" expression or pointer type
		pos := p.pos;
		p.next();
		x := p.parseUnaryExpr();
		return &ast.StarExpr{pos, x};
	}

	return p.parsePrimaryExpr();
}


func (p *parser) parseBinaryExpr(prec1 int) ast.Expr {
	if p.trace {
		defer un(trace(p, "BinaryExpr"));
	}

	x := p.parseUnaryExpr();
	for prec := p.tok.Precedence(); prec >= prec1; prec-- {
		for p.tok.Precedence() == prec {
			pos, tok := p.pos, p.tok;
			p.next();
			y := p.parseBinaryExpr(prec + 1);
			x = &ast.BinaryExpr{x, pos, tok, y};
		}
	}

	return x;
}


func (p *parser) parseExpression(prec int) ast.Expr {
	if p.trace {
		defer un(trace(p, "Expression"));
	}

	if prec < 0 {
		panic("precedence must be >= 0");
	}

	return p.parseBinaryExpr(prec);
}


// ----------------------------------------------------------------------------
// Statements


func (p *parser) parseSimpleStmt() ast.Stmt {
	if p.trace {
		defer un(trace(p, "SimpleStmt"));
	}

	x := p.parseExpressionList();

	switch p.tok {
	case token.COLON:
		// labeled statement
		p.expect(token.COLON);
		if len(x) == 1 {
			if label, is_ident := x[0].(*ast.Ident); is_ident {
				return &ast.LabeledStmt{label, p.parseStatement()};
			}
		}
		p.error(x[0].Pos(), "illegal label declaration");
		return &ast.BadStmt{x[0].Pos()};

	case
		token.DEFINE, token.ASSIGN, token.ADD_ASSIGN,
		token.SUB_ASSIGN, token.MUL_ASSIGN, token.QUO_ASSIGN,
		token.REM_ASSIGN, token.AND_ASSIGN, token.OR_ASSIGN,
		token.XOR_ASSIGN, token.SHL_ASSIGN, token.SHR_ASSIGN:
		// assignment statement
		pos, tok := p.pos, p.tok;
		p.next();
		y := p.parseExpressionList();
		if len(x) > 1 && len(y) > 1 && len(x) != len(y) {
			p.error(x[0].Pos(), "arity of lhs doesn't match rhs");
		}
		return &ast.AssignStmt{x, pos, tok, y};
	}

	if len(x) > 1 {
		p.error(x[0].Pos(), "only one expression allowed");
		// continue with first expression
	}

	if p.tok == token.INC || p.tok == token.DEC {
		// increment or decrement
		s := &ast.IncDecStmt{x[0], p.tok};
		p.next();  // consume "++" or "--"
		return s;
	}

	// expression
	return &ast.ExprStmt{x[0]};
}


func (p *parser) parseCallExpr() *ast.CallExpr {
	x := p.parseExpression(1);
	if call, is_call := x.(*ast.CallExpr); is_call {
		return call;
	}
	p.error(x.Pos(), "expected function/method call");
	return nil;
}


func (p *parser) parseGoStmt() ast.Stmt {
	if p.trace {
		defer un(trace(p, "GoStmt"));
	}

	pos := p.expect(token.GO);
	call := p.parseCallExpr();
	if call != nil {
		return &ast.GoStmt{pos, call};
	}
	return &ast.BadStmt{pos};
}


func (p *parser) parseDeferStmt() ast.Stmt {
	if p.trace {
		defer un(trace(p, "DeferStmt"));
	}

	pos := p.expect(token.DEFER);
	call := p.parseCallExpr();
	if call != nil {
		return &ast.DeferStmt{pos, call};
	}
	return &ast.BadStmt{pos};
}


func (p *parser) parseReturnStmt() *ast.ReturnStmt {
	if p.trace {
		defer un(trace(p, "ReturnStmt"));
	}

	pos := p.pos;
	p.expect(token.RETURN);
	var x []ast.Expr;
	if p.tok != token.SEMICOLON && p.tok != token.RBRACE {
		x = p.parseExpressionList();
	}

	return &ast.ReturnStmt{pos, x};
}


func (p *parser) parseBranchStmt(tok token.Token) *ast.BranchStmt {
	if p.trace {
		defer un(trace(p, "BranchStmt"));
	}

	s := &ast.BranchStmt{p.pos, tok, nil};
	p.expect(tok);
	if tok != token.FALLTHROUGH && p.tok == token.IDENT {
		s.Label = p.parseIdent();
	}

	return s;
}


func (p *parser) isExpr(s ast.Stmt) bool {
	if s == nil {
		return true;
	}
	dummy, is_expr := s.(*ast.ExprStmt);
	return is_expr;
}


func (p *parser) asExpr(s ast.Stmt) ast.Expr {
	if s == nil {
		return nil;
	}
	if es, is_expr := s.(*ast.ExprStmt); is_expr {
		return es.X;
	}
	p.error(s.Pos(), "condition expected; found simple statement");
	return &ast.BadExpr{s.Pos()};
}


func (p *parser) parseControlClause(isForStmt bool) (s1, s2, s3 ast.Stmt) {
	if p.trace {
		defer un(trace(p, "ControlClause"));
	}

	if p.tok != token.LBRACE {
		prev_lev := p.expr_lev;
		p.expr_lev = -1;

		if p.tok != token.SEMICOLON {
			s1 = p.parseSimpleStmt();
		}
		if p.tok == token.SEMICOLON {
			p.next();
			if p.tok != token.LBRACE && p.tok != token.SEMICOLON {
				s2 = p.parseSimpleStmt();
			}
			if isForStmt {
				// for statements have a 3rd section
				p.expect(token.SEMICOLON);
				if p.tok != token.LBRACE {
					s3 = p.parseSimpleStmt();
				}
			}
		} else {
			s1, s2 = nil, s1;
		}
		
		p.expr_lev = prev_lev;
	}

	return s1, s2, s3;
}


func (p *parser) parseIfStmt() *ast.IfStmt {
	if p.trace {
		defer un(trace(p, "IfStmt"));
	}

	pos := p.expect(token.IF);
	s1, s2, dummy := p.parseControlClause(false);
	body := p.parseBlockStmt();
	var else_ ast.Stmt;
	if p.tok == token.ELSE {
		p.next();
		else_ = p.parseStatement();
	}

	return &ast.IfStmt{pos, s1, p.asExpr(s2), body, else_};
}


func (p *parser) parseCaseClause() *ast.CaseClause {
	if p.trace {
		defer un(trace(p, "CaseClause"));
	}

	// SwitchCase
	pos := p.pos;
	var x []ast.Expr;
	if p.tok == token.CASE {
		p.next();
		x = p.parseExpressionList();
	} else {
		p.expect(token.DEFAULT);
	}
	
	colon := p.expect(token.COLON);
	body := p.parseStatementList();

	return &ast.CaseClause{pos, x, colon, body};
}


func (p *parser) parseTypeCaseClause() *ast.TypeCaseClause {
	if p.trace {
		defer un(trace(p, "CaseClause"));
	}

	// TypeSwitchCase
	pos := p.pos;
	var typ ast.Expr;
	if p.tok == token.CASE {
		p.next();
		typ = p.parseType();
	} else {
		p.expect(token.DEFAULT);
	}

	colon := p.expect(token.COLON);
	body := p.parseStatementList();

	return &ast.TypeCaseClause{pos, typ, colon, body};
}


func (p *parser) parseSwitchStmt() ast.Stmt {
	if p.trace {
		defer un(trace(p, "SwitchStmt"));
	}

	pos := p.expect(token.SWITCH);
	s1, s2, dummy := p.parseControlClause(false);

	if p.isExpr(s2) {
		// expression switch
		lbrace := p.expect(token.LBRACE);
		cases := vector.New(0);
		for p.tok == token.CASE || p.tok == token.DEFAULT {
			cases.Push(p.parseCaseClause());
		}
		rbrace := p.expect(token.RBRACE);
		p.opt_semi = true;
		body := &ast.BlockStmt{lbrace, asStmtList(cases), rbrace};
		return &ast.SwitchStmt{pos, s1, p.asExpr(s2), body};

	} else {
		// type switch
		// TODO do all the checks!
		lbrace := p.expect(token.LBRACE);
		cases := vector.New(0);
		for p.tok == token.CASE || p.tok == token.DEFAULT {
			cases.Push(p.parseTypeCaseClause());
		}
		rbrace := p.expect(token.RBRACE);
		p.opt_semi = true;
		body := &ast.BlockStmt{lbrace, asStmtList(cases), rbrace};
		return &ast.TypeSwitchStmt{pos, s1, s2, body};
	}

	unreachable();
	return nil;
}


func (p *parser) parseCommClause() *ast.CommClause {
	if p.trace {
		defer un(trace(p, "CommClause"));
	}

	// CommCase
	pos := p.pos;
	var tok token.Token;
	var lhs, rhs ast.Expr;
	if p.tok == token.CASE {
		p.next();
		if p.tok == token.ARROW {
			// RecvExpr without assignment
			rhs = p.parseExpression(1);
		} else {
			// SendExpr or RecvExpr
			rhs = p.parseExpression(1);
			if p.tok == token.ASSIGN || p.tok == token.DEFINE {
				// RecvExpr with assignment
				tok = p.tok;
				p.next();
				lhs = rhs;
				if p.tok == token.ARROW {
					rhs = p.parseExpression(1);
				} else {
					p.expect(token.ARROW);  // use expect() error handling
				}
			}
			// else SendExpr
		}
	} else {
		p.expect(token.DEFAULT);
	}

	colon := p.expect(token.COLON);
	body := p.parseStatementList();

	return &ast.CommClause{pos, tok, lhs, rhs, colon, body};
}


func (p *parser) parseSelectStmt() *ast.SelectStmt {
	if p.trace {
		defer un(trace(p, "SelectStmt"));
	}

	pos := p.expect(token.SELECT);
	lbrace := p.expect(token.LBRACE);
	cases := vector.New(0);
	for p.tok == token.CASE || p.tok == token.DEFAULT {
		cases.Push(p.parseCommClause());
	}
	rbrace := p.expect(token.RBRACE);
	p.opt_semi = true;
	body := &ast.BlockStmt{lbrace, asStmtList(cases), rbrace};

	return &ast.SelectStmt{pos, body};
}


func (p *parser) parseForStmt() ast.Stmt {
	if p.trace {
		defer un(trace(p, "ForStmt"));
	}

	pos := p.expect(token.FOR);
	s1, s2, s3 := p.parseControlClause(true);
	body := p.parseBlockStmt();

	if as, is_as := s2.(*ast.AssignStmt); is_as {
		// possibly a for statement with a range clause; check assignment operator
		if as.Tok != token.ASSIGN && as.Tok != token.DEFINE {
			p.error(as.TokPos, "'=' or ':=' expected");
			return &ast.BadStmt{pos};
		}
		// check lhs
		var key, value ast.Expr;
		switch len(as.Lhs) {
		case 2:
			value = as.Lhs[1];
			fallthrough;
		case 1:
			key = as.Lhs[0];
		default:
			p.error(as.Lhs[0].Pos(), "expected 1 or 2 expressions");
			return &ast.BadStmt{pos};
		}
		// check rhs
		if len(as.Rhs) != 1 {
			p.error(as.Rhs[0].Pos(), "expected 1 expressions");
			return &ast.BadStmt{pos};
		}
		if rhs, is_unary := as.Rhs[0].(*ast.UnaryExpr); is_unary && rhs.Op == token.RANGE {
			// rhs is range expression; check lhs
			return &ast.RangeStmt{pos, key, value, as.TokPos, as.Tok, rhs.X, body}
		} else {
			p.error(s2.Pos(), "range clause expected");
			return &ast.BadStmt{pos};
		}
	} else {
		// regular for statement
		return &ast.ForStmt{pos, s1, p.asExpr(s2), s3, body};
	}
	
	unreachable();
	return nil;
}


func (p *parser) parseStatement() ast.Stmt {
	if p.trace {
		defer un(trace(p, "Statement"));
	}

	switch p.tok {
	case token.CONST, token.TYPE, token.VAR:
		return &ast.DeclStmt{p.parseDeclaration()};
	case
		// tokens that may start a top-level expression
		token.IDENT, token.INT, token.FLOAT, token.CHAR, token.STRING, token.FUNC, token.LPAREN,  // operand
		token.LBRACK, token.STRUCT,  // composite type
		token.MUL, token.AND, token.ARROW:  // unary operators
		return p.parseSimpleStmt();
	case token.GO:
		return p.parseGoStmt();
	case token.DEFER:
		return p.parseDeferStmt();
	case token.RETURN:
		return p.parseReturnStmt();
	case token.BREAK, token.CONTINUE, token.GOTO, token.FALLTHROUGH:
		return p.parseBranchStmt(p.tok);
	case token.LBRACE:
		return p.parseBlockStmt();
	case token.IF:
		return p.parseIfStmt();
	case token.FOR:
		return p.parseForStmt();
	case token.SWITCH:
		return p.parseSwitchStmt();
	case token.SELECT:
		return p.parseSelectStmt();
	case token.SEMICOLON, token.RBRACE:
		// don't consume the ";", it is the separator following the empty statement
		return &ast.EmptyStmt{p.pos};
	}

	// no statement found
	p.error(p.pos, "statement expected");
	return &ast.BadStmt{p.pos};
}


// ----------------------------------------------------------------------------
// Declarations

func (p *parser) parseImportSpec(pos token.Position, doc ast.Comments) *ast.ImportDecl {
	if p.trace {
		defer un(trace(p, "ImportSpec"));
	}

	var ident *ast.Ident;
	if p.tok == token.PERIOD {
		p.error(p.pos, `"import ." not yet handled properly`);
		p.next();
	} else if p.tok == token.IDENT {
		ident = p.parseIdent();
	}

	var path []*ast.StringLit;
	if p.tok == token.STRING {
		path = p.parseStringList(nil);
	} else {
		p.expect(token.STRING);  // use expect() error handling
	}

	return &ast.ImportDecl{doc, pos, ident, path};
}


func (p *parser) parseConstSpec(pos token.Position, doc ast.Comments) *ast.ConstDecl {
	if p.trace {
		defer un(trace(p, "ConstSpec"));
	}

	names := p.parseIdentList(nil);
	typ := p.tryType();
	var values []ast.Expr;
	if typ != nil || p.tok == token.ASSIGN {
		p.expect(token.ASSIGN);
		values = p.parseExpressionList();
	}

	return &ast.ConstDecl{doc, pos, names, typ, values};
}


func (p *parser) parseTypeSpec(pos token.Position, doc ast.Comments) *ast.TypeDecl {
	if p.trace {
		defer un(trace(p, "TypeSpec"));
	}

	ident := p.parseIdent();
	typ := p.parseType();

	return &ast.TypeDecl{doc, pos, ident, typ};
}


func (p *parser) parseVarSpec(pos token.Position, doc ast.Comments) *ast.VarDecl {
	if p.trace {
		defer un(trace(p, "VarSpec"));
	}

	names := p.parseIdentList(nil);
	typ := p.tryType();
	var values []ast.Expr;
	if typ == nil || p.tok == token.ASSIGN {
		p.expect(token.ASSIGN);
		values = p.parseExpressionList();
	}

	return &ast.VarDecl{doc, pos, names, typ, values};
}


func (p *parser) parseSpec(pos token.Position, doc ast.Comments, keyword int) ast.Decl {
	switch keyword {
	case token.IMPORT: return p.parseImportSpec(pos, doc);
	case token.CONST: return p.parseConstSpec(pos, doc);
	case token.TYPE: return p.parseTypeSpec(pos, doc);
	case token.VAR: return p.parseVarSpec(pos, doc);
	}

	unreachable();
	return nil;
}


func (p *parser) parseDecl(keyword int) ast.Decl {
	if p.trace {
		defer un(trace(p, "Decl"));
	}

	doc := p.getDoc();
	pos := p.expect(keyword);
	if p.tok == token.LPAREN {
		lparen := p.pos;
		p.next();
		list := vector.New(0);
		for p.tok != token.RPAREN && p.tok != token.EOF {
			list.Push(p.parseSpec(nopos, nil, keyword));
			if p.tok == token.SEMICOLON {
				p.next();
			} else {
				break;
			}
		}
		rparen := p.expect(token.RPAREN);
		p.opt_semi = true;

		// convert vector
		decls := make([]ast.Decl, list.Len());
		for i := 0; i < list.Len(); i++ {
			decls[i] = list.At(i).(ast.Decl);
		}

		return &ast.DeclList{doc, pos, keyword, lparen, decls, rparen};
	}

	return p.parseSpec(pos, doc, keyword);
}


// Function and method declarations
//
// func        ident (params)
// func        ident (params) type
// func        ident (params) (results)
// func (recv) ident (params)
// func (recv) ident (params) type
// func (recv) ident (params) (results)

func (p *parser) parseFunctionDecl() *ast.FuncDecl {
	if p.trace {
		defer un(trace(p, "FunctionDecl"));
	}

	doc := p.getDoc();
	pos := p.expect(token.FUNC);

	var recv *ast.Field;
	if p.tok == token.LPAREN {
		pos := p.pos;
		tmp := p.parseParameters(true);
		if len(tmp) == 1 {
			recv = tmp[0];
		} else {
			p.error(pos, "must have exactly one receiver");
		}
	}

	ident := p.parseIdent();
	params, results := p.parseSignature();

	var body *ast.BlockStmt;
	if p.tok == token.LBRACE {
		body = p.parseBlockStmt();
	}

	return &ast.FuncDecl{doc, recv, ident, &ast.FunctionType{pos, params, results}, body};
}


func (p *parser) parseDeclaration() ast.Decl {
	if p.trace {
		defer un(trace(p, "Declaration"));
	}

	switch p.tok {
	case token.CONST, token.TYPE, token.VAR:
		return p.parseDecl(p.tok);
	case token.FUNC:
		return p.parseFunctionDecl();
	}

	pos := p.pos;
	p.error(pos, "declaration expected");
	p.next();  // make progress
	return &ast.BadDecl{pos};
}


// ----------------------------------------------------------------------------
// Packages

// The Mode constants control how much of the source text is parsed.
type Mode int;
const (
	ParseEntirePackage Mode = iota;
	ParseImportDeclsOnly;
	ParsePackageClauseOnly;
)


func (p *parser) parsePackage(mode Mode) *ast.Package {
	if p.trace {
		defer un(trace(p, "Program"));
	}

	// package clause
	comment := p.getDoc();
	pos := p.expect(token.PACKAGE);
	name := p.parseIdent();
	if p.tok == token.SEMICOLON {
		// common error
		p.error(p.pos, "extra semicolon");
		p.next();
	}
	
	
	var decls []ast.Decl;
	if mode <= ParseImportDeclsOnly {
		// import decls
		list := vector.New(0);
		for p.tok == token.IMPORT {
			list.Push(p.parseDecl(token.IMPORT));
			if p.tok == token.SEMICOLON {
				p.next();
			}
		}

		if mode <= ParseEntirePackage {
			// rest of package body
			for p.tok != token.EOF {
				list.Push(p.parseDeclaration());
				if p.tok == token.SEMICOLON {
					p.next();
				}
			}
		}

		// convert declaration list
		decls = make([]ast.Decl, list.Len());
		for i := 0; i < list.Len(); i++ {
			decls[i] = list.At(i).(ast.Decl);
		}
	}

	// convert comments list
	comments := make([]*ast.Comment, p.comments.Len());
	for i := 0; i < p.comments.Len(); i++ {
		comments[i] = p.comments.At(i).(*ast.Comment);
	}

	return &ast.Package{comment, pos, name, decls, comments};
}


// ----------------------------------------------------------------------------
// Parsing of entire programs.

// Parse invokes the Go parser. It calls the scanner's Scan method repeatedly
// to obtain a token sequence which is parsed according to Go syntax. If an
// error handler is provided (err != nil), it is invoked for each syntax error
// encountered.
//
// Parse returns an AST and the number of syntax errors encountered. If the
// error count is 0, the result is the correct AST for the token sequence
// returned by the scanner (*). If the error count is > 0, the AST may only
// be constructed partially, with ast.BadX nodes representing the fragments
// of source code that contained syntax errors.
//
// The amount of source text parsed can be controlled with the mode parameter.
// The flags parameter controls optional parser functionality such as tracing.
//
// (*) Note that a scanner may find lexical syntax errors but still return
//     a legal token sequence. To be sure there are no syntax errors in the
//     source (and not just the token sequence corresponding to the source)
//     both the parser and scanner error count must be 0.
//
func Parse(scanner Scanner, err ErrorHandler, mode Mode, flags uint) (*ast.Package, int) {
	// initialize parser state
	var p parser;
	p.scanner = scanner;
	p.err = err;
	p.trace = flags & Trace != 0;
	p.comments.Init(0);
	p.next();

	// parse program
	return p.parsePackage(mode), p.errorCount;
}
