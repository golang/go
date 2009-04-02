// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A parser for Go source text. The input is a stream of lexical tokens
// provided via the Scanner interface. The output is an abstract syntax
// tree (AST) representing the Go source. The parser is invoked by calling
// Parse.
//
package parser

import (
	"ast";
	"fmt";
	"io";
	"scanner";
	"token";
	"vector";
)


// An implementation of an ErrorHandler may be provided to the parser.
// If a syntax error is encountered and a handler was installed, Error
// is called with a position and an error message. The position points
// to the beginning of the offending token.
//
type ErrorHandler interface {
	Error(pos token.Position, msg string);
}


type interval struct {
	beg, end int;
}


// The parser structure holds the parser's internal state.
type parser struct {
	scanner scanner.Scanner;
	err ErrorHandler;  // nil if no handler installed
	errorCount int;

	// Tracing/debugging
	mode uint;  // parsing mode
	trace bool;  // == (mode & Trace != 0)
	indent uint;  // indentation used for tracing output

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


// noPos is used when there is no corresponding source position for a token
var noPos token.Position;


// ----------------------------------------------------------------------------
// Parsing support

func (p *parser) printTrace(a ...) {
	const dots =
		". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . "
		". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ";
	const n = uint(len(dots));
	fmt.Printf("%5d:%3d: ", p.pos.Line, p.pos.Column);
	i := 2*p.indent;
	for ; i > n; i -= n {
		fmt.Print(dots);
	}
	fmt.Print(dots[0 : i]);
	fmt.Println(a);
}


func trace(p *parser, msg string) *parser {
	p.printTrace(msg, "(");
	p.indent++;
	return p;
}


func un/*trace*/(p *parser) {
	p.indent--;
	p.printTrace(")");
}


func (p *parser) next0() {
	// Because of one-token look-ahead, print the previous token
	// when tracing as it provides a more readable output. The
	// very first token (p.pos.Line == 0) is not initialized (it
	// is token.ILLEGAL), so don't print it .
	if p.trace && p.pos.Line > 0 {
		s := p.tok.String();
		switch {
		case p.tok.IsLiteral():
			p.printTrace(s, string(p.lit));
		case p.tok.IsOperator(), p.tok.IsKeyword():
			p.printTrace("\"" + s + "\"");
		default:
			p.printTrace(s);
		}
	}

	p.pos, p.tok, p.lit = p.scanner.Scan();
	p.opt_semi = false;
}


// Collect a comment in the parser's comment list and return the line
// on which the comment ends.
//
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


func (p *parser) error_expected(pos token.Position, msg string) {
	msg = "expected " + msg;
	if pos.Offset == p.pos.Offset {
		// the error happened at the current position;
		// make the error message more specific
		msg += ", found '" + p.tok.String() + "'";
		if p.tok.IsLiteral() {
			msg += " " + string(p.lit);
		}
	}
	p.error(pos, msg);
}


func (p *parser) expect(tok token.Token) token.Position {
	pos := p.pos;
	if p.tok != tok {
		p.error_expected(pos, "'" + tok.String() + "'");
	}
	p.next();  // make progress in any case
	return pos;
}


// ----------------------------------------------------------------------------
// Common productions

func (p *parser) tryType() ast.Expr;
func (p *parser) parseStringList(x *ast.StringLit) []*ast.StringLit
func (p *parser) parseExpression() ast.Expr;
func (p *parser) parseStatement() ast.Stmt;
func (p *parser) parseDeclaration() ast.Decl;


func (p *parser) parseIdent() *ast.Ident {
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
	list.Push(p.parseExpression());
	for p.tok == token.COMMA {
		p.next();
		list.Push(p.parseExpression());
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
		p.error_expected(p.pos, "type");
		return &ast.BadExpr{p.pos};
	}

	return typ;
}


func (p *parser) parseQualifiedIdent() ast.Expr {
	if p.trace {
		defer un(trace(p, "QualifiedIdent"));
	}

	var x ast.Expr = p.parseIdent();
	if p.tok == token.PERIOD {
		// first identifier is a package identifier
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


func (p *parser) parseArrayOrSliceType(ellipsis_ok bool) ast.Expr {
	if p.trace {
		defer un(trace(p, "ArrayOrSliceType"));
	}

	lbrack := p.expect(token.LBRACK);
	var len ast.Expr;
	if ellipsis_ok && p.tok == token.ELLIPSIS {
		len = &ast.Ellipsis{p.pos};
		p.next();
	} else if p.tok != token.RBRACK {
		len = p.parseExpression();
	}
	p.expect(token.RBRACK);
	elt := p.parseType();

	if len != nil {
		return &ast.ArrayType{lbrack, len, elt};
	}
	
	return &ast.SliceType{lbrack, elt};
}


func (p *parser) makeIdentList(list *vector.Vector) []*ast.Ident {
	idents := make([]*ast.Ident, list.Len());
	for i := 0; i < list.Len(); i++ {
		ident, is_ident := list.At(i).(*ast.Ident);
		if !is_ident {
			pos := list.At(i).(ast.Expr).Pos();
			p.error_expected(pos, "identifier");
			idents[i] = &ast.Ident{pos, []byte{}};
		}
		idents[i] = ident;
	}
	return idents;
}


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
		// IdentifierList Type
		idents = p.makeIdentList(list);
	} else {
		// Type (anonymous field)
		if list.Len() == 1 {
			// TODO check that this looks like a type
			typ = list.At(0).(ast.Expr);
		} else {
			p.error_expected(p.pos, "anonymous field");
			typ = &ast.BadExpr{p.pos};
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


func (p *parser) tryParameterType(ellipsis_ok bool) ast.Expr {
	if ellipsis_ok && p.tok == token.ELLIPSIS {
		pos := p.pos;
		p.next();
		if p.tok != token.RPAREN {
			// "..." always must be at the very end of a parameter list
			p.error(pos, "expected type, found '...'");
		}
		return &ast.Ellipsis{pos};
	}
	return p.tryType();
}


func (p *parser) parseParameterType(ellipsis_ok bool) ast.Expr {
	typ := p.tryParameterType(ellipsis_ok);
	if typ == nil {
		p.error_expected(p.pos, "type");
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
		list.Push(p.parseParameterType(ellipsis_ok));
		if p.tok == token.COMMA {
			p.next();
		} else {
			break;
		}
	}

	// if we had a list of identifiers, it must be followed by a type
	typ := p.tryParameterType(ellipsis_ok);

	return list, typ;
}


func (p *parser) parseParameterList(ellipsis_ok bool) []*ast.Field {
	if p.trace {
		defer un(trace(p, "ParameterList"));
	}

	list, typ := p.parseParameterDecl(ellipsis_ok);
	if typ != nil {
		// IdentifierList Type
		idents := p.makeIdentList(list);
		list.Init(0);
		list.Push(&ast.Field{nil, idents, typ, nil});

		for p.tok == token.COMMA {
			p.next();
			idents := p.parseIdentList(nil);
			typ := p.parseParameterType(ellipsis_ok);
			list.Push(&ast.Field{nil, idents, typ, nil});
		}

	} else {
		// Type { "," Type } (anonymous parameters)
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


func (p *parser) parseSignature() (params []*ast.Field, results []*ast.Field) {
	if p.trace {
		defer un(trace(p, "Signature"));
	}

	params = p.parseParameters(true);
	results = p.parseResult();

	return params, results;
}


func (p *parser) parseFuncType() *ast.FuncType {
	if p.trace {
		defer un(trace(p, "FuncType"));
	}

	pos := p.expect(token.FUNC);
	params, results := p.parseSignature();

	return &ast.FuncType{pos, params, results};
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
		// methods
		idents = p.parseIdentList(x);
		params, results := p.parseSignature();
		typ = &ast.FuncType{noPos, params, results};
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


func (p *parser) parseChanType() *ast.ChanType {
	if p.trace {
		defer un(trace(p, "ChanType"));
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

	return &ast.ChanType{pos, dir, value};
}


func (p *parser) tryRawType(ellipsis_ok bool) ast.Expr {
	switch p.tok {
	case token.IDENT: return p.parseTypeName();
	case token.LBRACK: return p.parseArrayOrSliceType(ellipsis_ok);
	case token.STRUCT: return p.parseStructType();
	case token.MUL: return p.parsePointerType();
	case token.FUNC: return p.parseFuncType();
	case token.INTERFACE: return p.parseInterfaceType();
	case token.MAP: return p.parseMapType();
	case token.CHAN, token.ARROW: return p.parseChanType();
	case token.LPAREN:
		lparen := p.pos;
		p.next();
		typ := p.parseType();
		rparen := p.expect(token.RPAREN);
		return &ast.ParenExpr{lparen, typ, rparen};
	}

	// no type found
	return nil;
}


func (p *parser) tryType() ast.Expr {
	return p.tryRawType(false);
}


// ----------------------------------------------------------------------------
// Blocks

func makeStmtList(list *vector.Vector) []ast.Stmt {
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
	
	return makeStmtList(list);
}


func (p *parser) parseBlockStmt() *ast.BlockStmt {
	if p.trace {
		defer un(trace(p, "BlockStmt"));
	}

	lbrace := p.expect(token.LBRACE);
	list := p.parseStatementList();
	rbrace := p.expect(token.RBRACE);
	p.opt_semi = true;

	return &ast.BlockStmt{lbrace, list, rbrace};
}


// ----------------------------------------------------------------------------
// Expressions

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


func (p *parser) parseFuncLit() ast.Expr {
	if p.trace {
		defer un(trace(p, "FuncLit"));
	}

	typ := p.parseFuncType();
	p.expr_lev++;
	body := p.parseBlockStmt();
	p.expr_lev--;

	return &ast.FuncLit{typ, body};
}


// parseOperand may return an expression or a raw type (incl. array
// types of the form [...]T. Callers must verify the result.
//
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
		x := p.parseExpression();
		p.expr_lev--;
		rparen := p.expect(token.RPAREN);
		return &ast.ParenExpr{lparen, x, rparen};

	case token.FUNC:
		return p.parseFuncLit();

	default:
		t := p.tryRawType(true);  // could be type for composite literal
		if t != nil {
			return t;
		}
	}

	p.error_expected(p.pos, "operand");
	p.next();  // make progress
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
	}

	// type assertion
	p.expect(token.LPAREN);
	var typ ast.Expr;
	if p.tok == token.TYPE {
		// special case for type switch
		typ = &ast.Ident{p.pos, p.lit};
		p.next();
	} else {
		typ = p.parseType();
	}
	p.expect(token.RPAREN);

	return &ast.TypeAssertExpr{x, typ};
}


func (p *parser) parseIndexOrSlice(x ast.Expr) ast.Expr {
	if p.trace {
		defer un(trace(p, "IndexOrSlice"));
	}

	p.expect(token.LBRACK);
	p.expr_lev++;
	begin := p.parseExpression();
	var end ast.Expr;
	if p.tok == token.COLON {
		p.next();
		end = p.parseExpression();
	}
	p.expr_lev--;
	p.expect(token.RBRACK);

	if end != nil {
		return &ast.SliceExpr{x, begin, end};
	}

	return &ast.IndexExpr{x, begin};
}


func (p *parser) parseCallOrConversion(fun ast.Expr) *ast.CallExpr {
	if p.trace {
		defer un(trace(p, "CallOrConversion"));
	}

	lparen := p.expect(token.LPAREN);
	var args []ast.Expr;
	if p.tok != token.RPAREN {
		args = p.parseExpressionList();
	}
	rparen := p.expect(token.RPAREN);

	return &ast.CallExpr{fun, lparen, args, rparen};
}


func (p *parser) parseKeyValueExpr() ast.Expr {
	if p.trace {
		defer un(trace(p, "KeyValueExpr"));
	}

	key := p.parseExpression();

	if p.tok == token.COLON {
		colon := p.pos;
		p.next();
		value := p.parseExpression();
		return &ast.KeyValueExpr{key, colon, value};
	}
	
	return key;
}


func isPair(x ast.Expr) bool {
	tmp, is_pair := x.(*ast.KeyValueExpr);
	return is_pair;
}


func (p *parser) parseExpressionOrKeyValueList() []ast.Expr {
	if p.trace {
		defer un(trace(p, "ExpressionOrKeyValueList"));
	}

	var pairs bool;
	list := vector.New(0);
	for p.tok != token.RBRACE && p.tok != token.EOF {
		x := p.parseKeyValueExpr();

		if list.Len() == 0 {
			pairs = isPair(x);
		} else {
			// not the first element - check syntax
			if pairs != isPair(x) {
				p.error_expected(x.Pos(), "all single expressions or all key-value pairs");
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
		elts = p.parseExpressionOrKeyValueList();
	}
	rbrace := p.expect(token.RBRACE);
	return &ast.CompositeLit{typ, lbrace, elts, rbrace};
}


// TODO Consider different approach to checking syntax after parsing:
//      Provide a arguments (set of flags) to parsing functions
//      restricting what they are syupposed to accept depending
//      on context.

// checkExpr checks that x is an expression (and not a type).
func (p *parser) checkExpr(x ast.Expr) ast.Expr {
	// TODO should provide predicate in AST nodes
	switch t := x.(type) {
	case *ast.BadExpr:
	case *ast.Ident:
	case *ast.IntLit:
	case *ast.FloatLit:
	case *ast.CharLit:
	case *ast.StringLit:
	case *ast.StringList:
	case *ast.FuncLit:
	case *ast.CompositeLit:
	case *ast.ParenExpr:
	case *ast.SelectorExpr:
	case *ast.IndexExpr:
	case *ast.SliceExpr:
	case *ast.TypeAssertExpr:
	case *ast.CallExpr:
	case *ast.StarExpr:
	case *ast.UnaryExpr:
		if t.Op == token.RANGE {
			// the range operator is only allowed at the top of a for statement
			p.error_expected(x.Pos(), "expression");
			x = &ast.BadExpr{x.Pos()};
		}
	case *ast.BinaryExpr:
	default:
		// all other nodes are not proper expressions
		p.error_expected(x.Pos(), "expression");
		x = &ast.BadExpr{x.Pos()};
	}
	return x;
}


// checkTypeName checks that x is type name.
func (p *parser) checkTypeName(x ast.Expr) ast.Expr {
	// TODO should provide predicate in AST nodes
	switch t := x.(type) {
	case *ast.BadExpr:
	case *ast.Ident:
	case *ast.ParenExpr: p.checkTypeName(t.X);  // TODO should (TypeName) be illegal?
	case *ast.SelectorExpr: p.checkTypeName(t.X);
	default:
		// all other nodes are not type names
		p.error_expected(x.Pos(), "type name");
		x = &ast.BadExpr{x.Pos()};
	}
	return x;
}


// checkCompositeLitType checks that x is a legal composite literal type.
func (p *parser) checkCompositeLitType(x ast.Expr) ast.Expr {
	// TODO should provide predicate in AST nodes
	switch t := x.(type) {
	case *ast.BadExpr: return x;
	case *ast.Ident: return x;
	case *ast.ParenExpr: p.checkCompositeLitType(t.X);
	case *ast.SelectorExpr: p.checkTypeName(t.X);
	case *ast.ArrayType: return x;
	case *ast.SliceType: return x;
	case *ast.StructType: return x;
	case *ast.MapType: return x;
	default:
		// all other nodes are not legal composite literal types
		p.error_expected(x.Pos(), "composite literal type");
		x = &ast.BadExpr{x.Pos()};
	}
	return x;
}


// checkExprOrType checks that x is an expression or a type
// (and not a raw type such as [...]T).
//
func (p *parser) checkExprOrType(x ast.Expr) ast.Expr {
	// TODO should provide predicate in AST nodes
	switch t := x.(type) {
	case *ast.UnaryExpr:
		if t.Op == token.RANGE {
			// the range operator is only allowed at the top of a for statement
			p.error_expected(x.Pos(), "expression");
			x = &ast.BadExpr{x.Pos()};
		}
	case *ast.ArrayType:
		if len, is_ellipsis := t.Len.(*ast.Ellipsis); is_ellipsis {
			p.error(len.Pos(), "expected array length, found '...'");
			x = &ast.BadExpr{x.Pos()};
		}
	}
	
	// all other nodes are expressions or types
	return x;
}


func (p *parser) parsePrimaryExpr() ast.Expr {
	if p.trace {
		defer un(trace(p, "PrimaryExpr"));
	}

	x := p.parseOperand();
	for {
		switch p.tok {
		case token.PERIOD: x = p.parseSelectorOrTypeAssertion(p.checkExpr(x));
		case token.LBRACK: x = p.parseIndexOrSlice(p.checkExpr(x));
		case token.LPAREN: x = p.parseCallOrConversion(p.checkExprOrType(x));
		case token.LBRACE:
			if p.expr_lev >= 0 {
				x = p.parseCompositeLit(p.checkCompositeLitType(x));
			} else {
				return p.checkExprOrType(x);
			}
		default:
			return p.checkExprOrType(x);
		}
	}

	panic();  // unreachable
	return nil;
}


func (p *parser) parseUnaryExpr() ast.Expr {
	if p.trace {
		defer un(trace(p, "UnaryExpr"));
	}

	switch p.tok {
	case token.ADD, token.SUB, token.NOT, token.XOR, token.ARROW, token.AND, token.RANGE:
		pos, op := p.pos, p.tok;
		p.next();
		x := p.parseUnaryExpr();
		return &ast.UnaryExpr{pos, op, p.checkExpr(x)};

	case token.MUL:
		// unary "*" expression or pointer type
		pos := p.pos;
		p.next();
		x := p.parseUnaryExpr();
		return &ast.StarExpr{pos, p.checkExprOrType(x)};
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
			pos, op := p.pos, p.tok;
			p.next();
			y := p.parseBinaryExpr(prec + 1);
			x = &ast.BinaryExpr{p.checkExpr(x), pos, op, p.checkExpr(y)};
		}
	}

	return x;
}


func (p *parser) parseExpression() ast.Expr {
	if p.trace {
		defer un(trace(p, "Expression"));
	}

	return p.parseBinaryExpr(token.LowestPrec + 1);
}


// ----------------------------------------------------------------------------
// Statements


func (p *parser) parseSimpleStmt(label_ok bool) ast.Stmt {
	if p.trace {
		defer un(trace(p, "SimpleStmt"));
	}

	x := p.parseExpressionList();

	switch p.tok {
	case token.COLON:
		// labeled statement
		p.next();
		if label_ok && len(x) == 1 {
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
	x := p.parseExpression();
	if call, is_call := x.(*ast.CallExpr); is_call {
		return call;
	}
	p.error_expected(x.Pos(), "function/method call");
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


func (p *parser) makeExpr(s ast.Stmt) ast.Expr {
	if s == nil {
		return nil;
	}
	if es, is_expr := s.(*ast.ExprStmt); is_expr {
		return p.checkExpr(es.X);
	}
	p.error(s.Pos(), "expected condition, found simple statement");
	return &ast.BadExpr{s.Pos()};
}


func (p *parser) parseControlClause(isForStmt bool) (s1, s2, s3 ast.Stmt) {
	if p.tok != token.LBRACE {
		prev_lev := p.expr_lev;
		p.expr_lev = -1;

		if p.tok != token.SEMICOLON {
			s1 = p.parseSimpleStmt(false);
		}
		if p.tok == token.SEMICOLON {
			p.next();
			if p.tok != token.LBRACE && p.tok != token.SEMICOLON {
				s2 = p.parseSimpleStmt(false);
			}
			if isForStmt {
				// for statements have a 3rd section
				p.expect(token.SEMICOLON);
				if p.tok != token.LBRACE {
					s3 = p.parseSimpleStmt(false);
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

	return &ast.IfStmt{pos, s1, p.makeExpr(s2), body, else_};
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
		body := &ast.BlockStmt{lbrace, makeStmtList(cases), rbrace};
		return &ast.SwitchStmt{pos, s1, p.makeExpr(s2), body};
	}

	// type switch
	// TODO do all the checks!
	lbrace := p.expect(token.LBRACE);
	cases := vector.New(0);
	for p.tok == token.CASE || p.tok == token.DEFAULT {
		cases.Push(p.parseTypeCaseClause());
	}
	rbrace := p.expect(token.RBRACE);
	p.opt_semi = true;
	body := &ast.BlockStmt{lbrace, makeStmtList(cases), rbrace};
	return &ast.TypeSwitchStmt{pos, s1, s2, body};
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
			rhs = p.parseExpression();
		} else {
			// SendExpr or RecvExpr
			rhs = p.parseExpression();
			if p.tok == token.ASSIGN || p.tok == token.DEFINE {
				// RecvExpr with assignment
				tok = p.tok;
				p.next();
				lhs = rhs;
				if p.tok == token.ARROW {
					rhs = p.parseExpression();
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
	body := &ast.BlockStmt{lbrace, makeStmtList(cases), rbrace};

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
			p.error_expected(as.TokPos, "'=' or ':='");
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
			p.error_expected(as.Lhs[0].Pos(), "1 or 2 expressions");
			return &ast.BadStmt{pos};
		}
		// check rhs
		if len(as.Rhs) != 1 {
			p.error_expected(as.Rhs[0].Pos(), "1 expressions");
			return &ast.BadStmt{pos};
		}
		if rhs, is_unary := as.Rhs[0].(*ast.UnaryExpr); is_unary && rhs.Op == token.RANGE {
			// rhs is range expression; check lhs
			return &ast.RangeStmt{pos, key, value, as.TokPos, as.Tok, rhs.X, body}
		} else {
			p.error_expected(s2.Pos(), "range clause");
			return &ast.BadStmt{pos};
		}
	} else {
		// regular for statement
		return &ast.ForStmt{pos, s1, p.makeExpr(s2), s3, body};
	}
	
	panic();  // unreachable
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
		return p.parseSimpleStmt(true);
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
	case token.SWITCH:
		return p.parseSwitchStmt();
	case token.SELECT:
		return p.parseSelectStmt();
	case token.FOR:
		return p.parseForStmt();
	case token.SEMICOLON, token.RBRACE:
		// don't consume the ";", it is the separator following the empty statement
		return &ast.EmptyStmt{p.pos};
	}

	// no statement found
	p.error_expected(p.pos, "statement");
	return &ast.BadStmt{p.pos};
}


// ----------------------------------------------------------------------------
// Declarations

type parseSpecFunction func(p *parser, doc ast.Comments) ast.Spec

func parseImportSpec(p *parser, doc ast.Comments) ast.Spec {
	if p.trace {
		defer un(trace(p, "ImportSpec"));
	}

	var ident *ast.Ident;
	if p.tok == token.PERIOD {
		ident = &ast.Ident{p.pos, []byte{'.'}};
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

	return &ast.ImportSpec{doc, ident, path};
}


func parseConstSpec(p *parser, doc ast.Comments) ast.Spec {
	if p.trace {
		defer un(trace(p, "ConstSpec"));
	}

	idents := p.parseIdentList(nil);
	typ := p.tryType();
	var values []ast.Expr;
	if typ != nil || p.tok == token.ASSIGN {
		p.expect(token.ASSIGN);
		values = p.parseExpressionList();
	}

	return &ast.ValueSpec{doc, idents, typ, values};
}


func parseTypeSpec(p *parser, doc ast.Comments) ast.Spec {
	if p.trace {
		defer un(trace(p, "TypeSpec"));
	}

	ident := p.parseIdent();
	typ := p.parseType();

	return &ast.TypeSpec{doc, ident, typ};
}


func parseVarSpec(p *parser, doc ast.Comments) ast.Spec {
	if p.trace {
		defer un(trace(p, "VarSpec"));
	}

	idents := p.parseIdentList(nil);
	typ := p.tryType();
	var values []ast.Expr;
	if typ == nil || p.tok == token.ASSIGN {
		p.expect(token.ASSIGN);
		values = p.parseExpressionList();
	}

	return &ast.ValueSpec{doc, idents, typ, values};
}


func (p *parser) parseGenDecl(keyword token.Token, f parseSpecFunction) *ast.GenDecl {
	if p.trace {
		defer un(trace(p, keyword.String() + "Decl"));
	}

	doc := p.getDoc();
	pos := p.expect(keyword);
	var lparen, rparen token.Position;
	list := vector.New(0);
	if p.tok == token.LPAREN {
		lparen = p.pos;
		p.next();
		for p.tok != token.RPAREN && p.tok != token.EOF {
			doc := p.getDoc();
			list.Push(f(p, doc));
			if p.tok == token.SEMICOLON {
				p.next();
			} else {
				break;
			}
		}
		rparen = p.expect(token.RPAREN);
		p.opt_semi = true;
	} else {
		list.Push(f(p, doc));
	}

	// convert vector
	specs := make([]ast.Spec, list.Len());
	for i := 0; i < list.Len(); i++ {
		specs[i] = list.At(i);
	}
	return &ast.GenDecl{doc, pos, keyword, lparen, specs, rparen};
}


func (p *parser) parseReceiver() *ast.Field {
	if p.trace {
		defer un(trace(p, "Receiver"));
	}

	pos := p.pos;
	par := p.parseParameters(false);

	// must have exactly one receiver
	if len(par) != 1 || len(par) == 1 && len(par[0].Names) > 1 {
		p.error_expected(pos, "exactly one receiver");
		return &ast.Field{nil, nil, &ast.BadExpr{noPos}, nil};
	}

	recv := par[0];

	// recv type must be TypeName or *TypeName
	base := recv.Type;
	if ptr, is_ptr := base.(*ast.StarExpr); is_ptr {
		base = ptr.X;
	}
	p.checkTypeName(base);

	return recv;
}


func (p *parser) parseFunctionDecl() *ast.FuncDecl {
	if p.trace {
		defer un(trace(p, "FunctionDecl"));
	}

	doc := p.getDoc();
	pos := p.expect(token.FUNC);

	var recv *ast.Field;
	if p.tok == token.LPAREN {
		recv = p.parseReceiver();
	}

	ident := p.parseIdent();
	params, results := p.parseSignature();

	var body *ast.BlockStmt;
	if p.tok == token.LBRACE {
		body = p.parseBlockStmt();
	}

	return &ast.FuncDecl{doc, recv, ident, &ast.FuncType{pos, params, results}, body};
}


func (p *parser) parseDeclaration() ast.Decl {
	if p.trace {
		defer un(trace(p, "Declaration"));
	}

	var f parseSpecFunction;
	switch p.tok {
	case token.CONST: f = parseConstSpec;
	case token.TYPE: f = parseTypeSpec;
	case token.VAR: f = parseVarSpec;
	case token.FUNC:
		return p.parseFunctionDecl();
	default:
		pos := p.pos;
		p.error_expected(pos, "declaration");
		p.next();  // make progress
		return &ast.BadDecl{pos};
	}
	
	return p.parseGenDecl(p.tok, f);
}


// ----------------------------------------------------------------------------
// Packages

// The mode parameter to the Parse function is a set of flags (or 0).
// They control the amount of source code parsed and other optional
// parser functionality.
//
const (
	PackageClauseOnly uint = 1 << iota;  // parsing stops after package clause
	ImportsOnly;  // parsing stops after import declarations
	ParseComments;  // parse comments and add them to AST
	Trace;  // print a trace of parsed productions
)


func (p *parser) parsePackage() *ast.Program {
	if p.trace {
		defer un(trace(p, "Program"));
	}

	// package clause
	comment := p.getDoc();
	pos := p.expect(token.PACKAGE);
	ident := p.parseIdent();
	if p.tok == token.SEMICOLON {
		// common error
		p.error(p.pos, "extra semicolon");
		p.next();
	}

	var decls []ast.Decl;
	if p.mode & PackageClauseOnly == 0 {
		// import decls
		list := vector.New(0);
		for p.tok == token.IMPORT {
			list.Push(p.parseGenDecl(token.IMPORT, parseImportSpec));
			if p.tok == token.SEMICOLON {
				p.next();
			}
		}

		if p.mode & ImportsOnly == 0 {
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

	return &ast.Program{comment, pos, ident, decls, comments};
}


// ----------------------------------------------------------------------------
// Parsing of entire programs.

func readSource(src interface{}, err ErrorHandler) []byte {
	errmsg := "invalid input type (or nil)";

	switch s := src.(type) {
	case string:
		return io.StringBytes(s);
	case []byte:
		return s;
	case *io.ByteBuffer:
		// is io.Read, but src is already available in []byte form
		if s != nil {
			return s.Data();
		}
	case io.Read:
		var buf io.ByteBuffer;
		n, os_err := io.Copy(s, &buf);
		if os_err == nil {
			return buf.Data();
		}
		errmsg = os_err.String();
	}

	if err != nil {
		err.Error(noPos, errmsg);
	}
	return nil;
}


// Parse parses a Go program.
//
// The program source src may be provided in a variety of formats. At the
// moment the following types are supported: string, []byte, and io.Read.
//
// The ErrorHandler err, if not nil, is invoked if src cannot be read and
// for each syntax error found. The mode parameter controls the amount of
// source text parsed and other optional parser functionality.
//
// Parse returns an AST and the boolean value true if no errors occured;
// it returns a partial AST (or nil if the source couldn't be read) and
// the boolean value false to indicate failure.
// 
// If syntax errors were found, the AST may only be constructed partially,
// with ast.BadX nodes representing the fragments of erroneous source code.
//
func Parse(src interface{}, err ErrorHandler, mode uint) (*ast.Program, bool) {
	data := readSource(src, err);

	// initialize parser state
	var p parser;
	p.scanner.Init(data, err, mode & ParseComments != 0);
	p.err = err;
	p.mode = mode;
	p.trace = mode & Trace != 0;  // for convenience (p.trace is used frequently)
	p.comments.Init(0);
	p.next();

	// parse program
	prog := p.parsePackage();

	return prog, p.scanner.ErrorCount == 0 && p.errorCount == 0;
}
