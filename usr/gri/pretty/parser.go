// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A parser for Go source text. The input is a stream of lexical tokens
// provided via the Scanner interface. The output is an abstract syntax
// tree (AST) representing the Go source.
//
// A client may parse the entire program (ParseProgram), only the package
// clause (ParsePackageClause), or the package clause and the import
// declarations (ParseImportDecls).
//
package Parser

import (
	"fmt";
	"vector";
	"token";
	"scanner";
	"ast";
)


// TODO rename Position to scanner.Position, possibly factor out
type Position scanner.Location


type interval struct {
	beg, end int;
}


// A Parser holds the parser's internal state while processing
// a given text. It can be allocated as part of another data
// structure but must be initialized via Init before use.
//
type Parser struct {
	scanner *scanner.Scanner;
	err scanner.ErrorHandler;

	// Tracing/debugging
	trace bool;
	indent uint;

	comments vector.Vector;  // list of collected, unassociated comments
	last_doc interval;  // last comments interval of consecutive comments

	// The next token
	pos Position;  // token location
	tok int;  // one token look-ahead
	val []byte;  // token value

	// Non-syntactic parser control
	opt_semi bool;  // true if semicolon separator is optional in statement list
	expr_lev int;  // < 0: in control clause, >= 0: in expression
};


// When we don't have a location use nopos.
// TODO make sure we always have a location.
var nopos Position;


// ----------------------------------------------------------------------------
// Helper functions

func unreachable() {
	panic("unreachable");
}


// ----------------------------------------------------------------------------
// Parsing support

func (P *Parser) printIndent() {
	i := P.indent;
	// reduce printing time by a factor of 2 or more
	for ; i > 10; i -= 10 {
		fmt.Printf(". . . . . . . . . . ");
	}
	for ; i > 0; i-- {
		fmt.Printf(". ");
	}
}


func trace(P *Parser, msg string) *Parser {
	P.printIndent();
	fmt.Printf("%s (\n", msg);
	P.indent++;
	return P;
}


func un/*trace*/(P *Parser) {
	P.indent--;
	P.printIndent();
	fmt.Printf(")\n");
}


func (P *Parser) next0() {
	P.pos, P.tok, P.val = P.scanner.Scan();
	P.opt_semi = false;

	if P.trace {
		P.printIndent();
		switch P.tok {
		case token.IDENT, token.INT, token.FLOAT, token.CHAR, token.STRING:
			fmt.Printf("%d:%d: %s = %s\n", P.pos.Line, P.pos.Col, token.TokenString(P.tok), P.val);
		case token.LPAREN:
			// don't print '(' - screws up selection in terminal window
			fmt.Printf("%d:%d: LPAREN\n", P.pos.Line, P.pos.Col);
		case token.RPAREN:
			// don't print ')' - screws up selection in terminal window
			fmt.Printf("%d:%d: RPAREN\n", P.pos.Line, P.pos.Col);
		default:
			fmt.Printf("%d:%d: %s\n", P.pos.Line, P.pos.Col, token.TokenString(P.tok));
		}
	}
}


// Collect a comment in the parser's comment list and return the line
// on which the comment ends.
func (P *Parser) collectComment() int {
	// For /*-style comments, the comment may end on a different line.
	// Scan the comment for '\n' chars and adjust the end line accordingly.
	// (Note that the position of the next token may be even further down
	// as there may be more whitespace lines after the comment.)
	endline := P.pos.Line;
	if P.val[1] == '*' {
		for i, b := range P.val {
			if b == '\n' {
				endline++;
			}
		}
	}
	P.comments.Push(&ast.Comment{P.pos, P.val, endline});
	P.next0();
	
	return endline;
}


func (P *Parser) getComments() interval {
	// group adjacent comments, an empty line terminates a group
	beg := P.comments.Len();
	endline := P.pos.Line;
	for P.tok == token.COMMENT && endline+1 >= P.pos.Line {
		endline = P.collectComment();
	}
	end := P.comments.Len();
	return interval {beg, end};
}


func (P *Parser) next() {
	P.next0();
	P.last_doc = interval{0, 0};
	for P.tok == token.COMMENT {
		P.last_doc = P.getComments();
	}
}


func (P *Parser) Init(scanner *scanner.Scanner, err scanner.ErrorHandler, trace bool) {
	P.scanner = scanner;
	P.err = err;
	P.trace = trace;
	P.comments.Init(0);
	P.next();
}


func (P *Parser) error(pos Position, msg string) {
	P.err.Error(pos, msg);
}


func (P *Parser) expect(tok int) Position {
	if P.tok != tok {
		msg := "expected '" + token.TokenString(tok) + "', found '" + token.TokenString(P.tok) + "'";
		if token.IsLiteral(P.tok) {
			msg += " " + string(P.val);
		}
		P.error(P.pos, msg);
	}
	loc := P.pos;
	P.next();  // make progress in any case
	return loc;
}


func (P *Parser) getDoc() ast.Comments {
	doc := P.last_doc;
	n := doc.end - doc.beg;
	
	if n <= 0 || P.comments.At(doc.end - 1).(*ast.Comment).EndLine + 1 < P.pos.Line {
		// no comments or empty line between last comment and current token;
		// do not use as documentation
		return nil;
	}

	// found immediately adjacent comment interval;
	// use as documentation
	c := make(ast.Comments, n);
	for i := 0; i < n; i++ {
		c[i] = P.comments.At(doc.beg + i).(*ast.Comment);
		// TODO find a better way to do this
		P.comments.Set(doc.beg + i, nil);  // remove the comment from the general list
	}
	return c;
}


// ----------------------------------------------------------------------------
// Common productions

func (P *Parser) tryType() ast.Expr;
func (P *Parser) parseExpression(prec int) ast.Expr;
func (P *Parser) parseStatement() ast.Stat;
func (P *Parser) parseDeclaration() ast.Decl;


func (P *Parser) parseIdent() *ast.Ident {
	if P.trace {
		defer un(trace(P, "Ident"));
	}

	if P.tok == token.IDENT {
		x := &ast.Ident{P.pos, P.val};
		P.next();
		return x;
	}
	P.expect(token.IDENT);  // use expect() error handling

	return &ast.Ident{P.pos, [0]byte{}};
}


func (P *Parser) parseIdentList(x ast.Expr) []*ast.Ident {
	if P.trace {
		defer un(trace(P, "IdentList"));
	}

	list := vector.New(0);
	if x == nil {
		x = P.parseIdent();
	}
	list.Push(x);
	for P.tok == token.COMMA {
		P.next();
		list.Push(P.parseIdent());
	}

	// convert vector
	idents := make([]*ast.Ident, list.Len());
	for i := 0; i < list.Len(); i++ {
		idents[i] = list.At(i).(*ast.Ident);
	}

	return idents;
}


func (P *Parser) parseExpressionList() []ast.Expr {
	if P.trace {
		defer un(trace(P, "ExpressionList"));
	}

	list := vector.New(0);
	list.Push(P.parseExpression(1));
	for P.tok == token.COMMA {
		P.next();
		list.Push(P.parseExpression(1));
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

func (P *Parser) parseType() ast.Expr {
	if P.trace {
		defer un(trace(P, "Type"));
	}

	typ := P.tryType();
	if typ == nil {
		P.error(P.pos, "type expected");
		typ = &ast.BadExpr{P.pos};
	}

	return typ;
}


func (P *Parser) parseVarType() ast.Expr {
	if P.trace {
		defer un(trace(P, "VarType"));
	}

	return P.parseType();
}


func (P *Parser) parseQualifiedIdent() ast.Expr {
	if P.trace {
		defer un(trace(P, "QualifiedIdent"));
	}

	var x ast.Expr = P.parseIdent();
	for P.tok == token.PERIOD {
		P.next();
		sel := P.parseIdent();
		x = &ast.SelectorExpr{x, sel};
	}
	return x;
}


func (P *Parser) parseTypeName() ast.Expr {
	if P.trace {
		defer un(trace(P, "TypeName"));
	}

	return P.parseQualifiedIdent();
}


func (P *Parser) parseArrayType() *ast.ArrayType {
	if P.trace {
		defer un(trace(P, "ArrayType"));
	}

	lbrack := P.expect(token.LBRACK);
	var len ast.Expr;
	if P.tok == token.ELLIPSIS {
		len = &ast.Ellipsis{P.pos};
		P.next();
	} else if P.tok != token.RBRACK {
		len = P.parseExpression(1);
	}
	P.expect(token.RBRACK);
	elt := P.parseType();

	return &ast.ArrayType{lbrack, len, elt};
}


func (P *Parser) parseChannelType() *ast.ChannelType {
	if P.trace {
		defer un(trace(P, "ChannelType"));
	}

	pos := P.pos;
	dir := ast.SEND | ast.RECV;
	if P.tok == token.CHAN {
		P.next();
		if P.tok == token.ARROW {
			P.next();
			dir = ast.SEND;
		}
	} else {
		P.expect(token.ARROW);
		P.expect(token.CHAN);
		dir = ast.RECV;
	}
	value := P.parseVarType();

	return &ast.ChannelType{pos, dir, value};
}


func (P *Parser) tryParameterType() ast.Expr {
	if P.tok == token.ELLIPSIS {
		loc  := P.pos;
		P.next();
		return &ast.Ellipsis{loc};
	}
	return P.tryType();
}


func (P *Parser) parseParameterType() ast.Expr {
	typ := P.tryParameterType();
	if typ == nil {
		P.error(P.pos, "type expected");
		typ = &ast.BadExpr{P.pos};
	}

	return typ;
}


func (P *Parser) parseParameterDecl(ellipsis_ok bool) (*vector.Vector, ast.Expr) {
	if P.trace {
		defer un(trace(P, "ParameterDecl"));
	}

	// a list of identifiers looks like a list of type names
	list := vector.New(0);
	for {
		// TODO do not allow ()'s here
		list.Push(P.parseParameterType());
		if P.tok == token.COMMA {
			P.next();
		} else {
			break;
		}
	}

	// if we had a list of identifiers, it must be followed by a type
	typ := P.tryParameterType();

	return list, typ;
}


func (P *Parser) parseParameterList(ellipsis_ok bool) []*ast.Field {
	if P.trace {
		defer un(trace(P, "ParameterList"));
	}

	list, typ := P.parseParameterDecl(false);
	if typ != nil {
		// IdentifierList Type
		// convert list of identifiers into []*Ident
		idents := make([]*ast.Ident, list.Len());
		for i := 0; i < list.Len(); i++ {
			idents[i] = list.At(i).(*ast.Ident);
		}
		list.Init(0);
		list.Push(&ast.Field{nil, idents, typ, nil});

		for P.tok == token.COMMA {
			P.next();
			idents := P.parseIdentList(nil);
			typ := P.parseParameterType();
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
func (P *Parser) parseParameters(ellipsis_ok bool) []*ast.Field {
	if P.trace {
		defer un(trace(P, "Parameters"));
	}

	var params []*ast.Field;
	P.expect(token.LPAREN);
	if P.tok != token.RPAREN {
		params = P.parseParameterList(ellipsis_ok);
	}
	P.expect(token.RPAREN);

	return params;
}


func (P *Parser) parseResult() []*ast.Field {
	if P.trace {
		defer un(trace(P, "Result"));
	}

	var result []*ast.Field;
	if P.tok == token.LPAREN {
		result = P.parseParameters(false);
	} else if P.tok != token.FUNC {
		typ := P.tryType();
		if typ != nil {
			result = make([]*ast.Field, 1);
			result[0] = &ast.Field{nil, nil, typ, nil};
		}
	}

	return result;
}


// Function types
//
// (params)
// (params) type
// (params) (results)

func (P *Parser) parseSignature() *ast.Signature {
	if P.trace {
		defer un(trace(P, "Signature"));
	}

	params := P.parseParameters(true);  // TODO find better solution
	//t.End = P.pos;
	result := P.parseResult();

	return &ast.Signature{params, result};
}


func (P *Parser) parseFunctionType() *ast.FunctionType {
	if P.trace {
		defer un(trace(P, "FunctionType"));
	}

	pos := P.expect(token.FUNC);
	sig := P.parseSignature();

	return &ast.FunctionType{pos, sig};
}


func (P *Parser) parseMethodSpec() *ast.Field {
	if P.trace {
		defer un(trace(P, "MethodSpec"));
	}

	doc := P.getDoc();
	var idents []*ast.Ident;
	var typ ast.Expr;
	x := P.parseQualifiedIdent();
	if tmp, is_ident := x.(*ast.Ident); is_ident && (P.tok == token.COMMA || P.tok == token.LPAREN) {
		// method(s)
		idents = P.parseIdentList(x);
		typ = &ast.FunctionType{nopos, P.parseSignature()};
	} else {
		// embedded interface
		typ = x;
	}

	return &ast.Field{doc, idents, typ, nil};
}


func (P *Parser) parseInterfaceType() *ast.InterfaceType {
	if P.trace {
		defer un(trace(P, "InterfaceType"));
	}

	pos := P.expect(token.INTERFACE);
	var lbrace, rbrace Position;
	var methods []*ast.Field;
	if P.tok == token.LBRACE {
		lbrace = P.pos;
		P.next();

		list := vector.New(0);
		for P.tok == token.IDENT {
			list.Push(P.parseMethodSpec());
			if P.tok != token.RBRACE {
				P.expect(token.SEMICOLON);
			}
		}

		rbrace = P.expect(token.RBRACE);
		P.opt_semi = true;

		// convert vector
		methods = make([]*ast.Field, list.Len());
		for i := list.Len() - 1; i >= 0; i-- {
			methods[i] = list.At(i).(*ast.Field);
		}
	}

	return &ast.InterfaceType{pos, lbrace, methods, rbrace};
}


func (P *Parser) parseMapType() *ast.MapType {
	if P.trace {
		defer un(trace(P, "MapType"));
	}

	pos := P.expect(token.MAP);
	P.expect(token.LBRACK);
	key := P.parseVarType();
	P.expect(token.RBRACK);
	value := P.parseVarType();

	return &ast.MapType{pos, key, value};
}


func (P *Parser) parseStringLit(x *ast.BasicLit) *ast.StringLit

func (P *Parser) parseFieldDecl() *ast.Field {
	if P.trace {
		defer un(trace(P, "FieldDecl"));
	}

	doc := P.getDoc();

	// a list of identifiers looks like a list of type names
	list := vector.New(0);
	for {
		// TODO do not allow ()'s here
		list.Push(P.parseType());
		if P.tok == token.COMMA {
			P.next();
		} else {
			break;
		}
	}

	// if we had a list of identifiers, it must be followed by a type
	typ := P.tryType();

	// optional tag
	var tag ast.Expr;
	if P.tok == token.STRING {
		tag = P.parseStringLit(nil);
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
				P.error(list.At(i).(ast.Expr).Pos(), "identifier expected");
			}
		}
	} else {
		// anonymous field
		if list.Len() == 1 {
			// TODO should do more checks here
			typ = list.At(0).(ast.Expr);
		} else {
			P.error(P.pos, "anonymous field expected");
		}
	}

	return &ast.Field{doc, idents, typ, tag};
}


func (P *Parser) parseStructType() *ast.StructType {
	if P.trace {
		defer un(trace(P, "StructType"));
	}

	pos := P.expect(token.STRUCT);
	var lbrace, rbrace Position;
	var fields []*ast.Field;
	if P.tok == token.LBRACE {
		lbrace = P.pos;
		P.next();

		list := vector.New(0);
		for P.tok != token.RBRACE && P.tok != token.EOF {
			list.Push(P.parseFieldDecl());
			if P.tok == token.SEMICOLON {
				P.next();
			} else {
				break;
			}
		}
		if P.tok == token.SEMICOLON {
			P.next();
		}

		rbrace = P.expect(token.RBRACE);
		P.opt_semi = true;

		// convert vector
		fields = make([]*ast.Field, list.Len());
		for i := list.Len() - 1; i >= 0; i-- {
			fields[i] = list.At(i).(*ast.Field);
		}
	}

	return &ast.StructType{pos, lbrace, fields, rbrace};
}


func (P *Parser) parsePointerType() *ast.StarExpr {
	if P.trace {
		defer un(trace(P, "PointerType"));
	}

	star := P.expect(token.MUL);
	base := P.parseType();

	return &ast.StarExpr{star, base};
}


func (P *Parser) tryType() ast.Expr {
	if P.trace {
		defer un(trace(P, "Type (try)"));
	}

	switch P.tok {
	case token.IDENT: return P.parseTypeName();
	case token.LBRACK: return P.parseArrayType();
	case token.CHAN, token.ARROW: return P.parseChannelType();
	case token.INTERFACE: return P.parseInterfaceType();
	case token.FUNC: return P.parseFunctionType();
	case token.MAP: return P.parseMapType();
	case token.STRUCT: return P.parseStructType();
	case token.MUL: return P.parsePointerType();
	case token.LPAREN:
		lparen := P.pos;
		P.next();
		x := P.parseType();
		rparen := P.expect(token.RPAREN);
		return &ast.ParenExpr{lparen, x, rparen};
	}

	// no type found
	return nil;
}


// ----------------------------------------------------------------------------
// Blocks

func asStatList(list *vector.Vector) []ast.Stat {
	stats := make([]ast.Stat, list.Len());
	for i := 0; i < list.Len(); i++ {
		stats[i] = list.At(i).(ast.Stat);
	}
	return stats;
}


func (P *Parser) parseStatementList() []ast.Stat {
	if P.trace {
		defer un(trace(P, "StatementList"));
	}

	list := vector.New(0);
	expect_semi := false;
	for P.tok != token.CASE && P.tok != token.DEFAULT && P.tok != token.RBRACE && P.tok != token.EOF {
		if expect_semi {
			P.expect(token.SEMICOLON);
			expect_semi = false;
		}
		list.Push(P.parseStatement());
		if P.tok == token.SEMICOLON {
			P.next();
		} else if P.opt_semi {
			P.opt_semi = false;  // "consume" optional semicolon
		} else {
			expect_semi = true;
		}
	}
	
	return asStatList(list);
}


func (P *Parser) parseBlock(tok int) *ast.Block {
	if P.trace {
		defer un(trace(P, "Block"));
	}

	pos := P.expect(tok);
	list := P.parseStatementList();
	var end scanner.Location;
	if tok == token.LBRACE {
		end = P.expect(token.RBRACE);
		P.opt_semi = true;
	}

	return &ast.Block{pos, tok, list, end};
}


// ----------------------------------------------------------------------------
// Expressions

func (P *Parser) parseFunctionLit() ast.Expr {
	if P.trace {
		defer un(trace(P, "FunctionLit"));
	}

	pos := P.expect(token.FUNC);
	typ := P.parseSignature();
	P.expr_lev++;
	body := P.parseBlock(token.LBRACE);
	P.expr_lev--;

	return &ast.FunctionLit{pos, typ, body};
}


func (P *Parser) parseStringLit(x *ast.BasicLit) *ast.StringLit {
	if P.trace {
		defer un(trace(P, "StringLit"));
	}

	list := vector.New(0);
	if x != nil {
		list.Push(x);
	}
	
	for P.tok == token.STRING {
		list.Push(&ast.BasicLit{P.pos, token.STRING, P.val});
		P.next();
	}

	// convert list
	strings := make([]*ast.BasicLit, list.Len());
	for i := 0; i < list.Len(); i++ {
		strings[i] = list.At(i).(*ast.BasicLit);
	}
	
	return &ast.StringLit{strings};
}


func (P *Parser) parseOperand() ast.Expr {
	if P.trace {
		defer un(trace(P, "Operand"));
	}

	switch P.tok {
	case token.IDENT:
		return P.parseIdent();

	case token.INT, token.FLOAT, token.CHAR:
		x := &ast.BasicLit{P.pos, P.tok, P.val};
		P.next();
		return x;

	case token.STRING:
		x := &ast.BasicLit{P.pos, token.STRING, P.val};
		P.next();
		if P.tok == token.STRING {
			return P.parseStringLit(x);
		}
		return x;

	case token.LPAREN:
		lparen := P.pos;
		P.next();
		P.expr_lev++;
		x := P.parseExpression(1);
		P.expr_lev--;
		rparen := P.expect(token.RPAREN);
		return &ast.ParenExpr{lparen, x, rparen};

	case token.FUNC:
		return P.parseFunctionLit();

	default:
		t := P.tryType();
		if t != nil {
			return t;
		} else {
			P.error(P.pos, "operand expected");
			P.next();  // make progress
		}
	}

	return &ast.BadExpr{P.pos};
}


func (P *Parser) parseSelectorOrTypeAssertion(x ast.Expr) ast.Expr {
	if P.trace {
		defer un(trace(P, "SelectorOrTypeAssertion"));
	}

	P.expect(token.PERIOD);
	if P.tok == token.IDENT {
		// selector
		sel := P.parseIdent();
		return &ast.SelectorExpr{x, sel};
		
	} else {
		// type assertion
		P.expect(token.LPAREN);
		var typ ast.Expr;
		if P.tok == token.TYPE {
			// special case for type switch syntax
			typ = &ast.Ident{P.pos, P.val};
			P.next();
		} else {
			typ = P.parseType();
		}
		P.expect(token.RPAREN);
		return &ast.TypeAssertExpr{x, typ};
	}

	unreachable();
	return nil;
}


func (P *Parser) parseIndexOrSlice(x ast.Expr) ast.Expr {
	if P.trace {
		defer un(trace(P, "IndexOrSlice"));
	}

	P.expect(token.LBRACK);
	P.expr_lev++;
	index := P.parseExpression(1);
	P.expr_lev--;

	if P.tok == token.RBRACK {
		// index
		P.next();
		return &ast.IndexExpr{x, index};
	}
	
	// slice
	P.expect(token.COLON);
	P.expr_lev++;
	end := P.parseExpression(1);
	P.expr_lev--;
	P.expect(token.RBRACK);
	return &ast.SliceExpr{x, index, end};
}


func (P *Parser) parseCall(fun ast.Expr) *ast.CallExpr {
	if P.trace {
		defer un(trace(P, "Call"));
	}

	lparen := P.expect(token.LPAREN);
	var args []ast.Expr;
	if P.tok != token.RPAREN {
		args = P.parseExpressionList();
	}
	rparen := P.expect(token.RPAREN);
	return &ast.CallExpr{fun, lparen, args, rparen};
}


func (P *Parser) parseElementList() []ast.Expr {
	if P.trace {
		defer un(trace(P, "ElementList"));
	}

	list := vector.New(0);
	singles := true;
	for P.tok != token.RBRACE {
		x := P.parseExpression(0);
		if list.Len() == 0 {
			// first element determines syntax for remaining elements
			if t, is_binary := x.(*ast.BinaryExpr); is_binary && t.Tok == token.COLON {
				singles = false;
			}
		} else {
			// not the first element - check syntax
			if singles {
				if t, is_binary := x.(*ast.BinaryExpr); is_binary && t.Tok == token.COLON {
					P.error(t.X.Pos(), "single value expected; found pair");
				}
			} else {
				if t, is_binary := x.(*ast.BinaryExpr); !is_binary || t.Tok != token.COLON {
					P.error(x.Pos(), "key:value pair expected; found single value");
				}
			}
		}

		list.Push(x);

		if P.tok == token.COMMA {
			P.next();
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


func (P *Parser) parseCompositeLit(typ ast.Expr) ast.Expr {
	if P.trace {
		defer un(trace(P, "CompositeLit"));
	}

	lbrace := P.expect(token.LBRACE);
	var elts []ast.Expr;
	if P.tok != token.RBRACE {
		elts = P.parseElementList();
	}
	rbrace := P.expect(token.RBRACE);
	return &ast.CompositeLit{typ, lbrace, elts, rbrace};
}


func (P *Parser) parsePrimaryExpr() ast.Expr {
	if P.trace {
		defer un(trace(P, "PrimaryExpr"));
	}

	x := P.parseOperand();
	for {
		switch P.tok {
		case token.PERIOD: x = P.parseSelectorOrTypeAssertion(x);
		case token.LBRACK: x = P.parseIndexOrSlice(x);
		case token.LPAREN: x = P.parseCall(x);
		case token.LBRACE:
			if P.expr_lev >= 0 {
				x = P.parseCompositeLit(x);
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


func (P *Parser) parseUnaryExpr() ast.Expr {
	if P.trace {
		defer un(trace(P, "UnaryExpr"));
	}

	switch P.tok {
	case token.ADD, token.SUB, token.NOT, token.XOR, token.ARROW, token.AND, token.RANGE:
		pos, tok := P.pos, P.tok;
		P.next();
		x := P.parseUnaryExpr();
		return &ast.UnaryExpr{pos, tok, x};

	case token.MUL:
		// unary "*" expression or pointer type
		pos := P.pos;
		P.next();
		x := P.parseUnaryExpr();
		return &ast.StarExpr{pos, x};
	}

	return P.parsePrimaryExpr();
}


func (P *Parser) parseBinaryExpr(prec1 int) ast.Expr {
	if P.trace {
		defer un(trace(P, "BinaryExpr"));
	}

	x := P.parseUnaryExpr();
	for prec := token.Precedence(P.tok); prec >= prec1; prec-- {
		for token.Precedence(P.tok) == prec {
			pos, tok := P.pos, P.tok;
			P.next();
			y := P.parseBinaryExpr(prec + 1);
			x = &ast.BinaryExpr{x, pos, tok, y};
		}
	}

	return x;
}


func (P *Parser) parseExpression(prec int) ast.Expr {
	if P.trace {
		defer un(trace(P, "Expression"));
	}

	if prec < 0 {
		panic("precedence must be >= 0");
	}

	return P.parseBinaryExpr(prec);
}


// ----------------------------------------------------------------------------
// Statements


func (P *Parser) parseSimpleStat() ast.Stat {
	if P.trace {
		defer un(trace(P, "SimpleStat"));
	}

	x := P.parseExpressionList();

	switch P.tok {
	case token.COLON:
		// labeled statement
		P.expect(token.COLON);
		if len(x) == 1 {
			if label, is_ident := x[0].(*ast.Ident); is_ident {
				return &ast.LabeledStat{label, P.parseStatement()};
			}
		}
		P.error(x[0].Pos(), "illegal label declaration");
		return nil;

	case
		token.DEFINE, token.ASSIGN, token.ADD_ASSIGN,
		token.SUB_ASSIGN, token.MUL_ASSIGN, token.QUO_ASSIGN,
		token.REM_ASSIGN, token.AND_ASSIGN, token.OR_ASSIGN,
		token.XOR_ASSIGN, token.SHL_ASSIGN, token.SHR_ASSIGN:
		// assignment statement or range clause
		pos, tok := P.pos, P.tok;
		P.next();
		/*
		if mode & range_ok != 0 && P.tok == token.RANGE {
			// range clause
			P.next();
			if len(x) != 1 && len(x) != 2 {
				P.error(x[0].Pos(), "expected 1 or 2 expressions on lhs of range clause");
			}
			if tok != token.DEFINE && tok != token.ASSIGN {
				P.error(pos, "expected '=' or ':=', found '" + token.TokenString(tok) + "'");
			}
			y := P.parseExpression(1);
			return &ast.RangeClause{x, pos, tok, y};
		} else {
		*/
		// assignment statement
		y := P.parseExpressionList();
		xl, yl := len(x), len(y);
		if xl > 1 && yl > 1 && xl != yl {
			P.error(x[0].Pos(), "arity of lhs doesn't match rhs");  // TODO use better loc for error
		}
		return &ast.AssignmentStat{x, pos, tok, y};

	default:
		if len(x) > 1 {
			P.error(x[0].Pos(), "only one expression allowed");
		}

		if P.tok == token.INC || P.tok == token.DEC {
			s := &ast.IncDecStat{x[0], P.tok};
			P.next();  // consume "++" or "--"
			return s;
		}

		return &ast.ExprStat{x[0]};
	}

	unreachable();
	return nil;
}


func (P *Parser) parseGoStat() *ast.GoStat {
	if P.trace {
		defer un(trace(P, "GoStat"));
	}

	pos := P.expect(token.GO);
	call := P.parseExpression(1);
	return &ast.GoStat{pos, call};
}


func (P *Parser) parseDeferStat() *ast.DeferStat {
	if P.trace {
		defer un(trace(P, "DeferStat"));
	}

	pos := P.expect(token.DEFER);
	call := P.parseExpression(1);
	return &ast.DeferStat{pos, call};
}


func (P *Parser) parseReturnStat() *ast.ReturnStat {
	if P.trace {
		defer un(trace(P, "ReturnStat"));
	}

	loc := P.pos;
	P.expect(token.RETURN);
	var x []ast.Expr;
	if P.tok != token.SEMICOLON && P.tok != token.RBRACE {
		x = P.parseExpressionList();
	}

	return &ast.ReturnStat{loc, x};
}


func (P *Parser) parseControlFlowStat(tok int) *ast.ControlFlowStat {
	if P.trace {
		defer un(trace(P, "ControlFlowStat"));
	}

	s := &ast.ControlFlowStat{P.pos, tok, nil};
	P.expect(tok);
	if tok != token.FALLTHROUGH && P.tok == token.IDENT {
		s.Label = P.parseIdent();
	}

	return s;
}


/*
func (P *Parser) asIdent(x ast.Expr) *ast.Ident {
	if name, ok := x.(*ast.Ident); ok {
		return name;
	}
	P.error(x.Pos(), "identifier expected");
	return &ast.Ident{x.Pos(), [...]byte{'B', 'A', 'D'}};
}


func (P *Parser) isTypeSwitch(init ast.Stat) (lhs *ast.Ident, rhs ast.Expr) {
	if assign, ok := init.(*ast.AssignmentStat); ok {
		if guard, ok := assign.Rhs.(*ast.TypeAssertion); ok {
			if tmp, ok := guard.Typ.(*ast.TypeType); ok {
				// we appear to have a type switch
				// TODO various error checks
				return P.asIdent(assign.Lhs), guard.X;
			}
		}
	}
	return nil, nil;
}
*/



func (P *Parser) parseControlClause(isForStat bool) (s1, s2, s3 ast.Stat) {
	if P.trace {
		defer un(trace(P, "ControlClause"));
	}

	if P.tok != token.LBRACE {
		prev_lev := P.expr_lev;
		P.expr_lev = -1;

		if P.tok != token.SEMICOLON {
			s1 = P.parseSimpleStat();
		}
		if P.tok == token.SEMICOLON {
			P.next();
			if P.tok != token.LBRACE && P.tok != token.SEMICOLON {
				s2 = P.parseSimpleStat();
			}
			if isForStat {
				// for statements have a 3rd section
				P.expect(token.SEMICOLON);
				if P.tok != token.LBRACE {
					s3 = P.parseSimpleStat();
				}
			}
		} else {
			s1, s2 = nil, s1;
		}
		
		P.expr_lev = prev_lev;
	}

	return s1, s2, s3;
}


func (P *Parser) isExpr(s ast.Stat) bool {
	if s == nil {
		return true;
	}
	dummy, is_expr := s.(*ast.ExprStat);
	return is_expr;
}


func (P *Parser) asExpr(s ast.Stat) ast.Expr {
	if s == nil {
		return nil;
	}
	if es, is_expr := s.(*ast.ExprStat); is_expr {
		return es.X;
	}
	P.error(s.Pos(), "condition expected; found simple statement");
	return &ast.BadExpr{s.Pos()};
}


func (P *Parser) parseIfStat() *ast.IfStat {
	if P.trace {
		defer un(trace(P, "IfStat"));
	}

	pos := P.expect(token.IF);
	s1, s2, dummy := P.parseControlClause(false);
	body := P.parseBlock(token.LBRACE);
	var else_ ast.Stat;
	if P.tok == token.ELSE {
		P.next();
		else_ = P.parseStatement();
	}

	return &ast.IfStat{pos, s1, P.asExpr(s2), body, else_};
}


func (P *Parser) parseCaseClause() *ast.CaseClause {
	if P.trace {
		defer un(trace(P, "CaseClause"));
	}

	// SwitchCase
	loc := P.pos;
	var x []ast.Expr;
	if P.tok == token.CASE {
		P.next();
		x = P.parseExpressionList();
	} else {
		P.expect(token.DEFAULT);
	}

	return &ast.CaseClause{loc, x, P.parseBlock(token.COLON)};
}


func (P *Parser) parseTypeCaseClause() *ast.TypeCaseClause {
	if P.trace {
		defer un(trace(P, "CaseClause"));
	}

	// TypeSwitchCase
	pos := P.pos;
	var typ ast.Expr;
	if P.tok == token.CASE {
		P.next();
		typ = P.parseType();
	} else {
		P.expect(token.DEFAULT);
	}

	return &ast.TypeCaseClause{pos, typ, P.parseBlock(token.COLON)};
}


func (P *Parser) parseSwitchStat() ast.Stat {
	if P.trace {
		defer un(trace(P, "SwitchStat"));
	}

	pos := P.expect(token.SWITCH);
	s1, s2, dummy := P.parseControlClause(false);

	if P.isExpr(s2) {
		// expression switch
		lbrace := P.expect(token.LBRACE);
		cases := vector.New(0);
		for P.tok == token.CASE || P.tok == token.DEFAULT {
			cases.Push(P.parseCaseClause());
		}
		rbrace := P.expect(token.RBRACE);
		P.opt_semi = true;
		body := &ast.Block{lbrace, token.LBRACE, asStatList(cases), rbrace};
		return &ast.SwitchStat{pos, s1, P.asExpr(s2), body};

	} else {
		// type switch
		// TODO do all the checks!
		lbrace := P.expect(token.LBRACE);
		cases := vector.New(0);
		for P.tok == token.CASE || P.tok == token.DEFAULT {
			cases.Push(P.parseTypeCaseClause());
		}
		rbrace := P.expect(token.RBRACE);
		P.opt_semi = true;
		body := &ast.Block{lbrace, token.LBRACE, asStatList(cases), rbrace};
		return &ast.TypeSwitchStat{pos, s1, s2, body};
	}

	unreachable();
	return nil;
}


func (P *Parser) parseCommClause() *ast.CommClause {
	if P.trace {
		defer un(trace(P, "CommClause"));
	}

	// CommCase
	loc := P.pos;
	var tok int;
	var lhs, rhs ast.Expr;
	if P.tok == token.CASE {
		P.next();
		if P.tok == token.ARROW {
			// RecvExpr without assignment
			rhs = P.parseExpression(1);
		} else {
			// SendExpr or RecvExpr
			rhs = P.parseExpression(1);
			if P.tok == token.ASSIGN || P.tok == token.DEFINE {
				// RecvExpr with assignment
				tok = P.tok;
				P.next();
				lhs = rhs;
				if P.tok == token.ARROW {
					rhs = P.parseExpression(1);
				} else {
					P.expect(token.ARROW);  // use expect() error handling
				}
			}
			// else SendExpr
		}
	} else {
		P.expect(token.DEFAULT);
	}

	return &ast.CommClause{loc, tok, lhs, rhs, P.parseBlock(token.COLON)};
}


func (P *Parser) parseSelectStat() *ast.SelectStat {
	if P.trace {
		defer un(trace(P, "SelectStat"));
	}

	pos := P.expect(token.SELECT);
	lbrace := P.expect(token.LBRACE);
	cases := vector.New(0);
	for P.tok == token.CASE || P.tok == token.DEFAULT {
		cases.Push(P.parseCommClause());
	}
	rbrace := P.expect(token.RBRACE);
	P.opt_semi = true;
	body := &ast.Block{lbrace, token.LBRACE, asStatList(cases), rbrace};

	return &ast.SelectStat{pos, body};
}


func (P *Parser) parseForStat() ast.Stat {
	if P.trace {
		defer un(trace(P, "ForStat"));
	}

	pos := P.expect(token.FOR);
	s1, s2, s3 := P.parseControlClause(true);
	body := P.parseBlock(token.LBRACE);

	if as, is_as := s2.(*ast.AssignmentStat); is_as {
		// probably a for statement with a range clause
		// TODO do all the checks!
		return &ast.RangeStat{pos, s2, body};
	} else {
		// regular for statement
		return &ast.ForStat{pos, s1, P.asExpr(s2), s3, body};
	}
	
	unreachable();
	return nil;
}


func (P *Parser) parseStatement() ast.Stat {
	if P.trace {
		defer un(trace(P, "Statement"));
	}

	switch P.tok {
	case token.CONST, token.TYPE, token.VAR:
		return &ast.DeclStat{P.parseDeclaration()};
	case
		// tokens that may start a top-level expression
		token.IDENT, token.INT, token.FLOAT, token.CHAR, token.STRING, token.FUNC, token.LPAREN,  // operand
		token.LBRACK, token.STRUCT,  // composite type
		token.MUL, token.AND, token.ARROW:  // unary operators
		return P.parseSimpleStat();
	case token.GO:
		return P.parseGoStat();
	case token.DEFER:
		return P.parseDeferStat();
	case token.RETURN:
		return P.parseReturnStat();
	case token.BREAK, token.CONTINUE, token.GOTO, token.FALLTHROUGH:
		return P.parseControlFlowStat(P.tok);
	case token.LBRACE:
		return &ast.CompositeStat{P.parseBlock(token.LBRACE)};
	case token.IF:
		return P.parseIfStat();
	case token.FOR:
		return P.parseForStat();
	case token.SWITCH:
		return P.parseSwitchStat();
	case token.SELECT:
		return P.parseSelectStat();
	case token.SEMICOLON, token.RBRACE:
		// don't consume the ";", it is the separator following the empty statement
		return &ast.EmptyStat{P.pos};
	}

	// no statement found
	P.error(P.pos, "statement expected");
	return &ast.BadStat{P.pos};
}


// ----------------------------------------------------------------------------
// Declarations

func (P *Parser) parseImportSpec(pos Position, doc ast.Comments) *ast.ImportDecl {
	if P.trace {
		defer un(trace(P, "ImportSpec"));
	}

	var ident *ast.Ident;
	if P.tok == token.PERIOD {
		P.error(P.pos, `"import ." not yet handled properly`);
		P.next();
	} else if P.tok == token.IDENT {
		ident = P.parseIdent();
	}

	var path *ast.StringLit;
	if P.tok == token.STRING {
		path = P.parseStringLit(nil);
	} else {
		P.expect(token.STRING);  // use expect() error handling
	}

	return &ast.ImportDecl{doc, pos, ident, path};
}


func (P *Parser) parseConstSpec(pos Position, doc ast.Comments) *ast.ConstDecl {
	if P.trace {
		defer un(trace(P, "ConstSpec"));
	}

	names := P.parseIdentList(nil);
	typ := P.tryType();
	var values []ast.Expr;
	if typ != nil || P.tok == token.ASSIGN {
		P.expect(token.ASSIGN);
		values = P.parseExpressionList();
	}

	return &ast.ConstDecl{doc, pos, names, typ, values};
}


func (P *Parser) parseTypeSpec(pos Position, doc ast.Comments) *ast.TypeDecl {
	if P.trace {
		defer un(trace(P, "TypeSpec"));
	}

	ident := P.parseIdent();
	typ := P.parseType();

	return &ast.TypeDecl{doc, pos, ident, typ};
}


func (P *Parser) parseVarSpec(pos Position, doc ast.Comments) *ast.VarDecl {
	if P.trace {
		defer un(trace(P, "VarSpec"));
	}

	names := P.parseIdentList(nil);
	typ := P.tryType();
	var values []ast.Expr;
	if typ == nil || P.tok == token.ASSIGN {
		P.expect(token.ASSIGN);
		values = P.parseExpressionList();
	}

	return &ast.VarDecl{doc, pos, names, typ, values};
}


func (P *Parser) parseSpec(pos Position, doc ast.Comments, keyword int) ast.Decl {
	switch keyword {
	case token.IMPORT: return P.parseImportSpec(pos, doc);
	case token.CONST: return P.parseConstSpec(pos, doc);
	case token.TYPE: return P.parseTypeSpec(pos, doc);
	case token.VAR: return P.parseVarSpec(pos, doc);
	}

	unreachable();
	return nil;
}


func (P *Parser) parseDecl(keyword int) ast.Decl {
	if P.trace {
		defer un(trace(P, "Decl"));
	}

	doc := P.getDoc();
	pos := P.expect(keyword);
	if P.tok == token.LPAREN {
		lparen := P.pos;
		P.next();
		list := vector.New(0);
		for P.tok != token.RPAREN && P.tok != token.EOF {
			list.Push(P.parseSpec(nopos, nil, keyword));
			if P.tok == token.SEMICOLON {
				P.next();
			} else {
				break;
			}
		}
		rparen := P.expect(token.RPAREN);
		P.opt_semi = true;

		// convert vector
		decls := make([]ast.Decl, list.Len());
		for i := 0; i < list.Len(); i++ {
			decls[i] = list.At(i).(ast.Decl);
		}

		return &ast.DeclList{doc, pos, keyword, lparen, decls, rparen};
	}

	return P.parseSpec(pos, doc, keyword);
}


// Function and method declarations
//
// func        ident (params)
// func        ident (params) type
// func        ident (params) (results)
// func (recv) ident (params)
// func (recv) ident (params) type
// func (recv) ident (params) (results)

func (P *Parser) parseFunctionDecl() *ast.FuncDecl {
	if P.trace {
		defer un(trace(P, "FunctionDecl"));
	}

	doc := P.getDoc();
	pos := P.expect(token.FUNC);

	var recv *ast.Field;
	if P.tok == token.LPAREN {
		loc := P.pos;
		tmp := P.parseParameters(true);
		if len(tmp) == 1 {
			recv = tmp[0];
		} else {
			P.error(loc, "must have exactly one receiver");
		}
	}

	ident := P.parseIdent();
	sig := P.parseSignature();

	var body *ast.Block;
	if P.tok == token.LBRACE {
		body = P.parseBlock(token.LBRACE);
	}

	return &ast.FuncDecl{doc, pos, recv, ident, sig, body};
}


func (P *Parser) parseDeclaration() ast.Decl {
	if P.trace {
		defer un(trace(P, "Declaration"));
	}

	switch P.tok {
	case token.CONST, token.TYPE, token.VAR:
		return P.parseDecl(P.tok);
	case token.FUNC:
		return P.parseFunctionDecl();
	}

	loc := P.pos;
	P.error(loc, "declaration expected");
	P.next();  // make progress
	return &ast.BadDecl{loc};
}


// ----------------------------------------------------------------------------
// Program

// The Parse function is parametrized with one of the following
// constants. They control how much of the source text is parsed.
//
const (
	ParseEntirePackage = iota;
	ParseImportDeclsOnly;
	ParsePackageClauseOnly;
)


// Parse parses the source...
//      
// foo bar
//
func (P *Parser) Parse(mode int) *ast.Package {
	if P.trace {
		defer un(trace(P, "Program"));
	}

	// package clause
	comment := P.getDoc();
	pos := P.expect(token.PACKAGE);
	name := P.parseIdent();
	if P.tok == token.SEMICOLON {
		// common error
		P.error(P.pos, "extra semicolon");
		P.next();
	}
	
	
	var decls []ast.Decl;
	if mode <= ParseImportDeclsOnly {
		// import decls
		list := vector.New(0);
		for P.tok == token.IMPORT {
			list.Push(P.parseDecl(token.IMPORT));
			if P.tok == token.SEMICOLON {
				P.next();
			}
		}

		if mode <= ParseEntirePackage {
			// rest of package body
			for P.tok != token.EOF {
				list.Push(P.parseDeclaration());
				if P.tok == token.SEMICOLON {
					P.next();
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
	comments := make([]*ast.Comment, P.comments.Len());
	for i := 0; i < P.comments.Len(); i++ {
		c := P.comments.At(i);
		if c != nil {
			comments[i] = c.(*ast.Comment);
		}
	}

	return &ast.Package{comment, pos, name, decls, comments};
}
