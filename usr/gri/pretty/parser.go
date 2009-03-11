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

	comments *vector.Vector;

	// The next token
	loc scanner.Location;  // token location
	tok int;  // one token look-ahead
	val []byte;  // token value

	// Non-syntactic parser control
	opt_semi bool;  // true if semicolon separator is optional in statement list

	// Nesting levels
	expr_lev int;  // < 0: in control clause, >= 0: in expression
};


// When we don't have a location use noloc.
// TODO make sure we always have a location.
var noloc scanner.Location;


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
	P.loc, P.tok, P.val = P.scanner.Scan();
	P.opt_semi = false;

	if P.trace {
		P.printIndent();
		switch P.tok {
		case token.IDENT, token.INT, token.FLOAT, token.CHAR, token.STRING:
			fmt.Printf("%d:%d: %s = %s\n", P.loc.Line, P.loc.Col, token.TokenString(P.tok), P.val);
		case token.LPAREN:
			// don't print '(' - screws up selection in terminal window
			fmt.Printf("%d:%d: LPAREN\n", P.loc.Line, P.loc.Col);
		case token.RPAREN:
			// don't print ')' - screws up selection in terminal window
			fmt.Printf("%d:%d: RPAREN\n", P.loc.Line, P.loc.Col);
		default:
			fmt.Printf("%d:%d: %s\n", P.loc.Line, P.loc.Col, token.TokenString(P.tok));
		}
	}
}


func (P *Parser) next() {
	for P.next0(); P.tok == token.COMMENT; P.next0() {
		P.comments.Push(&ast.Comment{P.loc, P.val});
	}
}


func (P *Parser) Init(scanner *scanner.Scanner, err scanner.ErrorHandler, trace bool) {
	P.scanner = scanner;
	P.err = err;

	P.trace = trace;
	P.indent = 0;

	P.comments = vector.New(0);

	P.next();
	P.expr_lev = 0;
}


func (P *Parser) error(loc scanner.Location, msg string) {
	P.err.Error(loc, msg);
}


func (P *Parser) expect(tok int) {
	if P.tok != tok {
		msg := "expected '" + token.TokenString(tok) + "', found '" + token.TokenString(P.tok) + "'";
		if token.IsLiteral(P.tok) {
			msg += " " + string(P.val);
		}
		P.error(P.loc, msg);
	}
	P.next();  // make progress in any case
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
		x := &ast.Ident{P.loc, string(P.val)};
		P.next();
		return x;
	}

	P.expect(token.IDENT);  // use expect() error handling
	return &ast.Ident{P.loc, ""};
}


func (P *Parser) parseIdentList(x ast.Expr) ast.Expr {
	if P.trace {
		defer un(trace(P, "IdentList"));
	}

	var last *ast.BinaryExpr;
	if x == nil {
		x = P.parseIdent();
	}
	for P.tok == token.COMMA {
		loc := P.loc;
		P.next();
		y := P.parseIdent();
		if last == nil {
			last = &ast.BinaryExpr{loc, token.COMMA, x, y};
			x = last;
		} else {
			last.Y = &ast.BinaryExpr{loc, token.COMMA, last.Y, y};
			last = last.Y.(*ast.BinaryExpr);
		}
	}

	return x;
}


func (P *Parser) parseIdentList2(x ast.Expr) []*ast.Ident {
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


// ----------------------------------------------------------------------------
// Types

func (P *Parser) parseType() ast.Expr {
	if P.trace {
		defer un(trace(P, "Type"));
	}

	t := P.tryType();
	if t == nil {
		P.error(P.loc, "type expected");
		t = &ast.BadExpr{P.loc};
	}

	return t;
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
		loc := P.loc;
		P.next();
		y := P.parseIdent();
		x = &ast.Selector{loc, x, y};
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

	loc := P.loc;
	P.expect(token.LBRACK);
	var len ast.Expr;
	if P.tok == token.ELLIPSIS {
		len = &ast.Ellipsis{P.loc};
		P.next();
	} else if P.tok != token.RBRACK {
		len = P.parseExpression(1);
	}
	P.expect(token.RBRACK);
	elt := P.parseType();

	return &ast.ArrayType{loc, len, elt};
}


func (P *Parser) parseChannelType() *ast.ChannelType {
	if P.trace {
		defer un(trace(P, "ChannelType"));
	}

	loc := P.loc;
	mode := ast.FULL;
	if P.tok == token.CHAN {
		P.next();
		if P.tok == token.ARROW {
			P.next();
			mode = ast.SEND;
		}
	} else {
		P.expect(token.ARROW);
		P.expect(token.CHAN);
		mode = ast.RECV;
	}
	val := P.parseVarType();

	return &ast.ChannelType{loc, mode, val};
}


func (P *Parser) tryParameterType() ast.Expr {
	if P.tok == token.ELLIPSIS {
		loc  := P.loc;
		P.next();
		return &ast.Ellipsis{loc};
	}
	return P.tryType();
}


func (P *Parser) parseParameterType() ast.Expr {
	typ := P.tryParameterType();
	if typ == nil {
		P.error(P.loc, "type expected");
		typ = &ast.BadExpr{P.loc};
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
		list.Push(&ast.Field{idents, typ, nil});

		for P.tok == token.COMMA {
			P.next();
			idents := P.parseIdentList2(nil);
			typ := P.parseParameterType();
			list.Push(&ast.Field{idents, typ, nil});
		}

	} else {
		// Type { "," Type }
		// convert list of types into list of *Param
		for i := 0; i < list.Len(); i++ {
			list.Set(i, &ast.Field{nil, list.At(i).(ast.Expr), nil});
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
			result[0] = &ast.Field{nil, typ, nil};
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
	//t.End = P.loc;
	result := P.parseResult();

	return &ast.Signature{params, result};
}


func (P *Parser) parseFunctionType() *ast.FunctionType {
	if P.trace {
		defer un(trace(P, "FunctionType"));
	}

	loc := P.loc;
	P.expect(token.FUNC);
	sig := P.parseSignature();

	return &ast.FunctionType{loc, sig};
}


func (P *Parser) parseMethodSpec() *ast.Field {
	if P.trace {
		defer un(trace(P, "MethodSpec"));
	}

	var idents []*ast.Ident;
	var typ ast.Expr;
	x := P.parseQualifiedIdent();
	if tmp, is_ident := x.(*ast.Ident); is_ident && (P.tok == token.COMMA || P.tok == token.LPAREN) {
		// method(s)
		idents = P.parseIdentList2(x);
		typ = &ast.FunctionType{noloc, P.parseSignature()};
	} else {
		// embedded interface
		typ = x;
	}

	return &ast.Field{idents, typ, nil};
}


func (P *Parser) parseInterfaceType() *ast.InterfaceType {
	if P.trace {
		defer un(trace(P, "InterfaceType"));
	}

	loc := P.loc;
	var end scanner.Location;
	var methods []*ast.Field;

	P.expect(token.INTERFACE);
	if P.tok == token.LBRACE {
		P.next();

		list := vector.New(0);
		for P.tok == token.IDENT {
			list.Push(P.parseMethodSpec());
			if P.tok != token.RBRACE {
				P.expect(token.SEMICOLON);
			}
		}

		end = P.loc;
		P.expect(token.RBRACE);
		P.opt_semi = true;

		// convert vector
		methods = make([]*ast.Field, list.Len());
		for i := list.Len() - 1; i >= 0; i-- {
			methods[i] = list.At(i).(*ast.Field);
		}
	}

	return &ast.InterfaceType{loc, methods, end};
}


func (P *Parser) parseMapType() *ast.MapType {
	if P.trace {
		defer un(trace(P, "MapType"));
	}

	loc := P.loc;
	P.expect(token.MAP);
	P.expect(token.LBRACK);
	key := P.parseVarType();
	P.expect(token.RBRACK);
	val := P.parseVarType();

	return &ast.MapType{loc, key, val};
}


func (P *Parser) parseStringLit() ast.Expr

func (P *Parser) parseFieldDecl() *ast.Field {
	if P.trace {
		defer un(trace(P, "FieldDecl"));
	}

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
		tag = P.parseStringLit();
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
				P.error(list.At(i).(ast.Expr).Loc(), "identifier expected");
			}
		}
	} else {
		// anonymous field
		if list.Len() == 1 {
			// TODO should do more checks here
			typ = list.At(0).(ast.Expr);
		} else {
			P.error(P.loc, "anonymous field expected");
		}
	}

	return &ast.Field{idents, typ, tag};
}


func (P *Parser) parseStructType() ast.Expr {
	if P.trace {
		defer un(trace(P, "StructType"));
	}

	loc := P.loc;
	var end scanner.Location;
	var fields []*ast.Field;

	P.expect(token.STRUCT);
	if P.tok == token.LBRACE {
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

		end = P.loc;
		P.expect(token.RBRACE);
		P.opt_semi = true;

		// convert vector
		fields = make([]*ast.Field, list.Len());
		for i := list.Len() - 1; i >= 0; i-- {
			fields[i] = list.At(i).(*ast.Field);
		}
	}

	return &ast.StructType{loc, fields, end};
}


func (P *Parser) parsePointerType() ast.Expr {
	if P.trace {
		defer un(trace(P, "PointerType"));
	}

	loc := P.loc;
	P.expect(token.MUL);
	base := P.parseType();

	return &ast.PointerType{loc, base};
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
		loc := P.loc;
		P.next();
		t := P.parseType();
		P.expect(token.RPAREN);
		return &ast.Group{loc, t};
	}

	// no type found
	return nil;
}


// ----------------------------------------------------------------------------
// Blocks


func (P *Parser) parseStatementList(list *vector.Vector) {
	if P.trace {
		defer un(trace(P, "StatementList"));
	}

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
}


func (P *Parser) parseBlock(tok int) *ast.Block {
	if P.trace {
		defer un(trace(P, "Block"));
	}

	b := ast.NewBlock(P.loc, tok);
	P.expect(tok);

	P.parseStatementList(b.List);

	if tok == token.LBRACE {
		b.End = P.loc;
		P.expect(token.RBRACE);
		P.opt_semi = true;
	}

	return b;
}


// ----------------------------------------------------------------------------
// Expressions

func (P *Parser) parseExpressionList() ast.Expr {
	if P.trace {
		defer un(trace(P, "ExpressionList"));
	}

	x := P.parseExpression(1);
	for first := true; P.tok == token.COMMA; {
		loc := P.loc;
		P.next();
		y := P.parseExpression(1);
		if first {
			x = &ast.BinaryExpr{loc, token.COMMA, x, y};
			first = false;
		} else {
			x.(*ast.BinaryExpr).Y = &ast.BinaryExpr{loc, token.COMMA, x.(*ast.BinaryExpr).Y, y};
		}
	}

	return x;
}


func (P *Parser) parseFunctionLit() ast.Expr {
	if P.trace {
		defer un(trace(P, "FunctionLit"));
	}

	loc := P.loc;
	P.expect(token.FUNC);
	typ := P.parseSignature();
	P.expr_lev++;
	body := P.parseBlock(token.LBRACE);
	P.expr_lev--;

	return &ast.FunctionLit{loc, typ, body};
}


func (P *Parser) parseStringLit() ast.Expr {
	if P.trace {
		defer un(trace(P, "StringLit"));
	}

	var x ast.Expr = &ast.BasicLit{P.loc, P.tok, P.val};
	P.expect(token.STRING);  // always satisfied

	for P.tok == token.STRING {
		y := &ast.BasicLit{P.loc, P.tok, P.val};
		P.next();
		x = &ast.ConcatExpr{x, y};
	}

	return x;
}


func (P *Parser) parseOperand() ast.Expr {
	if P.trace {
		defer un(trace(P, "Operand"));
	}

	switch P.tok {
	case token.IDENT:
		return P.parseIdent();

	case token.INT, token.FLOAT, token.CHAR:
		x := &ast.BasicLit{P.loc, P.tok, P.val};
		P.next();
		return x;

	case token.STRING:
		return P.parseStringLit();

	case token.LPAREN:
		loc := P.loc;
		P.next();
		P.expr_lev++;
		x := P.parseExpression(1);
		P.expr_lev--;
		P.expect(token.RPAREN);
		return &ast.Group{loc, x};

	case token.FUNC:
		return P.parseFunctionLit();

	default:
		t := P.tryType();
		if t != nil {
			return t;
		} else {
			P.error(P.loc, "operand expected");
			P.next();  // make progress
		}
	}

	return &ast.BadExpr{P.loc};
}


func (P *Parser) parseSelectorOrTypeGuard(x ast.Expr) ast.Expr {
	if P.trace {
		defer un(trace(P, "SelectorOrTypeGuard"));
	}

	loc := P.loc;
	P.expect(token.PERIOD);

	if P.tok == token.IDENT {
		x = &ast.Selector{loc, x, P.parseIdent()};

	} else {
		P.expect(token.LPAREN);
		var typ ast.Expr;
		if P.tok == token.TYPE {
			typ = &ast.TypeType{P.loc};
			P.next();
		} else {
			typ = P.parseType();
		}
		x = &ast.TypeGuard{loc, x, typ};
		P.expect(token.RPAREN);
	}

	return x;
}


func (P *Parser) parseIndex(x ast.Expr) ast.Expr {
	if P.trace {
		defer un(trace(P, "IndexOrSlice"));
	}

	loc := P.loc;
	P.expect(token.LBRACK);
	P.expr_lev++;
	i := P.parseExpression(0);
	P.expr_lev--;
	P.expect(token.RBRACK);

	return &ast.Index{loc, x, i};
}


func (P *Parser) parseBinaryExpr(prec1 int) ast.Expr

func (P *Parser) parseCompositeElements(close int) ast.Expr {
	x := P.parseExpression(0);
	if P.tok == token.COMMA {
		loc := P.loc;
		P.next();

		// first element determines mode
		singles := true;
		if t, is_binary := x.(*ast.BinaryExpr); is_binary && t.Tok == token.COLON {
			singles = false;
		}

		var last *ast.BinaryExpr;
		for P.tok != close && P.tok != token.EOF {
			y := P.parseExpression(0);

			if singles {
				if t, is_binary := y.(*ast.BinaryExpr); is_binary && t.Tok == token.COLON {
					P.error(t.X.Loc(), "single value expected; found pair");
				}
			} else {
				if t, is_binary := y.(*ast.BinaryExpr); !is_binary || t.Tok != token.COLON {
					P.error(y.Loc(), "key:value pair expected; found single value");
				}
			}

			if last == nil {
				last = &ast.BinaryExpr{loc, token.COMMA, x, y};
				x = last;
			} else {
				last.Y = &ast.BinaryExpr{loc, token.COMMA, last.Y, y};
				last = last.Y.(*ast.BinaryExpr);
			}

			if P.tok == token.COMMA {
				loc = P.loc;
				P.next();
			} else {
				break;
			}

		}
	}
	return x;
}


func (P *Parser) parseCallOrCompositeLit(f ast.Expr, open, close int) ast.Expr {
	if P.trace {
		defer un(trace(P, "CallOrCompositeLit"));
	}

	loc := P.loc;
	P.expect(open);
	var args ast.Expr;
	if P.tok != close {
		args = P.parseCompositeElements(close);
	}
	P.expect(close);

	return &ast.Call{loc, open, f, args};
}


func (P *Parser) parsePrimaryExpr() ast.Expr {
	if P.trace {
		defer un(trace(P, "PrimaryExpr"));
	}

	x := P.parseOperand();
	for {
		switch P.tok {
		case token.PERIOD: x = P.parseSelectorOrTypeGuard(x);
		case token.LBRACK: x = P.parseIndex(x);
		// TODO fix once we have decided on literal/conversion syntax
		case token.LPAREN: x = P.parseCallOrCompositeLit(x, token.LPAREN, token.RPAREN);
		case token.LBRACE:
			if P.expr_lev >= 0 {
				x = P.parseCallOrCompositeLit(x, token.LBRACE, token.RBRACE);
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
	case token.ADD, token.SUB, token.MUL, token.NOT, token.XOR, token.ARROW, token.AND:
		loc, tok := P.loc, P.tok;
		P.next();
		y := P.parseUnaryExpr();
		return &ast.UnaryExpr{loc, tok, y};
		/*
		if lit, ok := y.(*ast.TypeLit); ok && tok == token.MUL {
			// pointer type
			t := ast.NewType(pos, ast.POINTER);
			t.Elt = lit.Typ;
			return &ast.TypeLit{t};
		} else {
			return &ast.UnaryExpr{loc, tok, y};
		}
		*/
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
			loc, tok := P.loc, P.tok;
			P.next();
			y := P.parseBinaryExpr(prec + 1);
			x = &ast.BinaryExpr{loc, tok, x, y};
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

func (P *Parser) parseSimpleStat(range_ok bool) ast.Stat {
	if P.trace {
		defer un(trace(P, "SimpleStat"));
	}

	x := P.parseExpressionList();

	switch P.tok {
	case token.COLON:
		// label declaration
		loc := P.loc;
		P.next();  // consume ":"
		P.opt_semi = true;
		if ast.ExprLen(x) == 1 {
			if label, is_ident := x.(*ast.Ident); is_ident {
				return &ast.LabelDecl{loc, label};
			}
		}
		P.error(x.Loc(), "illegal label declaration");
		return nil;

	case
		token.DEFINE, token.ASSIGN, token.ADD_ASSIGN,
		token.SUB_ASSIGN, token.MUL_ASSIGN, token.QUO_ASSIGN,
		token.REM_ASSIGN, token.AND_ASSIGN, token.OR_ASSIGN,
		token.XOR_ASSIGN, token.SHL_ASSIGN, token.SHR_ASSIGN:
		// declaration/assignment
		loc, tok := P.loc, P.tok;
		P.next();
		var y ast.Expr;
		if range_ok && P.tok == token.RANGE {
			range_loc := P.loc;
			P.next();
			y = &ast.UnaryExpr{range_loc, token.RANGE, P.parseExpression(1)};
			if tok != token.DEFINE && tok != token.ASSIGN {
				P.error(loc, "expected '=' or ':=', found '" + token.TokenString(tok) + "'");
			}
		} else {
			y = P.parseExpressionList();
			if xl, yl := ast.ExprLen(x), ast.ExprLen(y); xl > 1 && yl > 1 && xl != yl {
				P.error(x.Loc(), "arity of lhs doesn't match rhs");
			}
		}
		// TODO changed ILLEGAL -> NONE
		return &ast.ExpressionStat{x.Loc(), token.ILLEGAL, &ast.BinaryExpr{loc, tok, x, y}};

	default:
		if ast.ExprLen(x) != 1 {
			P.error(x.Loc(), "only one expression allowed");
		}

		if P.tok == token.INC || P.tok == token.DEC {
			s := &ast.ExpressionStat{P.loc, P.tok, x};
			P.next();  // consume "++" or "--"
			return s;
		}

		// TODO changed ILLEGAL -> NONE
		return &ast.ExpressionStat{x.Loc(), token.ILLEGAL, x};
	}

	unreachable();
	return nil;
}


func (P *Parser) parseInvocationStat(keyword int) *ast.ExpressionStat {
	if P.trace {
		defer un(trace(P, "InvocationStat"));
	}

	loc := P.loc;
	P.expect(keyword);
	return &ast.ExpressionStat{loc, keyword, P.parseExpression(1)};
}


func (P *Parser) parseReturnStat() *ast.ExpressionStat {
	if P.trace {
		defer un(trace(P, "ReturnStat"));
	}

	loc := P.loc;
	P.expect(token.RETURN);
	var x ast.Expr;
	if P.tok != token.SEMICOLON && P.tok != token.RBRACE {
		x = P.parseExpressionList();
	}

	return &ast.ExpressionStat{loc, token.RETURN, x};
}


func (P *Parser) parseControlFlowStat(tok int) *ast.ControlFlowStat {
	if P.trace {
		defer un(trace(P, "ControlFlowStat"));
	}

	s := &ast.ControlFlowStat{P.loc, tok, nil};
	P.expect(tok);
	if tok != token.FALLTHROUGH && P.tok == token.IDENT {
		s.Label = P.parseIdent();
	}

	return s;
}


func (P *Parser) parseControlClause(isForStat bool) (init ast.Stat, expr ast.Expr, post ast.Stat) {
	if P.trace {
		defer un(trace(P, "ControlClause"));
	}

	if P.tok != token.LBRACE {
		prev_lev := P.expr_lev;
		P.expr_lev = -1;
		if P.tok != token.SEMICOLON {
			init = P.parseSimpleStat(isForStat);
			// TODO check for range clause and exit if found
		}
		if P.tok == token.SEMICOLON {
			P.next();
			if P.tok != token.SEMICOLON && P.tok != token.LBRACE {
				expr = P.parseExpression(1);
			}
			if isForStat {
				P.expect(token.SEMICOLON);
				if P.tok != token.LBRACE {
					post = P.parseSimpleStat(false);
				}
			}
		} else {
			if init != nil {  // guard in case of errors
				if s, is_expr_stat := init.(*ast.ExpressionStat); is_expr_stat {
					expr, init = s.Expr, nil;
				} else {
					P.error(noloc, "illegal control clause");
				}
			}
		}
		P.expr_lev = prev_lev;
	}

	return init, expr, post;
}


func (P *Parser) parseIfStat() *ast.IfStat {
	if P.trace {
		defer un(trace(P, "IfStat"));
	}

	loc := P.loc;
	P.expect(token.IF);
	init, cond, dummy := P.parseControlClause(false);
	body := P.parseBlock(token.LBRACE);
	var else_ ast.Stat;
	if P.tok == token.ELSE {
		P.next();
		else_ = P.parseStatement();
	}

	return &ast.IfStat{loc, init, cond, body, else_};
}


func (P *Parser) parseForStat() *ast.ForStat {
	if P.trace {
		defer un(trace(P, "ForStat"));
	}

	loc := P.loc;
	P.expect(token.FOR);
	init, cond, post := P.parseControlClause(true);
	body := P.parseBlock(token.LBRACE);

	return &ast.ForStat{loc, init, cond, post, body};
}


func (P *Parser) parseCaseClause() *ast.CaseClause {
	if P.trace {
		defer un(trace(P, "CaseClause"));
	}

	// SwitchCase
	loc := P.loc;
	var expr ast.Expr;
	if P.tok == token.CASE {
		P.next();
		expr = P.parseExpressionList();
	} else {
		P.expect(token.DEFAULT);
	}

	return &ast.CaseClause{loc, expr, P.parseBlock(token.COLON)};
}


func (P *Parser) parseSwitchStat() *ast.SwitchStat {
	if P.trace {
		defer un(trace(P, "SwitchStat"));
	}

	loc := P.loc;
	P.expect(token.SWITCH);
	init, tag, post := P.parseControlClause(false);
	body := ast.NewBlock(P.loc, token.LBRACE);
	P.expect(token.LBRACE);
	for P.tok != token.RBRACE && P.tok != token.EOF {
		body.List.Push(P.parseCaseClause());
	}
	body.End = P.loc;
	P.expect(token.RBRACE);
	P.opt_semi = true;

	return &ast.SwitchStat{loc, init, tag, body};
}


func (P *Parser) parseCommClause() *ast.CaseClause {
	if P.trace {
		defer un(trace(P, "CommClause"));
	}

	// CommCase
	loc := P.loc;
	var expr ast.Expr;
	if P.tok == token.CASE {
		P.next();
		x := P.parseExpression(1);
		if P.tok == token.ASSIGN || P.tok == token.DEFINE {
			loc, tok := P.loc, P.tok;
			P.next();
			if P.tok == token.ARROW {
				y := P.parseExpression(1);
				x = &ast.BinaryExpr{loc, tok, x, y};
			} else {
				P.expect(token.ARROW);  // use expect() error handling
			}
		}
		expr = x;
	} else {
		P.expect(token.DEFAULT);
	}

	return &ast.CaseClause{loc, expr, P.parseBlock(token.COLON)};
}


func (P *Parser) parseSelectStat() *ast.SelectStat {
	if P.trace {
		defer un(trace(P, "SelectStat"));
	}

	loc := P.loc;
	P.expect(token.SELECT);
	body := ast.NewBlock(P.loc, token.LBRACE);
	P.expect(token.LBRACE);
	for P.tok != token.RBRACE && P.tok != token.EOF {
		body.List.Push(P.parseCommClause());
	}
	body.End = P.loc;
	P.expect(token.RBRACE);
	P.opt_semi = true;

	return &ast.SelectStat{loc, body};
}


func (P *Parser) parseStatement() ast.Stat {
	if P.trace {
		defer un(trace(P, "Statement"));
	}

	switch P.tok {
	case token.CONST, token.TYPE, token.VAR:
		return &ast.DeclarationStat{P.parseDeclaration()};
	case token.FUNC:
		// for now we do not allow local function declarations,
		// instead we assume this starts a function literal
		fallthrough;
	case
		// only the tokens that are legal top-level expression starts
		token.IDENT, token.INT, token.FLOAT, token.CHAR, token.STRING, token.LPAREN,  // operand
		token.LBRACK, token.STRUCT,  // composite type
		token.MUL, token.AND, token.ARROW:  // unary
		return P.parseSimpleStat(false);
	case token.GO, token.DEFER:
		return P.parseInvocationStat(P.tok);
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
	case token.SEMICOLON:
		// don't consume the ";", it is the separator following the empty statement
		return &ast.EmptyStat{P.loc};
	}

	// no statement found
	P.error(P.loc, "statement expected");
	return &ast.BadStat{P.loc};
}


// ----------------------------------------------------------------------------
// Declarations

func (P *Parser) parseImportSpec(loc scanner.Location) *ast.ImportDecl {
	if P.trace {
		defer un(trace(P, "ImportSpec"));
	}

	var ident *ast.Ident;
	if P.tok == token.PERIOD {
		P.error(P.loc, `"import ." not yet handled properly`);
		P.next();
	} else if P.tok == token.IDENT {
		ident = P.parseIdent();
	}

	var path ast.Expr;
	if P.tok == token.STRING {
		path = P.parseStringLit();
	} else {
		P.expect(token.STRING);  // use expect() error handling
	}

	return &ast.ImportDecl{loc, ident, path};
}


func (P *Parser) parseConstSpec(loc scanner.Location) *ast.ConstDecl {
	if P.trace {
		defer un(trace(P, "ConstSpec"));
	}

	idents := P.parseIdentList2(nil);
	typ := P.tryType();
	var vals ast.Expr;
	if P.tok == token.ASSIGN {
		P.next();
		vals = P.parseExpressionList();
	}

	return &ast.ConstDecl{loc, idents, typ, vals};
}


func (P *Parser) parseTypeSpec(loc scanner.Location) *ast.TypeDecl {
	if P.trace {
		defer un(trace(P, "TypeSpec"));
	}

	ident := P.parseIdent();
	typ := P.parseType();

	return &ast.TypeDecl{loc, ident, typ};
}


func (P *Parser) parseVarSpec(loc scanner.Location) *ast.VarDecl {
	if P.trace {
		defer un(trace(P, "VarSpec"));
	}

	idents := P.parseIdentList2(nil);
	var typ ast.Expr;
	var vals ast.Expr;
	if P.tok == token.ASSIGN {
		P.next();
		vals = P.parseExpressionList();
	} else {
		typ = P.parseVarType();
		if P.tok == token.ASSIGN {
			P.next();
			vals = P.parseExpressionList();
		}
	}

	return &ast.VarDecl{loc, idents, typ, vals};
}


func (P *Parser) parseSpec(loc scanner.Location, keyword int) ast.Decl {
	switch keyword {
	case token.IMPORT: return P.parseImportSpec(loc);
	case token.CONST: return P.parseConstSpec(loc);
	case token.TYPE: return P.parseTypeSpec(loc);
	case token.VAR: return P.parseVarSpec(loc);
	}

	unreachable();
	return nil;
}


func (P *Parser) parseDecl(keyword int) ast.Decl {
	if P.trace {
		defer un(trace(P, "Decl"));
	}

	loc := P.loc;
	P.expect(keyword);
	if P.tok == token.LPAREN {
		P.next();
		list := vector.New(0);
		for P.tok != token.RPAREN && P.tok != token.EOF {
			list.Push(P.parseSpec(noloc, keyword));
			if P.tok == token.SEMICOLON {
				P.next();
			} else {
				break;
			}
		}
		end := P.loc;
		P.expect(token.RPAREN);
		P.opt_semi = true;

		// convert vector
		decls := make([]ast.Decl, list.Len());
		for i := 0; i < list.Len(); i++ {
			decls[i] = list.At(i).(ast.Decl);
		}

		return &ast.DeclList{loc, keyword, decls, end};
	}

	return P.parseSpec(loc, keyword);
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

	loc := P.loc;
	P.expect(token.FUNC);

	var recv *ast.Field;
	if P.tok == token.LPAREN {
		loc := P.loc;
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

	return &ast.FuncDecl{loc, recv, ident, sig, body};
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

	loc := P.loc;
	P.error(loc, "declaration expected");
	P.next();  // make progress
	return &ast.BadDecl{loc};
}


// ----------------------------------------------------------------------------
// Program

// The top level parsing routines:
//
// ParsePackageClause
// - parses the package clause only and returns the package name
//
// ParseImportDecls
// - parses all import declarations and returns a list of them
// - the package clause must have been parsed before
// - useful to determine package dependencies
//
// ParseProgram
// - parses the entire program and returns the complete AST


func (P *Parser) ParsePackageClause() *ast.Ident {
	if P.trace {
		defer un(trace(P, "PackageClause"));
	}

	P.expect(token.PACKAGE);
	return P.parseIdent();
}


func (P *Parser) parseImportDecls() *vector.Vector {
	if P.trace {
		defer un(trace(P, "ImportDecls"));
	}

	list := vector.New(0);
	for P.tok == token.IMPORT {
		list.Push(P.parseDecl(token.IMPORT));
		if P.tok == token.SEMICOLON {
			P.next();
		}
	}

	return list;
}


func (P *Parser) ParseImportDecls() []ast.Decl {
	list := P.parseImportDecls();

	// convert list
	imports := make([]ast.Decl, list.Len());
	for i := 0; i < list.Len(); i++ {
		imports[i] = list.At(i).(ast.Decl);
	}

	return imports;
}


// Returns the list of comments accumulated during parsing, if any.
// (The scanner must return token.COMMENT tokens for comments to be
// collected in the first place.)

func (P *Parser) Comments() []*ast.Comment {
	// convert comments vector
	list := make([]*ast.Comment, P.comments.Len());
	for i := 0; i < P.comments.Len(); i++ {
		list[i] = P.comments.At(i).(*ast.Comment);
	}
	return list;
}


func (P *Parser) ParseProgram() *ast.Program {
	if P.trace {
		defer un(trace(P, "Program"));
	}

	p := ast.NewProgram(P.loc);
	p.Ident = P.ParsePackageClause();

	// package body
	list := P.parseImportDecls();
	for P.tok != token.EOF {
		list.Push(P.parseDeclaration());
		if P.tok == token.SEMICOLON {
			P.next();
		}
	}

	// convert list
	p.Decls = make([]ast.Decl, list.Len());
	for i := 0; i < list.Len(); i++ {
		p.Decls[i] = list.At(i).(ast.Decl);
	}

	p.Comments = P.Comments();

	return p;
}
