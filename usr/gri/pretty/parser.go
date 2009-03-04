// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Parser

import (
	"flag";
	"fmt";
	"vector";
	"token";
	"scanner";
	AST "ast";
	SymbolTable "symboltable";
)


type ErrorHandler interface {
	Error(pos int, msg string);
}


type Parser struct {
	scanner *scanner.Scanner;
	err ErrorHandler;

	// Tracing/debugging
	trace, sixg, deps bool;
	indent uint;

	comments *vector.Vector;

	// The next token
	pos int;  // token source position
	tok int;  // one token look-ahead
	val string;  // token value (for IDENT, NUMBER, STRING only)

	// Non-syntactic parser control
	opt_semi bool;  // true if semicolon separator is optional in statement list

	// Nesting levels
	expr_lev int;  // < 0: in control clause, >= 0: in expression
	scope_lev int;  // 0: global scope, 1: function scope of global functions, etc.

	// Scopes
	top_scope *SymbolTable.Scope;
};


// ----------------------------------------------------------------------------
// Elementary support

func unimplemented() {
	panic("unimplemented");
}


func unreachable() {
	panic("unreachable");
}


func assert(pred bool) {
	if !pred {
		panic("assertion failed");
	}
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
	// TODO make P.val a []byte
	var val []byte;
	P.pos, P.tok, val = P.scanner.Scan();
	P.val = string(val);
	P.opt_semi = false;

	if P.trace {
		P.printIndent();
		switch P.tok {
		case token.IDENT, token.INT, token.FLOAT, token.CHAR, token.STRING:
			fmt.Printf("[%d] %s = %s\n", P.pos, token.TokenString(P.tok), P.val);
		case token.LPAREN:
			// don't print '(' - screws up selection in terminal window
			fmt.Printf("[%d] LPAREN\n", P.pos);
		case token.RPAREN:
			// don't print ')' - screws up selection in terminal window
			fmt.Printf("[%d] RPAREN\n", P.pos);
		default:
			fmt.Printf("[%d] %s\n", P.pos, token.TokenString(P.tok));
		}
	}
}


func (P *Parser) next() {
	for P.next0(); P.tok == token.COMMENT; P.next0() {
		P.comments.Push(AST.NewComment(P.pos, P.val));
	}
}


func (P *Parser) Open(scanner *scanner.Scanner, err ErrorHandler, trace, sixg, deps bool) {
	P.scanner = scanner;
	P.err = err;

	P.trace = trace;
	P.sixg = sixg;
	P.deps = deps;
	P.indent = 0;

	P.comments = vector.New(0);

	P.next();
	P.scope_lev = 0;
	P.expr_lev = 0;
}


func (P *Parser) error(pos int, msg string) {
	P.err.Error(pos, msg);
}


func (P *Parser) expect(tok int) {
	if P.tok != tok {
		msg := "expected '" + token.TokenString(tok) + "', found '" + token.TokenString(P.tok) + "'";
		if token.IsLiteral(P.tok) {
			msg += " " + P.val;
		}
		P.error(P.pos, msg);
	}
	P.next();  // make progress in any case
}


func (P *Parser) OptSemicolon() {
	if P.tok == token.SEMICOLON {
		P.next();
	}
}


// ----------------------------------------------------------------------------
// Scopes

func (P *Parser) openScope() {
	P.top_scope = SymbolTable.NewScope(P.top_scope);
}


func (P *Parser) closeScope() {
	P.top_scope = P.top_scope.Parent;
}


/*
func (P *Parser) declareInScope(scope *SymbolTable.Scope, x AST.Expr, kind int, typ *AST.Type) {
	if P.scope_lev < 0 {
		panic("cannot declare objects in other packages");
	}
	if ident, ok := x.(*AST.Ident); ok {  // ignore bad exprs
		obj := ident.Obj;
		obj.Kind = kind;
		//TODO fix typ setup!
		//obj.Typ = typ;
		obj.Pnolev = P.scope_lev;
		switch {
		case scope.LookupLocal(obj.Ident) == nil:
			scope.Insert(obj);
		case kind == SymbolTable.TYPE:
			// possibly a forward declaration
		case kind == SymbolTable.FUNC:
			// possibly a forward declaration
		default:
			P.error(obj.Pos, `"` + obj.Ident + `" is declared already`);
		}
	}
}


// Declare a comma-separated list of idents or a single ident.
func (P *Parser) declare(x AST.Expr, kind int, typ *AST.Type) {
	for {
		p, ok := x.(*AST.BinaryExpr);
		if ok && p.Tok == token.COMMA {
			P.declareInScope(P.top_scope, p.X, kind, typ);
			x = p.Y;
		} else {
			break;
		}
	}
	P.declareInScope(P.top_scope, x, kind, typ);
}
*/


// ----------------------------------------------------------------------------
// Common productions

func (P *Parser) tryType() AST.Expr;
func (P *Parser) parseExpression(prec int) AST.Expr;
func (P *Parser) parseStatement() AST.Stat;
func (P *Parser) parseDeclaration() AST.Decl;


// If scope != nil, lookup identifier in scope. Otherwise create one.
func (P *Parser) parseIdent(scope *SymbolTable.Scope) *AST.Ident {
	if P.trace {
		defer un(trace(P, "Ident"));
	}

	if P.tok == token.IDENT {
		var obj *SymbolTable.Object;
		if scope != nil {
			obj = scope.Lookup(P.val);
		}
		if obj == nil {
			obj = SymbolTable.NewObject(P.pos, SymbolTable.NONE, P.val);
		} else {
			assert(obj.Kind != SymbolTable.NONE);
		}
		x := &AST.Ident{P.pos, obj};
		P.next();
		return x;
	}

	P.expect(token.IDENT);  // use expect() error handling
	return &AST.Ident{P.pos, nil};
}


func (P *Parser) parseIdentList(x AST.Expr) AST.Expr {
	if P.trace {
		defer un(trace(P, "IdentList"));
	}

	var last *AST.BinaryExpr;
	if x == nil {
		x = P.parseIdent(nil);
	}
	for P.tok == token.COMMA {
		pos := P.pos;
		P.next();
		y := P.parseIdent(nil);
		if last == nil {
			last = &AST.BinaryExpr{pos, token.COMMA, x, y};
			x = last;
		} else {
			last.Y = &AST.BinaryExpr{pos, token.COMMA, last.Y, y};
			last = last.Y.(*AST.BinaryExpr);
		}
	}

	return x;
}


func (P *Parser) parseIdentList2(x AST.Expr) []*AST.Ident {
	if P.trace {
		defer un(trace(P, "IdentList"));
	}

	list := vector.New(0);
	if x == nil {
		x = P.parseIdent(nil);
	}
	list.Push(x);
	for P.tok == token.COMMA {
		P.next();
		list.Push(P.parseIdent(nil));
	}

	// convert vector
	idents := make([]*AST.Ident, list.Len());
	for i := 0; i < list.Len(); i++ {
		idents[i] = list.At(i).(*AST.Ident);
	}
	return idents;
}


// ----------------------------------------------------------------------------
// Types

func (P *Parser) parseType() AST.Expr {
	if P.trace {
		defer un(trace(P, "Type"));
	}

	t := P.tryType();
	if t == nil {
		P.error(P.pos, "type expected");
		t = &AST.BadExpr{P.pos};
	}

	return t;
}


func (P *Parser) parseVarType() AST.Expr {
	if P.trace {
		defer un(trace(P, "VarType"));
	}

	return P.parseType();
}


func (P *Parser) parseQualifiedIdent() AST.Expr {
	if P.trace {
		defer un(trace(P, "QualifiedIdent"));
	}

	var x AST.Expr = P.parseIdent(P.top_scope);
	for P.tok == token.PERIOD {
		pos := P.pos;
		P.next();
		y := P.parseIdent(nil);
		x = &AST.Selector{pos, x, y};
	}

	return x;
}


func (P *Parser) parseTypeName() AST.Expr {
	if P.trace {
		defer un(trace(P, "TypeName"));
	}

	return P.parseQualifiedIdent();
}


func (P *Parser) parseArrayType() *AST.ArrayType {
	if P.trace {
		defer un(trace(P, "ArrayType"));
	}

	pos := P.pos;
	P.expect(token.LBRACK);
	var len AST.Expr;
	if P.tok == token.ELLIPSIS {
		len = &AST.Ellipsis{P.pos};
		P.next();
	} else if P.tok != token.RBRACK {
		len = P.parseExpression(1);
	}
	P.expect(token.RBRACK);
	elt := P.parseType();

	return &AST.ArrayType{pos, len, elt};
}


func (P *Parser) parseChannelType() *AST.ChannelType {
	if P.trace {
		defer un(trace(P, "ChannelType"));
	}

	pos := P.pos;
	mode := AST.FULL;
	if P.tok == token.CHAN {
		P.next();
		if P.tok == token.ARROW {
			P.next();
			mode = AST.SEND;
		}
	} else {
		P.expect(token.ARROW);
		P.expect(token.CHAN);
		mode = AST.RECV;
	}
	val := P.parseVarType();

	return &AST.ChannelType{pos, mode, val};
}


func (P *Parser) tryParameterType() AST.Expr {
	if P.tok == token.ELLIPSIS {
		pos := P.tok;
		P.next();
		return &AST.Ellipsis{pos};
	}
	return P.tryType();
}


func (P *Parser) parseParameterType() AST.Expr {
	typ := P.tryParameterType();
	if typ == nil {
		P.error(P.tok, "type expected");
		typ = &AST.BadExpr{P.pos};
	}
	return typ;
}


func (P *Parser) parseParameterDecl(ellipsis_ok bool) (*vector.Vector, AST.Expr) {
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


func (P *Parser) parseParameterList(ellipsis_ok bool) []*AST.Field {
	if P.trace {
		defer un(trace(P, "ParameterList"));
	}

	list, typ := P.parseParameterDecl(false);
	if typ != nil {
		// IdentifierList Type
		// convert list of identifiers into []*Ident
		idents := make([]*AST.Ident, list.Len());
		for i := 0; i < list.Len(); i++ {
			idents[i] = list.At(i).(*AST.Ident);
		}
		list.Init(0);
		list.Push(&AST.Field{idents, typ, nil});
		
		for P.tok == token.COMMA {
			P.next();
			idents := P.parseIdentList2(nil);
			typ := P.parseParameterType();
			list.Push(&AST.Field{idents, typ, nil});
		}

	} else {
		// Type { "," Type }
		// convert list of types into list of *Param
		for i := 0; i < list.Len(); i++ {
			list.Set(i, &AST.Field{nil, list.At(i).(AST.Expr), nil});
		}
	}

	// convert list
	params := make([]*AST.Field, list.Len());
	for i := 0; i < list.Len(); i++ {
		params[i] = list.At(i).(*AST.Field);
	}

	return params;
}


// TODO make sure Go spec is updated
func (P *Parser) parseParameters(ellipsis_ok bool) []*AST.Field {
	if P.trace {
		defer un(trace(P, "Parameters"));
	}

	var params []*AST.Field;
	P.expect(token.LPAREN);
	if P.tok != token.RPAREN {
		params = P.parseParameterList(ellipsis_ok);
	}
	P.expect(token.RPAREN);

	return params;
}


func (P *Parser) parseResult() []*AST.Field {
	if P.trace {
		defer un(trace(P, "Result"));
	}

	var result []*AST.Field;
	if P.tok == token.LPAREN {
		result = P.parseParameters(false);
	} else if P.tok != token.FUNC {
		typ := P.tryType();
		if typ != nil {
			result = make([]*AST.Field, 1);
			result[0] = &AST.Field{nil, typ, nil};
		}
	}

	return result;
}


// Function types
//
// (params)
// (params) type
// (params) (results)

func (P *Parser) parseSignature() *AST.Signature {
	if P.trace {
		defer un(trace(P, "Signature"));
	}

	//P.openScope();
	//P.scope_lev++;

	//t.Scope = P.top_scope;
	params := P.parseParameters(true);  // TODO find better solution
	//t.End = P.pos;
	result := P.parseResult();

	//P.scope_lev--;
	//P.closeScope();

	return &AST.Signature{params, result};
}


func (P *Parser) parseFunctionType() *AST.FunctionType {
	if P.trace {
		defer un(trace(P, "FunctionType"));
	}

	pos := P.pos;
	P.expect(token.FUNC);
	sig := P.parseSignature();
	
	return &AST.FunctionType{pos, sig};
}


func (P *Parser) parseMethodSpec() *AST.Field {
	if P.trace {
		defer un(trace(P, "MethodSpec"));
	}

	var idents []*AST.Ident;
	var typ AST.Expr;
	x := P.parseQualifiedIdent();
	if tmp, is_ident := x.(*AST.Ident); is_ident && (P.tok == token.COMMA || P.tok == token.LPAREN) {
		// method(s)
		idents = P.parseIdentList2(x);
		typ = &AST.FunctionType{0, P.parseSignature()};
	} else {
		// embedded interface
		typ = x;
	}
	
	return &AST.Field{idents, typ, nil};
}


func (P *Parser) parseInterfaceType() *AST.InterfaceType {
	if P.trace {
		defer un(trace(P, "InterfaceType"));
	}

	pos := P.pos;
	end := 0;
	var methods []*AST.Field;

	P.expect(token.INTERFACE);
	if P.tok == token.LBRACE {
		P.next();
		//P.openScope();
		//P.scope_lev++;

		list := vector.New(0);
		for P.tok == token.IDENT {
			list.Push(P.parseMethodSpec());
			if P.tok != token.RBRACE {
				P.expect(token.SEMICOLON);
			}
		}
		//t.End = P.pos;

		//P.scope_lev--;
		//P.closeScope();
		end = P.pos;
		P.expect(token.RBRACE);
		P.opt_semi = true;
		
		// convert vector
		methods = make([]*AST.Field, list.Len());
		for i := list.Len() - 1; i >= 0; i-- {
			methods[i] = list.At(i).(*AST.Field);
		}
	}

	return &AST.InterfaceType{pos, methods, end};
}


func (P *Parser) parseMapType() *AST.MapType {
	if P.trace {
		defer un(trace(P, "MapType"));
	}

	pos := P.pos;
	P.expect(token.MAP);
	P.expect(token.LBRACK);
	key := P.parseVarType();
	P.expect(token.RBRACK);
	val := P.parseVarType();

	return &AST.MapType{pos, key, val};
}


func (P *Parser) parseOperand() AST.Expr


func (P *Parser) parseFieldDecl() *AST.Field {
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
	var tag AST.Expr;
	if P.tok == token.STRING {
		// ParseOperand takes care of string concatenation
		tag = P.parseOperand();
	}

	// analyze case
	var idents []*AST.Ident;
	if typ != nil {
		// non-empty identifier list followed by a type
		idents = make([]*AST.Ident, list.Len());
		for i := 0; i < list.Len(); i++ {
			if ident, is_ident := list.At(i).(*AST.Ident); is_ident {
				idents[i] = ident;
			} else {
				P.error(list.At(i).(AST.Expr).Pos(), "identifier expected");
			}
		}
	} else {
		// anonymous field
		if list.Len() == 1 {
			// TODO should do more checks here
			typ = list.At(0).(AST.Expr);
		} else {
			P.error(P.pos, "anonymous field expected");
		}
	}
	
	return &AST.Field{idents, typ, tag};
}


func (P *Parser) parseStructType() AST.Expr {
	if P.trace {
		defer un(trace(P, "StructType"));
	}

	pos := P.pos;
	end := 0;
	var fields []*AST.Field;
	
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
		P.OptSemicolon();

		end = P.pos;
		P.expect(token.RBRACE);
		P.opt_semi = true;

		// convert vector
		fields = make([]*AST.Field, list.Len());
		for i := list.Len() - 1; i >= 0; i-- {
			fields[i] = list.At(i).(*AST.Field);
		}
	}

	return AST.StructType{pos, fields, end};
}


func (P *Parser) parsePointerType() AST.Expr {
	if P.trace {
		defer un(trace(P, "PointerType"));
	}

	pos := P.pos;
	P.expect(token.MUL);
	base := P.parseType();

	return &AST.PointerType{pos, base};
}


func (P *Parser) tryType() AST.Expr {
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
		pos := P.pos;
		P.next();
		t := P.parseType();
		P.expect(token.RPAREN);
		return &AST.Group{pos, t};
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


func (P *Parser) parseBlock(tok int) *AST.Block {
	if P.trace {
		defer un(trace(P, "Block"));
	}

	b := AST.NewBlock(P.pos, tok);
	P.expect(tok);

	/*
	P.openScope();
	// enter recv and parameters into function scope
	if ftyp != nil {
		assert(ftyp.Form == AST.FUNCTION);
		if ftyp.Key != nil {
		}
		if ftyp.List != nil {
			for i, n := 0, ftyp.List.Len(); i < n; i++ {
				if x, ok := ftyp.List.At(i).(*AST.Ident); ok {
					P.declareInScope(P.top_scope, x, SymbolTable.VAR, nil);
				}
			}
		}
	}
	*/
	
	P.parseStatementList(b.List);
	
	/*
	P.closeScope();
	*/

	if tok == token.LBRACE {
		b.End = P.pos;
		P.expect(token.RBRACE);
		P.opt_semi = true;
	}

	return b;
}


// ----------------------------------------------------------------------------
// Expressions

func (P *Parser) parseExpressionList() AST.Expr {
	if P.trace {
		defer un(trace(P, "ExpressionList"));
	}

	x := P.parseExpression(1);
	for first := true; P.tok == token.COMMA; {
		pos := P.pos;
		P.next();
		y := P.parseExpression(1);
		if first {
			x = &AST.BinaryExpr{pos, token.COMMA, x, y};
			first = false;
		} else {
			x.(*AST.BinaryExpr).Y = &AST.BinaryExpr{pos, token.COMMA, x.(*AST.BinaryExpr).Y, y};
		}
	}

	return x;
}


func (P *Parser) parseFunctionLit() AST.Expr {
	if P.trace {
		defer un(trace(P, "FunctionLit"));
	}

	pos := P.pos;
	P.expect(token.FUNC);
	typ := P.parseSignature();
	P.expr_lev++;
	P.scope_lev++;
	body := P.parseBlock(token.LBRACE);
	P.scope_lev--;
	P.expr_lev--;

	return &AST.FunctionLit{pos, typ, body};
}


func (P *Parser) parseOperand() AST.Expr {
	if P.trace {
		defer un(trace(P, "Operand"));
	}

	switch P.tok {
	case token.IDENT:
		return P.parseIdent(P.top_scope);

	case token.INT, token.FLOAT, token.CHAR, token.STRING:
		x := &AST.BasicLit{P.pos, P.tok, P.val};
		P.next();
		if x.Tok == token.STRING {
			// TODO should remember the list instead of
			//      concatenate the strings here
			for ; P.tok == token.STRING; P.next() {
				x.Val += P.val;
			}
		}
		return x;

	case token.LPAREN:
		pos := P.pos;
		P.next();
		P.expr_lev++;
		x := P.parseExpression(1);
		P.expr_lev--;
		P.expect(token.RPAREN);
		return &AST.Group{pos, x};

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

	return &AST.BadExpr{P.pos};
}


func (P *Parser) parseSelectorOrTypeGuard(x AST.Expr) AST.Expr {
	if P.trace {
		defer un(trace(P, "SelectorOrTypeGuard"));
	}

	pos := P.pos;
	P.expect(token.PERIOD);

	if P.tok == token.IDENT {
		x = &AST.Selector{pos, x, P.parseIdent(nil)};

	} else {
		P.expect(token.LPAREN);
		x = &AST.TypeGuard{pos, x, P.parseType()};
		P.expect(token.RPAREN);
	}

	return x;
}


func (P *Parser) parseIndex(x AST.Expr) AST.Expr {
	if P.trace {
		defer un(trace(P, "IndexOrSlice"));
	}

	pos := P.pos;
	P.expect(token.LBRACK);
	P.expr_lev++;
	i := P.parseExpression(0);
	P.expr_lev--;
	P.expect(token.RBRACK);

	return &AST.Index{pos, x, i};
}


func (P *Parser) parseBinaryExpr(prec1 int) AST.Expr

func (P *Parser) parseCompositeElements(close int) AST.Expr {
	x := P.parseExpression(0);
	if P.tok == token.COMMA {
		pos := P.pos;
		P.next();

		// first element determines mode
		singles := true;
		if t, is_binary := x.(*AST.BinaryExpr); is_binary && t.Tok == token.COLON {
			singles = false;
		}

		var last *AST.BinaryExpr;
		for P.tok != close && P.tok != token.EOF {
			y := P.parseExpression(0);

			if singles {
				if t, is_binary := y.(*AST.BinaryExpr); is_binary && t.Tok == token.COLON {
					P.error(t.X.Pos(), "single value expected; found pair");
				}
			} else {
				if t, is_binary := y.(*AST.BinaryExpr); !is_binary || t.Tok != token.COLON {
					P.error(y.Pos(), "key:value pair expected; found single value");
				}
			}

			if last == nil {
				last = &AST.BinaryExpr{pos, token.COMMA, x, y};
				x = last;
			} else {
				last.Y = &AST.BinaryExpr{pos, token.COMMA, last.Y, y};
				last = last.Y.(*AST.BinaryExpr);
			}

			if P.tok == token.COMMA {
				pos = P.pos;
				P.next();
			} else {
				break;
			}

		}
	}
	return x;
}


func (P *Parser) parseCallOrCompositeLit(f AST.Expr, open, close int) AST.Expr {
	if P.trace {
		defer un(trace(P, "CallOrCompositeLit"));
	}

	pos := P.pos;
	P.expect(open);
	var args AST.Expr;
	if P.tok != close {
		args = P.parseCompositeElements(close);
	}
	P.expect(close);

	return &AST.Call{pos, open, f, args};
}


func (P *Parser) parsePrimaryExpr() AST.Expr {
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


func (P *Parser) parseUnaryExpr() AST.Expr {
	if P.trace {
		defer un(trace(P, "UnaryExpr"));
	}

	switch P.tok {
	case token.ADD, token.SUB, token.MUL, token.NOT, token.XOR, token.ARROW, token.AND:
		pos, tok := P.pos, P.tok;
		P.next();
		y := P.parseUnaryExpr();
		return &AST.UnaryExpr{pos, tok, y};
		/*
		if lit, ok := y.(*AST.TypeLit); ok && tok == token.MUL {
			// pointer type
			t := AST.NewType(pos, AST.POINTER);
			t.Elt = lit.Typ;
			return &AST.TypeLit{t};
		} else {
			return &AST.UnaryExpr{pos, tok, y};
		}
		*/
	}

	return P.parsePrimaryExpr();
}


func (P *Parser) parseBinaryExpr(prec1 int) AST.Expr {
	if P.trace {
		defer un(trace(P, "BinaryExpr"));
	}

	x := P.parseUnaryExpr();
	for prec := token.Precedence(P.tok); prec >= prec1; prec-- {
		for token.Precedence(P.tok) == prec {
			pos, tok := P.pos, P.tok;
			P.next();
			y := P.parseBinaryExpr(prec + 1);
			x = &AST.BinaryExpr{pos, tok, x, y};
		}
	}

	return x;
}


func (P *Parser) parseExpression(prec int) AST.Expr {
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

func (P *Parser) parseSimpleStat(range_ok bool) AST.Stat {
	if P.trace {
		defer un(trace(P, "SimpleStat"));
	}

	x := P.parseExpressionList();

	switch P.tok {
	case token.COLON:
		// label declaration
		pos := P.pos;
		P.next();  // consume ":"
		P.opt_semi = true;
		if AST.ExprLen(x) == 1 {
			if label, is_ident := x.(*AST.Ident); is_ident {
				return &AST.LabelDecl{pos, label};
			}
		}
		P.error(x.Pos(), "illegal label declaration");
		return nil;

	case
		token.DEFINE, token.ASSIGN, token.ADD_ASSIGN,
		token.SUB_ASSIGN, token.MUL_ASSIGN, token.QUO_ASSIGN,
		token.REM_ASSIGN, token.AND_ASSIGN, token.OR_ASSIGN,
		token.XOR_ASSIGN, token.SHL_ASSIGN, token.SHR_ASSIGN:
		// declaration/assignment
		pos, tok := P.pos, P.tok;
		P.next();
		var y AST.Expr;
		if range_ok && P.tok == token.RANGE {
			range_pos := P.pos;
			P.next();
			y = &AST.UnaryExpr{range_pos, token.RANGE, P.parseExpression(1)};
			if tok != token.DEFINE && tok != token.ASSIGN {
				P.error(pos, "expected '=' or ':=', found '" + token.TokenString(tok) + "'");
			}
		} else {
			y = P.parseExpressionList();
			if xl, yl := AST.ExprLen(x), AST.ExprLen(y); xl > 1 && yl > 1 && xl != yl {
				P.error(x.Pos(), "arity of lhs doesn't match rhs");
			}
		}
		// TODO changed ILLEGAL -> NONE
		return &AST.ExpressionStat{x.Pos(), token.ILLEGAL, &AST.BinaryExpr{pos, tok, x, y}};

	default:
		if AST.ExprLen(x) != 1 {
			P.error(x.Pos(), "only one expression allowed");
		}

		if P.tok == token.INC || P.tok == token.DEC {
			s := &AST.ExpressionStat{P.pos, P.tok, x};
			P.next();  // consume "++" or "--"
			return s;
		}

		// TODO changed ILLEGAL -> NONE
		return &AST.ExpressionStat{x.Pos(), token.ILLEGAL, x};
	}

	unreachable();
	return nil;
}


func (P *Parser) parseInvocationStat(keyword int) *AST.ExpressionStat {
	if P.trace {
		defer un(trace(P, "InvocationStat"));
	}

	pos := P.pos;
	P.expect(keyword);
	return &AST.ExpressionStat{pos, keyword, P.parseExpression(1)};
}


func (P *Parser) parseReturnStat() *AST.ExpressionStat {
	if P.trace {
		defer un(trace(P, "ReturnStat"));
	}

	pos := P.pos;
	P.expect(token.RETURN);
	var x AST.Expr;
	if P.tok != token.SEMICOLON && P.tok != token.RBRACE {
		x = P.parseExpressionList();
	}

	return &AST.ExpressionStat{pos, token.RETURN, x};
}


func (P *Parser) parseControlFlowStat(tok int) *AST.ControlFlowStat {
	if P.trace {
		defer un(trace(P, "ControlFlowStat"));
	}

	s := &AST.ControlFlowStat{P.pos, tok, nil};
	P.expect(tok);
	if tok != token.FALLTHROUGH && P.tok == token.IDENT {
		s.Label = P.parseIdent(P.top_scope);
	}

	return s;
}


func (P *Parser) parseControlClause(isForStat bool) (init AST.Stat, expr AST.Expr, post AST.Stat) {
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
				if s, is_expr_stat := init.(*AST.ExpressionStat); is_expr_stat {
					expr, init = s.Expr, nil;
				} else {
					P.error(0, "illegal control clause");
				}
			}
		}
		P.expr_lev = prev_lev;
	}

	return init, expr, post;
}


func (P *Parser) parseIfStat() *AST.IfStat {
	if P.trace {
		defer un(trace(P, "IfStat"));
	}

	P.openScope();
	pos := P.pos;
	P.expect(token.IF);
	init, cond, dummy := P.parseControlClause(false);
	body := P.parseBlock(token.LBRACE);
	var else_ AST.Stat;
	if P.tok == token.ELSE {
		P.next();
		if ok := P.tok == token.IF || P.tok == token.LBRACE; ok || P.sixg {
			else_ = P.parseStatement();
			if !ok {
				// wrap in a block since we don't have one
				body := AST.NewBlock(0, token.LBRACE);
				body.List.Push(else_);
				else_ = &AST.CompositeStat{body};
			}
		} else {
			P.error(P.pos, "'if' or '{' expected - illegal 'else' branch");
		}
	}
	P.closeScope();

	return &AST.IfStat{pos, init, cond, body, else_};
}


func (P *Parser) parseForStat() *AST.ForStat {
	if P.trace {
		defer un(trace(P, "ForStat"));
	}

	P.openScope();
	pos := P.pos;
	P.expect(token.FOR);
	init, cond, post := P.parseControlClause(true);
	body := P.parseBlock(token.LBRACE);
	P.closeScope();

	return &AST.ForStat{pos, init, cond, post, body};
}


func (P *Parser) parseCaseClause() *AST.CaseClause {
	if P.trace {
		defer un(trace(P, "CaseClause"));
	}

	// SwitchCase
	pos := P.pos;
	var expr AST.Expr;
	if P.tok == token.CASE {
		P.next();
		expr = P.parseExpressionList();
	} else {
		P.expect(token.DEFAULT);
	}

	return &AST.CaseClause{pos, expr, P.parseBlock(token.COLON)};
}


func (P *Parser) parseSwitchStat() *AST.SwitchStat {
	if P.trace {
		defer un(trace(P, "SwitchStat"));
	}

	P.openScope();
	pos := P.pos;
	P.expect(token.SWITCH);
	init, tag, post := P.parseControlClause(false);
	body := AST.NewBlock(P.pos, token.LBRACE);
	P.expect(token.LBRACE);
	for P.tok != token.RBRACE && P.tok != token.EOF {
		body.List.Push(P.parseCaseClause());
	}
	body.End = P.pos;
	P.expect(token.RBRACE);
	P.opt_semi = true;
	P.closeScope();

	return &AST.SwitchStat{pos, init, tag, body};
}


func (P *Parser) parseCommClause() *AST.CaseClause {
	if P.trace {
		defer un(trace(P, "CommClause"));
	}

	// CommCase
	pos := P.pos;
	var expr AST.Expr;
	if P.tok == token.CASE {
		P.next();
		x := P.parseExpression(1);
		if P.tok == token.ASSIGN || P.tok == token.DEFINE {
			pos, tok := P.pos, P.tok;
			P.next();
			if P.tok == token.ARROW {
				y := P.parseExpression(1);
				x = &AST.BinaryExpr{pos, tok, x, y};
			} else {
				P.expect(token.ARROW);  // use expect() error handling
			}
		}
		expr = x;
	} else {
		P.expect(token.DEFAULT);
	}

	return &AST.CaseClause{pos, expr, P.parseBlock(token.COLON)};
}


func (P *Parser) parseSelectStat() *AST.SelectStat {
	if P.trace {
		defer un(trace(P, "SelectStat"));
	}

	P.openScope();
	pos := P.pos;
	P.expect(token.SELECT);
	body := AST.NewBlock(P.pos, token.LBRACE);
	P.expect(token.LBRACE);
	for P.tok != token.RBRACE && P.tok != token.EOF {
		body.List.Push(P.parseCommClause());
	}
	body.End = P.pos;
	P.expect(token.RBRACE);
	P.opt_semi = true;
	P.closeScope();

	return &AST.SelectStat{pos, body};
}


func (P *Parser) parseStatement() AST.Stat {
	if P.trace {
		defer un(trace(P, "Statement"));
	}

	switch P.tok {
	case token.CONST, token.TYPE, token.VAR:
		return &AST.DeclarationStat{P.parseDeclaration()};
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
		return &AST.CompositeStat{P.parseBlock(token.LBRACE)};
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
		return &AST.EmptyStat{P.pos};
	}

	// no statement found
	P.error(P.pos, "statement expected");
	return &AST.BadStat{P.pos};
}


// ----------------------------------------------------------------------------
// Declarations

func (P *Parser) parseImportSpec(pos int) *AST.ImportDecl {
	if P.trace {
		defer un(trace(P, "ImportSpec"));
	}

	var ident *AST.Ident;
	if P.tok == token.PERIOD {
		P.error(P.pos, `"import ." not yet handled properly`);
		P.next();
	} else if P.tok == token.IDENT {
		ident = P.parseIdent(nil);
	}

	var path AST.Expr;
	if P.tok == token.STRING {
		// TODO eventually the scanner should strip the quotes
		path = &AST.BasicLit{P.pos, token.STRING, P.val};
		P.next();
	} else {
		P.expect(token.STRING);  // use expect() error handling
	}
	
	return &AST.ImportDecl{pos, ident, path};
}


func (P *Parser) parseConstSpec(pos int) *AST.ConstDecl {
	if P.trace {
		defer un(trace(P, "ConstSpec"));
	}

	idents := P.parseIdentList2(nil);
	typ := P.tryType();
	var vals AST.Expr;
	if P.tok == token.ASSIGN {
		P.next();
		vals = P.parseExpressionList();
	}
	
	return &AST.ConstDecl{pos, idents, typ, vals};
}


func (P *Parser) parseTypeSpec(pos int) *AST.TypeDecl {
	if P.trace {
		defer un(trace(P, "TypeSpec"));
	}

	ident := P.parseIdent(nil);
	typ := P.parseType();
	
	return &AST.TypeDecl{pos, ident, typ};
}


func (P *Parser) parseVarSpec(pos int) *AST.VarDecl {
	if P.trace {
		defer un(trace(P, "VarSpec"));
	}

	idents := P.parseIdentList2(nil);
	var typ AST.Expr;
	var vals AST.Expr;
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
	
	return &AST.VarDecl{pos, idents, typ, vals};
}


func (P *Parser) parseSpec(pos, keyword int) AST.Decl {
	kind := SymbolTable.NONE;

	switch keyword {
	case token.IMPORT: return P.parseImportSpec(pos);
	case token.CONST: return P.parseConstSpec(pos);
	case token.TYPE: return P.parseTypeSpec(pos);
	case token.VAR: return P.parseVarSpec(pos);
	}
	
	unreachable();
	return nil;

	/*
	// semantic checks
	if d.Tok == token.IMPORT {
		if d.Ident != nil {
			//P.declare(d.Ident, kind, nil);
		}
	} else {
		//P.declare(d.Ident, kind, d.Typ);
		if d.Val != nil {
			// initialization/assignment
			llen := AST.ExprLen(d.Ident);
			rlen := AST.ExprLen(d.Val);
			if llen == rlen {
				// TODO
			} else if rlen == 1 {
				// TODO
			} else {
				if llen < rlen {
					P.error(AST.ExprAt(d.Val, llen).Pos(), "more expressions than variables");
				} else {
					P.error(AST.ExprAt(d.Ident, rlen).Pos(), "more variables than expressions");
				}
			}
		} else {
			// TODO
		}
	}
	*/
}


func (P *Parser) parseDecl(keyword int) AST.Decl {
	if P.trace {
		defer un(trace(P, "Decl"));
	}

	pos := P.pos;
	P.expect(keyword);
	if P.tok == token.LPAREN {
		P.next();
		list := vector.New(0);
		for P.tok != token.RPAREN && P.tok != token.EOF {
			list.Push(P.parseSpec(0, keyword));
			if P.tok == token.SEMICOLON {
				P.next();
			} else {
				break;
			}
		}
		end := P.pos;
		P.expect(token.RPAREN);
		P.opt_semi = true;
		
		// convert vector
		decls := make([]AST.Decl, list.Len());
		for i := 0; i < list.Len(); i++ {
			decls[i] = list.At(i).(AST.Decl);
		}
		
		return &AST.DeclList{pos, keyword, decls, end};
	}

	return P.parseSpec(pos, keyword);
}


// Function declarations
//
// func        ident (params)
// func        ident (params) type
// func        ident (params) (results)
// func (recv) ident (params)
// func (recv) ident (params) type
// func (recv) ident (params) (results)

func (P *Parser) parseFunctionDecl() *AST.FuncDecl {
	if P.trace {
		defer un(trace(P, "FunctionDecl"));
	}

	pos := P.pos;
	P.expect(token.FUNC);

	var recv *AST.Field;
	if P.tok == token.LPAREN {
		pos := P.pos;
		tmp := P.parseParameters(true);
		if len(tmp) == 1 {
			recv = tmp[0];
		} else {
			P.error(pos, "must have exactly one receiver");
		}
	}

	ident := P.parseIdent(nil);
	sig := P.parseSignature();

	var body *AST.Block;
	if P.tok == token.LBRACE {
		body = P.parseBlock(token.LBRACE);
	}

	return &AST.FuncDecl{pos, recv, ident, sig, body};
}


func (P *Parser) parseDeclaration() AST.Decl {
	if P.trace {
		defer un(trace(P, "Declaration"));
	}

	switch P.tok {
	case token.CONST, token.TYPE, token.VAR:
		return P.parseDecl(P.tok);
	case token.FUNC:
		return P.parseFunctionDecl();
	}
	
	pos := P.pos;
	P.error(pos, "declaration expected");
	P.next();  // make progress
	return &AST.BadDecl{pos};
}


// ----------------------------------------------------------------------------
// Program

func (P *Parser) ParseProgram() *AST.Program {
	if P.trace {
		defer un(trace(P, "Program"));
	}

	P.openScope();
	p := AST.NewProgram(P.pos);
	P.expect(token.PACKAGE);
	p.Ident = P.parseIdent(nil);

	// package body
	{	P.openScope();
		list := vector.New(0);
		for P.tok == token.IMPORT {
			list.Push(P.parseDecl(token.IMPORT));
			P.OptSemicolon();
		}
		if !P.deps {
			for P.tok != token.EOF {
				list.Push(P.parseDeclaration());
				P.OptSemicolon();
			}
		}
		P.closeScope();

		// convert list
		p.Decls = make([]AST.Decl, list.Len());
		for i := 0; i < list.Len(); i++ {
			p.Decls[i] = list.At(i).(AST.Decl);
		}
	}

	p.Comments = P.comments;
	P.closeScope();

	return p;
}
