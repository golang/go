// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Parser

import (
	"flag";
	"fmt";
	"vector";
	Scanner "scanner";
	AST "ast";
	SymbolTable "symboltable";
)


type Parser struct {
	// Tracing/debugging
	trace, sixg, deps bool;
	indent uint;

	// Scanner
	scanner *Scanner.Scanner;
	comments *vector.Vector;

	// Scanner.Token
	pos int;  // token source position
	tok int;  // one token look-ahead
	val string;  // token value (for IDENT, NUMBER, STRING only)

	// Non-syntactic parser control
	opt_semi bool;  // true if semicolon is optional

	// Nesting levels
	scope_lev int;  // 0 = global scope, 1 = function scope of global functions, etc.

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
	P.pos, P.tok, P.val = P.scanner.Scan();
	P.opt_semi = false;

	if P.trace {
		P.printIndent();
		switch P.tok {
		case Scanner.IDENT, Scanner.INT, Scanner.FLOAT, Scanner.STRING:
			fmt.Printf("[%d] %s = %s\n", P.pos, Scanner.TokenString(P.tok), P.val);
		case Scanner.LPAREN:
			// don't print '(' - screws up selection in terminal window
			fmt.Printf("[%d] LPAREN\n", P.pos);
		case Scanner.RPAREN:
			// don't print ')' - screws up selection in terminal window
			fmt.Printf("[%d] RPAREN\n", P.pos);
		default:
			fmt.Printf("[%d] %s\n", P.pos, Scanner.TokenString(P.tok));
		}
	}
}


func (P *Parser) next() {
	for P.next0(); P.tok == Scanner.COMMENT; P.next0() {
		P.comments.Push(AST.NewComment(P.pos, P.val));
	}
}


func (P *Parser) Open(trace, sixg, deps bool, scanner *Scanner.Scanner) {
	P.trace = trace;
	P.sixg = sixg;
	P.deps = deps;
	P.indent = 0;

	P.scanner = scanner;
	P.comments = vector.New(0);

	P.next();
	P.scope_lev = 0;
}


func (P *Parser) error(pos int, msg string) {
	P.scanner.Error(pos, msg);
}


func (P *Parser) expect(tok int) {
	if P.tok != tok {
		msg := "expected '" + Scanner.TokenString(tok) + "', found '" + Scanner.TokenString(P.tok) + "'";
		switch P.tok {
		case Scanner.IDENT, Scanner.INT, Scanner.FLOAT, Scanner.STRING:
			msg += " " + P.val;
		}
		P.error(P.pos, msg);
	}
	P.next();  // make progress in any case
}


func (P *Parser) OptSemicolon() {
	if P.tok == Scanner.SEMICOLON {
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


// declare a comma-separated list of idents or a single ident.
func (P *Parser) declare(x AST.Expr, kind int, typ *AST.Type) {
	for {
		p, ok := x.(*AST.BinaryExpr);
		if ok && p.Tok == Scanner.COMMA {
			P.declareInScope(P.top_scope, p.X, kind, typ);
			x = p.Y;
		} else {
			break;
		}
	}
	P.declareInScope(P.top_scope, x, kind, typ);
}


// ----------------------------------------------------------------------------
// Common productions

func (P *Parser) tryType() *AST.Type;
func (P *Parser) parseExpression(prec int) AST.Expr;
func (P *Parser) parseStatement() AST.Stat;
func (P *Parser) parseDeclaration() *AST.Decl;


// If scope != nil, lookup identifier in scope. Otherwise create one.
func (P *Parser) parseIdent(scope *SymbolTable.Scope) *AST.Ident {
	if P.trace {
		defer un(trace(P, "Ident"));
	}

	if P.tok == Scanner.IDENT {
		var obj *SymbolTable.Object;
		if scope != nil {
			obj = scope.Lookup(P.val);
		}
		if obj == nil {
			obj = SymbolTable.NewObject(P.pos, SymbolTable.NONE, P.val);
		} else {
			assert(obj.Kind != SymbolTable.NONE);
		}
		x := &AST.Ident(P.pos, obj);
		P.next();
		return x;
	}

	P.expect(Scanner.IDENT);  // use expect() error handling
	return &AST.Ident(P.pos, nil);
}


func (P *Parser) parseIdentList() AST.Expr {
	if P.trace {
		defer un(trace(P, "IdentList"));
	}

	var last *AST.BinaryExpr;
	var x AST.Expr = P.parseIdent(nil);
	for P.tok == Scanner.COMMA {
		pos := P.pos;
		P.next();
		y := P.parseIdent(nil);
		if last == nil {
			last = &AST.BinaryExpr(pos, Scanner.COMMA, x, y);
			x = last;
		} else {
			last.Y = &AST.BinaryExpr(pos, Scanner.COMMA, last.Y, y);
			last = last.Y.(*AST.BinaryExpr);
		}
	}

	return x;
}


// ----------------------------------------------------------------------------
// Types

func (P *Parser) parseType() *AST.Type {
	if P.trace {
		defer un(trace(P, "Type"));
	}

	t := P.tryType();
	if t == nil {
		P.error(P.pos, "type expected");
		t = AST.BadType;
	}

	return t;
}


func (P *Parser) parseVarType() *AST.Type {
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
	for P.tok == Scanner.PERIOD {
		pos := P.pos;
		P.next();
		y := P.parseIdent(nil);
		x = &AST.Selector(pos, x, y);
	}

	return x;
}


func (P *Parser) parseTypeName() *AST.Type {
	if P.trace {
		defer un(trace(P, "TypeName"));
	}

	t := AST.NewType(P.pos, AST.TYPENAME);
	t.Expr = P.parseQualifiedIdent();

	return t;
}


func (P *Parser) parseArrayType() *AST.Type {
	if P.trace {
		defer un(trace(P, "ArrayType"));
	}

	t := AST.NewType(P.pos, AST.ARRAY);
	P.expect(Scanner.LBRACK);
	if P.tok == Scanner.ELLIPSIS {
		t.Expr = &AST.BinaryExpr(P.pos, Scanner.ELLIPSIS, nil, nil);
		P.next();
	} else if P.tok != Scanner.RBRACK {
		t.Expr = P.parseExpression(1);
	}
	P.expect(Scanner.RBRACK);
	t.Elt = P.parseType();

	return t;
}


func (P *Parser) parseChannelType() *AST.Type {
	if P.trace {
		defer un(trace(P, "ChannelType"));
	}

	t := AST.NewType(P.pos, AST.CHANNEL);
	t.Mode = AST.FULL;
	if P.tok == Scanner.CHAN {
		P.next();
		if P.tok == Scanner.ARROW {
			P.next();
			t.Mode = AST.SEND;
		}
	} else {
		P.expect(Scanner.ARROW);
		P.expect(Scanner.CHAN);
		t.Mode = AST.RECV;
	}
	t.Elt = P.parseVarType();

	return t;
}


func (P *Parser) parseVar(expect_ident bool) *AST.Type {
	t := AST.BadType;
	if expect_ident {
		x := P.parseIdent(nil);
		t = AST.NewType(x.Pos(), AST.TYPENAME);
		t.Expr = x;
	} else if P.tok == Scanner.ELLIPSIS {
		t = AST.NewType(P.pos, AST.ELLIPSIS);
		P.next();
	} else {
		t = P.parseType();
	}
	return t;
}


func (P *Parser) parseVarList(list *vector.Vector, ellipsis_ok bool) {
	if P.trace {
		defer un(trace(P, "VarList"));
	}

	// assume a list of types
	// (a list of identifiers looks like a list of type names)
	i0 := list.Len();
	for {
		list.Push(P.parseVar(ellipsis_ok /* param list */ && i0 > 0));
		if P.tok == Scanner.COMMA {
			P.next();
		} else {
			break;
		}
	}

	// if we had a list of identifiers, it must be followed by a type
	typ := P.tryType();
	if typ == nil && P.tok == Scanner.ELLIPSIS {
		typ = AST.NewType(P.pos, AST.ELLIPSIS);
		P.next();
	}

	if ellipsis_ok /* param list */ && i0 > 0 && typ == nil {
		// not the first parameter section; we must have a type
		P.error(P.pos, "type expected");
		typ = AST.BadType;
	}

	// convert the list into a list of (type) expressions
	if typ != nil {
		// all list entries must be identifiers
		// convert the type entries into identifiers
		for i, n := i0, list.Len(); i < n; i++ {
			t := list.At(i).(*AST.Type);
			if t.Form == AST.TYPENAME {
				if ident, ok := t.Expr.(*AST.Ident); ok {
					list.Set(i, ident);
					continue;
				}
			}
			list.Set(i, &AST.BadExpr(0));
			P.error(t.Pos, "identifier expected");
		}
		// add type
		list.Push(&AST.TypeLit(typ));

	} else {
		// all list entries are types
		// convert all type entries into type expressions
		for i, n := i0, list.Len(); i < n; i++ {
			t := list.At(i).(*AST.Type);
			list.Set(i, &AST.TypeLit(t));
		}
	}
}


func (P *Parser) parseParameterList(ellipsis_ok bool) *vector.Vector {
	if P.trace {
		defer un(trace(P, "ParameterList"));
	}

	list := vector.New(0);
	P.parseVarList(list, ellipsis_ok);
	for P.tok == Scanner.COMMA {
		P.next();
		P.parseVarList(list, ellipsis_ok);
	}

	return list;
}


func (P *Parser) parseParameters(ellipsis_ok bool) *AST.Type {
	if P.trace {
		defer un(trace(P, "Parameters"));
	}

	t := AST.NewType(P.pos, AST.STRUCT);
	P.expect(Scanner.LPAREN);
	if P.tok != Scanner.RPAREN {
		t.List = P.parseParameterList(ellipsis_ok);
	}
	t.End = P.pos;
	P.expect(Scanner.RPAREN);

	return t;
}


func (P *Parser) parseResultList() {
	if P.trace {
		defer un(trace(P, "ResultList"));
	}

	P.parseType();
	for P.tok == Scanner.COMMA {
		P.next();
		P.parseType();
	}
	if P.tok != Scanner.RPAREN {
		P.parseType();
	}
}


func (P *Parser) parseResult(ftyp *AST.Type) *AST.Type {
	if P.trace {
		defer un(trace(P, "Result"));
	}

	var t *AST.Type;
	if P.tok == Scanner.LPAREN {
		t = P.parseParameters(false);
	} else if P.tok != Scanner.FUNC {
		typ := P.tryType();
		if typ != nil {
			t = AST.NewType(P.pos, AST.STRUCT);
			t.List = vector.New(0);
			t.List.Push(&AST.TypeLit(typ));
			t.End = P.pos;
		}
	}

	return t;
}


// Function types
//
// (params)
// (params) type
// (params) (results)

func (P *Parser) parseSignature() *AST.Type {
	if P.trace {
		defer un(trace(P, "Signature"));
	}

	P.openScope();
	P.scope_lev++;

	t := AST.NewType(P.pos, AST.FUNCTION);
	t.Scope = P.top_scope;
	t.List = P.parseParameters(true).List;  // TODO find better solution
	t.End = P.pos;
	t.Elt = P.parseResult(t);

	P.scope_lev--;
	P.closeScope();

	return t;
}


func (P *Parser) parseFunctionType() *AST.Type {
	if P.trace {
		defer un(trace(P, "FunctionType"));
	}

	P.expect(Scanner.FUNC);
	return P.parseSignature();
}


func (P *Parser) parseMethodSpec(list *vector.Vector) {
	if P.trace {
		defer un(trace(P, "MethodDecl"));
	}

	list.Push(P.parseIdentList());
	t := P.parseSignature();
	list.Push(&AST.TypeLit(t));
}


func (P *Parser) parseInterfaceType() *AST.Type {
	if P.trace {
		defer un(trace(P, "InterfaceType"));
	}

	t := AST.NewType(P.pos, AST.INTERFACE);
	P.expect(Scanner.INTERFACE);
	if P.tok == Scanner.LBRACE {
		P.next();
		P.openScope();
		P.scope_lev++;

		t.List = vector.New(0);
		for P.tok == Scanner.IDENT {
			P.parseMethodSpec(t.List);
			if P.tok != Scanner.RBRACE {
				P.expect(Scanner.SEMICOLON);
			}
		}
		t.End = P.pos;

		P.scope_lev--;
		P.closeScope();
		P.expect(Scanner.RBRACE);
	}

	return t;
}


func (P *Parser) parseMapType() *AST.Type {
	if P.trace {
		defer un(trace(P, "MapType"));
	}

	t := AST.NewType(P.pos, AST.MAP);
	P.expect(Scanner.MAP);
	P.expect(Scanner.LBRACK);
	t.Key = P.parseVarType();
	P.expect(Scanner.RBRACK);
	t.Elt = P.parseVarType();

	return t;
}


func (P *Parser) parseOperand() AST.Expr

func (P *Parser) parseStructType() *AST.Type {
	if P.trace {
		defer un(trace(P, "StructType"));
	}

	t := AST.NewType(P.pos, AST.STRUCT);
	P.expect(Scanner.STRUCT);
	if P.tok == Scanner.LBRACE {
		P.next();

		t.List = vector.New(0);
		t.Scope = SymbolTable.NewScope(nil);
		for P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
			P.parseVarList(t.List, false);
			if P.tok == Scanner.STRING {
				// ParseOperand takes care of string concatenation
				t.List.Push(P.parseOperand());
			}
			if P.tok == Scanner.SEMICOLON {
				P.next();
			} else {
				break;
			}
		}
		P.OptSemicolon();
		t.End = P.pos;

		P.expect(Scanner.RBRACE);

		// enter fields into struct scope
		for i, n := 0, t.List.Len(); i < n; i++ {
			if x, ok := t.List.At(i).(*AST.Ident); ok {
				P.declareInScope(t.Scope, x, SymbolTable.FIELD, nil);
			}
		}
	}

	return t;
}


func (P *Parser) parsePointerType() *AST.Type {
	if P.trace {
		defer un(trace(P, "PointerType"));
	}

	t := AST.NewType(P.pos, AST.POINTER);
	P.expect(Scanner.MUL);
	t.Elt = P.parseType();

	return t;
}


func (P *Parser) tryType() *AST.Type {
	if P.trace {
		defer un(trace(P, "Type (try)"));
	}

	switch P.tok {
	case Scanner.IDENT: return P.parseTypeName();
	case Scanner.LBRACK: return P.parseArrayType();
	case Scanner.CHAN, Scanner.ARROW: return P.parseChannelType();
	case Scanner.INTERFACE: return P.parseInterfaceType();
	case Scanner.FUNC: return P.parseFunctionType();
	case Scanner.MAP: return P.parseMapType();
	case Scanner.STRUCT: return P.parseStructType();
	case Scanner.MUL: return P.parsePointerType();
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
	for P.tok != Scanner.CASE && P.tok != Scanner.DEFAULT && P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
		if expect_semi {
			P.expect(Scanner.SEMICOLON);
			expect_semi = false;
		}
		list.Push(P.parseStatement());
		if P.tok == Scanner.SEMICOLON {
			P.next();
		} else if P.opt_semi {
			P.opt_semi = false;  // "consume" optional semicolon
		} else {
			expect_semi = true;
		}
	}
}


func (P *Parser) parseBlock(ftyp *AST.Type, tok int) *AST.Block {
	if P.trace {
		defer un(trace(P, "Block"));
	}

	b := AST.NewBlock(P.pos, tok);
	P.expect(tok);

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

	P.parseStatementList(b.List);
	P.closeScope();

	if tok == Scanner.LBRACE {
		b.End = P.pos;
		P.expect(Scanner.RBRACE);
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
	for first := true; P.tok == Scanner.COMMA; {
		pos := P.pos;
		P.next();
		y := P.parseExpression(1);
		if first {
			x = &AST.BinaryExpr(pos, Scanner.COMMA, x, y);
			first = false;
		} else {
			x.(*AST.BinaryExpr).Y = &AST.BinaryExpr(pos, Scanner.COMMA, x.(*AST.BinaryExpr).Y, y);
		}
	}

	return x;
}


func (P *Parser) parseFunctionLit() AST.Expr {
	if P.trace {
		defer un(trace(P, "FunctionLit"));
	}

	pos := P.pos;
	P.expect(Scanner.FUNC);
	typ := P.parseSignature();
	P.scope_lev++;
	body := P.parseBlock(typ, Scanner.LBRACE);
	P.scope_lev--;

	return &AST.FunctionLit(pos, typ, body);
}


func (P *Parser) parseOperand() AST.Expr {
	if P.trace {
		defer un(trace(P, "Operand"));
	}

	switch P.tok {
	case Scanner.IDENT:
		return P.parseIdent(P.top_scope);

	case Scanner.LPAREN:
		P.next();
		x := P.parseExpression(1);
		P.expect(Scanner.RPAREN);
		return x;

	case Scanner.INT, Scanner.FLOAT, Scanner.STRING:
		x := &AST.BasicLit(P.pos, P.tok, P.val);
		P.next();
		if x.Tok == Scanner.STRING {
			// TODO should remember the list instead of
			//      concatenate the strings here
			for ; P.tok == Scanner.STRING; P.next() {
				x.Val += P.val;
			}
		}
		return x;

	case Scanner.FUNC:
		return P.parseFunctionLit();

	default:
		t := P.tryType();
		if t != nil {
			return &AST.TypeLit(t);
		} else {
			P.error(P.pos, "operand expected");
			P.next();  // make progress
		}
	}

	return &AST.BadExpr(P.pos);
}


func (P *Parser) parseSelectorOrTypeGuard(x AST.Expr) AST.Expr {
	if P.trace {
		defer un(trace(P, "SelectorOrTypeGuard"));
	}

	pos := P.pos;
	P.expect(Scanner.PERIOD);

	if P.tok == Scanner.IDENT {
		x = &AST.Selector(pos, x, P.parseIdent(nil));

	} else {
		P.expect(Scanner.LPAREN);
		x = &AST.TypeGuard(pos, x, P.parseType());
		P.expect(Scanner.RPAREN);
	}

	return x;
}


func (P *Parser) parseIndex(x AST.Expr) AST.Expr {
	if P.trace {
		defer un(trace(P, "IndexOrSlice"));
	}

	pos := P.pos;
	P.expect(Scanner.LBRACK);
	i := P.parseExpression(0);
	P.expect(Scanner.RBRACK);

	return &AST.Index(pos, x, i);
}


func (P *Parser) parseBinaryExpr(prec1 int) AST.Expr

func (P *Parser) parseCompositeElements() AST.Expr {
	x := P.parseExpression(0);
	if P.tok == Scanner.COMMA {
		pos := P.pos;
		P.next();

		// first element determines mode
		singles := true;
		if t, is_binary := x.(*AST.BinaryExpr); is_binary && t.Tok == Scanner.COLON {
			singles = false;
		}

		var last *AST.BinaryExpr;
		for P.tok != Scanner.RPAREN && P.tok != Scanner.EOF {
			y := P.parseExpression(0);

			if singles {
				if t, is_binary := y.(*AST.BinaryExpr); is_binary && t.Tok == Scanner.COLON {
					P.error(t.X.Pos(), "single value expected; found pair");
				}
			} else {
				if t, is_binary := y.(*AST.BinaryExpr); !is_binary || t.Tok != Scanner.COLON {
					P.error(y.Pos(), "key:value pair expected; found single value");
				}
			}

			if last == nil {
				last = &AST.BinaryExpr(pos, Scanner.COMMA, x, y);
				x = last;
			} else {
				last.Y = &AST.BinaryExpr(pos, Scanner.COMMA, last.Y, y);
				last = last.Y.(*AST.BinaryExpr);
			}

			if P.tok == Scanner.COMMA {
				pos = P.pos;
				P.next();
			} else {
				break;
			}

		}
	}
	return x;
}


func (P *Parser) parseCallOrCompositeLit(f AST.Expr) AST.Expr {
	if P.trace {
		defer un(trace(P, "CallOrCompositeLit"));
	}

	pos := P.pos;
	P.expect(Scanner.LPAREN);
	var args AST.Expr;
	if P.tok != Scanner.RPAREN {
		args = P.parseCompositeElements();
	}
	P.expect(Scanner.RPAREN);

	return &AST.Call(pos, f, args);
}


func (P *Parser) parsePrimaryExpr() AST.Expr {
	if P.trace {
		defer un(trace(P, "PrimaryExpr"));
	}

	x := P.parseOperand();
	for {
		switch P.tok {
		case Scanner.PERIOD: x = P.parseSelectorOrTypeGuard(x);
		case Scanner.LBRACK: x = P.parseIndex(x);
		case Scanner.LPAREN: x = P.parseCallOrCompositeLit(x);
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
	case Scanner.ADD, Scanner.SUB, Scanner.MUL, Scanner.NOT, Scanner.XOR, Scanner.ARROW, Scanner.AND:
		pos, tok := P.pos, P.tok;
		P.next();
		y := P.parseUnaryExpr();
		if lit, ok := y.(*AST.TypeLit); ok && tok == Scanner.MUL {
			// pointer type
			t := AST.NewType(pos, AST.POINTER);
			t.Elt = lit.Typ;
			return &AST.TypeLit(t);
		} else {
			return &AST.UnaryExpr(pos, tok, y);
		}
	}

	return P.parsePrimaryExpr();
}


func (P *Parser) parseBinaryExpr(prec1 int) AST.Expr {
	if P.trace {
		defer un(trace(P, "BinaryExpr"));
	}

	x := P.parseUnaryExpr();
	for prec := Scanner.Precedence(P.tok); prec >= prec1; prec-- {
		for Scanner.Precedence(P.tok) == prec {
			pos, tok := P.pos, P.tok;
			P.next();
			y := P.parseBinaryExpr(prec + 1);
			x = &AST.BinaryExpr(pos, tok, x, y);
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
	case Scanner.COLON:
		// label declaration
		pos := P.pos;
		P.next();  // consume ":"
		P.opt_semi = true;
		if AST.ExprLen(x) == 1 {
			if label, is_ident := x.(*AST.Ident); is_ident {
				return &AST.LabelDecl(pos, label);
			}
		}
		P.error(x.Pos(), "illegal label declaration");
		return nil;

	case
		Scanner.DEFINE, Scanner.ASSIGN, Scanner.ADD_ASSIGN,
		Scanner.SUB_ASSIGN, Scanner.MUL_ASSIGN, Scanner.QUO_ASSIGN,
		Scanner.REM_ASSIGN, Scanner.AND_ASSIGN, Scanner.OR_ASSIGN,
		Scanner.XOR_ASSIGN, Scanner.SHL_ASSIGN, Scanner.SHR_ASSIGN:
		// declaration/assignment
		pos, tok := P.pos, P.tok;
		P.next();
		var y AST.Expr;
		if range_ok && P.tok == Scanner.RANGE {
			range_pos := P.pos;
			P.next();
			y = &AST.UnaryExpr(range_pos, Scanner.RANGE, P.parseExpression(1));
			if tok != Scanner.DEFINE && tok != Scanner.ASSIGN {
				P.error(pos, "expected '=' or ':=', found '" + Scanner.TokenString(tok) + "'");
			}
		} else {
			y = P.parseExpressionList();
			if xl, yl := AST.ExprLen(x), AST.ExprLen(y); xl > 1 && yl > 1 && xl != yl {
				P.error(x.Pos(), "arity of lhs doesn't match rhs");
			}
		}
		// TODO changed ILLEGAL -> NONE
		return &AST.ExpressionStat(x.Pos(), Scanner.ILLEGAL, &AST.BinaryExpr(pos, tok, x, y));

	default:
		if AST.ExprLen(x) != 1 {
			P.error(x.Pos(), "only one expression allowed");
		}

		if P.tok == Scanner.INC || P.tok == Scanner.DEC {
			s := &AST.ExpressionStat(P.pos, P.tok, x);
			P.next();  // consume "++" or "--"
			return s;
		}

		// TODO changed ILLEGAL -> NONE
		return &AST.ExpressionStat(x.Pos(), Scanner.ILLEGAL, x);
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
	return &AST.ExpressionStat(pos, keyword, P.parseExpression(1));
}


func (P *Parser) parseReturnStat() *AST.ExpressionStat {
	if P.trace {
		defer un(trace(P, "ReturnStat"));
	}

	pos := P.pos;
	P.expect(Scanner.RETURN);
	var x AST.Expr;
	if P.tok != Scanner.SEMICOLON && P.tok != Scanner.RBRACE {
		x = P.parseExpressionList();
	}

	return &AST.ExpressionStat(pos, Scanner.RETURN, x);
}


func (P *Parser) parseControlFlowStat(tok int) *AST.ControlFlowStat {
	if P.trace {
		defer un(trace(P, "ControlFlowStat"));
	}

	s := &AST.ControlFlowStat(P.pos, tok, nil);
	P.expect(tok);
	if tok != Scanner.FALLTHROUGH && P.tok == Scanner.IDENT {
		s.Label = P.parseIdent(P.top_scope);
	}

	return s;
}


func (P *Parser) parseControlClause(isForStat bool) (init AST.Stat, expr AST.Expr, post AST.Stat) {
	if P.trace {
		defer un(trace(P, "ControlClause"));
	}

	if P.tok != Scanner.LBRACE {
		if P.tok != Scanner.SEMICOLON {
			init = P.parseSimpleStat(isForStat);
			// TODO check for range clause and exit if found
		}
		if P.tok == Scanner.SEMICOLON {
			P.next();
			if P.tok != Scanner.SEMICOLON && P.tok != Scanner.LBRACE {
				expr = P.parseExpression(1);
			}
			if isForStat {
				P.expect(Scanner.SEMICOLON);
				if P.tok != Scanner.LBRACE {
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
	}

	return init, expr, post;
}


func (P *Parser) parseIfStat() *AST.IfStat {
	if P.trace {
		defer un(trace(P, "IfStat"));
	}

	P.openScope();
	pos := P.pos;
	P.expect(Scanner.IF);
	init, cond, dummy := P.parseControlClause(false);
	body := P.parseBlock(nil, Scanner.LBRACE);
	var else_ AST.Stat;
	if P.tok == Scanner.ELSE {
		P.next();
		if ok := P.tok == Scanner.IF || P.tok == Scanner.LBRACE; ok || P.sixg {
			else_ = P.parseStatement();
			if !ok {
				// wrap in a block since we don't have one
				body := AST.NewBlock(0, Scanner.LBRACE);
				body.List.Push(else_);
				else_ = &AST.CompositeStat(body);
			}
		} else {
			P.error(P.pos, "'if' or '{' expected - illegal 'else' branch");
		}
	}
	P.closeScope();

	return &AST.IfStat(pos, init, cond, body, else_);
}


func (P *Parser) parseForStat() *AST.ForStat {
	if P.trace {
		defer un(trace(P, "ForStat"));
	}

	P.openScope();
	pos := P.pos;
	P.expect(Scanner.FOR);
	init, cond, post := P.parseControlClause(true);
	body := P.parseBlock(nil, Scanner.LBRACE);
	P.closeScope();

	return &AST.ForStat(pos, init, cond, post, body);
}


func (P *Parser) parseCaseClause() *AST.CaseClause {
	if P.trace {
		defer un(trace(P, "CaseClause"));
	}

	// SwitchCase
	pos := P.pos;
	var expr AST.Expr;
	if P.tok == Scanner.CASE {
		P.next();
		expr = P.parseExpressionList();
	} else {
		P.expect(Scanner.DEFAULT);
	}

	return &AST.CaseClause(pos, expr, P.parseBlock(nil, Scanner.COLON));
}


func (P *Parser) parseSwitchStat() *AST.SwitchStat {
	if P.trace {
		defer un(trace(P, "SwitchStat"));
	}

	P.openScope();
	pos := P.pos;
	P.expect(Scanner.SWITCH);
	init, tag, post := P.parseControlClause(false);
	body := AST.NewBlock(P.pos, Scanner.LBRACE);
	P.expect(Scanner.LBRACE);
	for P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
		body.List.Push(P.parseCaseClause());
	}
	body.End = P.pos;
	P.expect(Scanner.RBRACE);
	P.opt_semi = true;
	P.closeScope();

	return &AST.SwitchStat(pos, init, tag, body);
}


func (P *Parser) parseCommClause() *AST.CaseClause {
	if P.trace {
		defer un(trace(P, "CommClause"));
	}

	// CommCase
	pos := P.pos;
	var expr AST.Expr;
	if P.tok == Scanner.CASE {
		P.next();
		x := P.parseExpression(1);
		if P.tok == Scanner.ASSIGN || P.tok == Scanner.DEFINE {
			pos, tok := P.pos, P.tok;
			P.next();
			if P.tok == Scanner.ARROW {
				y := P.parseExpression(1);
				x = &AST.BinaryExpr(pos, tok, x, y);
			} else {
				P.expect(Scanner.ARROW);  // use expect() error handling
			}
		}
		expr = x;
	} else {
		P.expect(Scanner.DEFAULT);
	}

	return &AST.CaseClause(pos, expr, P.parseBlock(nil, Scanner.COLON));
}


func (P *Parser) parseSelectStat() *AST.SelectStat {
	if P.trace {
		defer un(trace(P, "SelectStat"));
	}

	P.openScope();
	pos := P.pos;
	P.expect(Scanner.SELECT);
	body := AST.NewBlock(P.pos, Scanner.LBRACE);
	P.expect(Scanner.LBRACE);
	for P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
		body.List.Push(P.parseCommClause());
	}
	body.End = P.pos;
	P.expect(Scanner.RBRACE);
	P.opt_semi = true;
	P.closeScope();

	return &AST.SelectStat(pos, body);
}


func (P *Parser) parseStatement() AST.Stat {
	if P.trace {
		defer un(trace(P, "Statement"));
	}

	switch P.tok {
	case Scanner.CONST, Scanner.TYPE, Scanner.VAR:
		return &AST.DeclarationStat(P.parseDeclaration());
	case Scanner.FUNC:
		// for now we do not allow local function declarations,
		// instead we assume this starts a function literal
		fallthrough;
	case
		// only the tokens that are legal top-level expression starts
		Scanner.IDENT, Scanner.INT, Scanner.FLOAT, Scanner.STRING, Scanner.LPAREN,  // operand
		Scanner.LBRACK, Scanner.STRUCT,  // composite type
		Scanner.MUL, Scanner.AND, Scanner.ARROW:  // unary
		return P.parseSimpleStat(false);
	case Scanner.GO, Scanner.DEFER:
		return P.parseInvocationStat(P.tok);
	case Scanner.RETURN:
		return P.parseReturnStat();
	case Scanner.BREAK, Scanner.CONTINUE, Scanner.GOTO, Scanner.FALLTHROUGH:
		return P.parseControlFlowStat(P.tok);
	case Scanner.LBRACE:
		return &AST.CompositeStat(P.parseBlock(nil, Scanner.LBRACE));
	case Scanner.IF:
		return P.parseIfStat();
	case Scanner.FOR:
		return P.parseForStat();
	case Scanner.SWITCH:
		return P.parseSwitchStat();
	case Scanner.SELECT:
		return P.parseSelectStat();
	case Scanner.SEMICOLON:
		// don't consume the ";", it is the separator following the empty statement
		return &AST.EmptyStat(P.pos);
	}

	// no statement found
	P.error(P.pos, "statement expected");
	return &AST.BadStat(P.pos);
}


// ----------------------------------------------------------------------------
// Declarations

func (P *Parser) parseImportSpec(d *AST.Decl) {
	if P.trace {
		defer un(trace(P, "ImportSpec"));
	}

	if P.tok == Scanner.PERIOD {
		P.error(P.pos, `"import ." not yet handled properly`);
		P.next();
	} else if P.tok == Scanner.IDENT {
		d.Ident = P.parseIdent(nil);
	}

	if P.tok == Scanner.STRING {
		// TODO eventually the scanner should strip the quotes
		d.Val = &AST.BasicLit(P.pos, Scanner.STRING, P.val);
		P.next();
	} else {
		P.expect(Scanner.STRING);  // use expect() error handling
	}
}


func (P *Parser) parseConstSpec(d *AST.Decl) {
	if P.trace {
		defer un(trace(P, "ConstSpec"));
	}

	d.Ident = P.parseIdentList();
	d.Typ = P.tryType();
	if P.tok == Scanner.ASSIGN {
		P.next();
		d.Val = P.parseExpressionList();
	}
}


func (P *Parser) parseTypeSpec(d *AST.Decl) {
	if P.trace {
		defer un(trace(P, "TypeSpec"));
	}

	d.Ident = P.parseIdent(nil);
	d.Typ = P.parseType();
	P.opt_semi = true;
}


func (P *Parser) parseVarSpec(d *AST.Decl) {
	if P.trace {
		defer un(trace(P, "VarSpec"));
	}

	d.Ident = P.parseIdentList();
	if P.tok == Scanner.ASSIGN {
		P.next();
		d.Val = P.parseExpressionList();
	} else {
		d.Typ = P.parseVarType();
		if P.tok == Scanner.ASSIGN {
			P.next();
			d.Val = P.parseExpressionList();
		}
	}
}


func (P *Parser) parseSpec(d *AST.Decl) {
	kind := SymbolTable.NONE;

	switch d.Tok {
	case Scanner.IMPORT: P.parseImportSpec(d); kind = SymbolTable.PACKAGE;
	case Scanner.CONST: P.parseConstSpec(d); kind = SymbolTable.CONST;
	case Scanner.TYPE: P.parseTypeSpec(d); kind = SymbolTable.TYPE;
	case Scanner.VAR: P.parseVarSpec(d); kind = SymbolTable.VAR;
	default: unreachable();
	}

	// semantic checks
	if d.Tok == Scanner.IMPORT {
		if d.Ident != nil {
			P.declare(d.Ident, kind, nil);
		}
	} else {
		P.declare(d.Ident, kind, d.Typ);
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
}


func (P *Parser) parseDecl(keyword int) *AST.Decl {
	if P.trace {
		defer un(trace(P, "Decl"));
	}

	d := AST.NewDecl(P.pos, keyword);
	P.expect(keyword);
	if P.tok == Scanner.LPAREN {
		P.next();
		d.List = vector.New(0);
		for P.tok != Scanner.RPAREN && P.tok != Scanner.EOF {
			d1 := AST.NewDecl(P.pos, keyword);
			P.parseSpec(d1);
			d.List.Push(d1);
			if P.tok == Scanner.SEMICOLON {
				P.next();
			} else {
				break;
			}
		}
		d.End = P.pos;
		P.expect(Scanner.RPAREN);
		P.opt_semi = true;

	} else {
		P.parseSpec(d);
	}

	return d;
}


// Function declarations
//
// func        ident (params)
// func        ident (params) type
// func        ident (params) (results)
// func (recv) ident (params)
// func (recv) ident (params) type
// func (recv) ident (params) (results)

func (P *Parser) parseFunctionDecl() *AST.Decl {
	if P.trace {
		defer un(trace(P, "FunctionDecl"));
	}

	d := AST.NewDecl(P.pos, Scanner.FUNC);
	P.expect(Scanner.FUNC);

	var recv *AST.Type;
	if P.tok == Scanner.LPAREN {
		pos := P.pos;
		recv = P.parseParameters(true);
		if recv.Nfields() != 1 {
			P.error(pos, "must have exactly one receiver");
		}
	}

	ident := P.parseIdent(nil);
	d.Ident = ident;
	d.Typ = P.parseSignature();
	d.Typ.Key = recv;

	if P.tok == Scanner.LBRACE {
		d.Body = P.parseBlock(d.Typ, Scanner.LBRACE);
	}

	return d;
}


func (P *Parser) parseDeclaration() *AST.Decl {
	if P.trace {
		defer un(trace(P, "Declaration"));
	}

	d := AST.BadDecl;

	switch P.tok {
	case Scanner.CONST, Scanner.TYPE, Scanner.VAR:
		d = P.parseDecl(P.tok);
	case Scanner.FUNC:
		d = P.parseFunctionDecl();
	default:
		P.error(P.pos, "declaration expected");
		P.next();  // make progress
	}

	return d;
}


// ----------------------------------------------------------------------------
// Program

func (P *Parser) ParseProgram() *AST.Program {
	if P.trace {
		defer un(trace(P, "Program"));
	}

	P.openScope();
	p := AST.NewProgram(P.pos);
	P.expect(Scanner.PACKAGE);
	p.Ident = P.parseIdent(nil);

	// package body
	{	P.openScope();
		p.Decls = vector.New(0);
		for P.tok == Scanner.IMPORT {
			p.Decls.Push(P.parseDecl(Scanner.IMPORT));
			P.OptSemicolon();
		}
		if !P.deps {
			for P.tok != Scanner.EOF {
				p.Decls.Push(P.parseDeclaration());
				P.OptSemicolon();
			}
		}
		P.closeScope();
	}

	p.Comments = P.comments;
	P.closeScope();

	return p;
}
