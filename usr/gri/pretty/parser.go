// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Parser

import (
	"array";
	Scanner "scanner";
	AST "ast";
)


export type Parser struct {
	// Tracing/debugging
	verbose, sixg, deps bool;
	indent uint;

	// Scanner
	scanner *Scanner.Scanner;
	tokchan <-chan *Scanner.Token;
	comments *array.Array;

	// Scanner.Token
	pos int;  // token source position
	tok int;  // one token look-ahead
	val string;  // token value (for IDENT, NUMBER, STRING only)

	// Non-syntactic parser control
	opt_semi bool;  // true if semicolon is optional

	// Nesting levels
	expr_lev int;  // 0 = control clause level, 1 = expr inside ()'s
	scope_lev int;  // 0 = global scope, 1 = function scope of global functions, etc.
	
	// Scopes
	top_scope *AST.Scope;
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

func (P *Parser) PrintIndent() {
	for i := P.indent; i > 0; i-- {
		print(". ");
	}
}


func (P *Parser) Trace(msg string) {
	if P.verbose {
		P.PrintIndent();
		print(msg, " {\n");
	}
	P.indent++;  // always check proper identation
}


func (P *Parser) Ecart() {
	P.indent--;  // always check proper identation
	if P.verbose {
		P.PrintIndent();
		print("}\n");
	}
}


func (P *Parser) Next0() {
	if P.tokchan == nil {
		P.pos, P.tok, P.val = P.scanner.Scan();
	} else {
		t := <-P.tokchan;
		P.tok, P.pos, P.val = t.tok, t.pos, t.val;
	}
	P.opt_semi = false;

	if P.verbose {
		P.PrintIndent();
		print("[", P.pos, "] ", Scanner.TokenString(P.tok), "\n");
	}
}


func (P *Parser) Next() {
	for P.Next0(); P.tok == Scanner.COMMENT; P.Next0() {
		P.comments.Push(AST.NewComment(P.pos, P.val));
	}
}


func (P *Parser) Open(verbose, sixg, deps bool, scanner *Scanner.Scanner, tokchan <-chan *Scanner.Token) {
	P.verbose = verbose;
	P.sixg = sixg;
	P.deps = deps;
	P.indent = 0;

	P.scanner = scanner;
	P.tokchan = tokchan;
	P.comments = array.New(0);

	P.Next();
	P.expr_lev = 0;
	P.scope_lev = 0;
}


func (P *Parser) Error(pos int, msg string) {
	P.scanner.Error(pos, msg);
}


func (P *Parser) Expect(tok int) {
	if P.tok != tok {
		msg := "expected '" + Scanner.TokenString(tok) + "', found '" + Scanner.TokenString(P.tok) + "'";
		switch P.tok {
		case Scanner.IDENT, Scanner.INT, Scanner.FLOAT, Scanner.STRING:
			msg += " " + P.val;
		}
		P.Error(P.pos, msg);
	}
	P.Next();  // make progress in any case
}


func (P *Parser) OptSemicolon() {
	if P.tok == Scanner.SEMICOLON {
		P.Next();
	}
}


// ----------------------------------------------------------------------------
// Scopes

func (P *Parser) OpenScope() {
	P.top_scope = AST.NewScope(P.top_scope);
}


func (P *Parser) CloseScope() {
	P.top_scope = P.top_scope.parent;
}


func (P *Parser) DeclareInScope(scope *AST.Scope, x *AST.Expr, kind int) {
	if P.scope_lev < 0 {
		panic("cannot declare objects in other packages");
	}
	obj := x.obj;
	assert(x.tok == Scanner.IDENT && obj.kind == AST.NONE);
	obj.kind = kind;
	obj.pnolev = P.scope_lev;
	if scope.LookupLocal(obj.ident) != nil {
		P.Error(obj.pos, `"` + obj.ident + `" is declared already`);
		return;  // don't insert it into the scope
	}
	scope.Insert(obj);
}


// Declare a comma-separated list of idents or a single ident.
func (P *Parser) Declare(p *AST.Expr, kind int) {
	for p.tok == Scanner.COMMA {
		P.DeclareInScope(P.top_scope, p.x, kind);
		p = p.y;
	}
	P.DeclareInScope(P.top_scope, p, kind);
}


// ----------------------------------------------------------------------------
// AST support

func ExprType(x *AST.Expr) *AST.Type {
	var t *AST.Type;
	if x.tok == Scanner.TYPE {
		t = x.obj.typ;
	} else if x.tok == Scanner.IDENT {
		// assume a type name
		t = AST.NewType(x.pos, AST.TYPENAME);
		t.expr = x;
	} else if x.tok == Scanner.PERIOD && x.y != nil && ExprType(x.x) != nil {
		// possibly a qualified (type) identifier
		t = AST.NewType(x.pos, AST.TYPENAME);
		t.expr = x;
	}
	return t;
}


func (P *Parser) NoType(x *AST.Expr) *AST.Expr {
	if x != nil && x.tok == Scanner.TYPE {
		P.Error(x.pos, "expected expression, found type");
		val := AST.NewObject(x.pos, AST.NONE, "0");
		x = AST.NewLit(Scanner.INT, val);
	}
	return x;
}


func (P *Parser) NewExpr(pos, tok int, x, y *AST.Expr) *AST.Expr {
	return AST.NewExpr(pos, tok, P.NoType(x), P.NoType(y));
}


// ----------------------------------------------------------------------------
// Common productions

func (P *Parser) TryType() *AST.Type;
func (P *Parser) ParseExpression(prec int) *AST.Expr;
func (P *Parser) ParseStatement() *AST.Stat;
func (P *Parser) ParseDeclaration() *AST.Decl;


// If scope != nil, lookup identifier in scope. Otherwise create one.
func (P *Parser) ParseIdent(scope *AST.Scope) *AST.Expr {
	P.Trace("Ident");
	
	x := AST.BadExpr;
	if P.tok == Scanner.IDENT {
		var obj *AST.Object;
		if scope != nil {
			obj = scope.Lookup(P.val);
		}
		if obj == nil {
			obj = AST.NewObject(P.pos, AST.NONE, P.val);
		} else {
			assert(obj.kind != AST.NONE);
		}
		x = AST.NewLit(Scanner.IDENT, obj);
		x.pos = P.pos;  // override obj.pos (incorrect if object was looked up!)
		if P.verbose {
			P.PrintIndent();
			print("Ident = \"", P.val, "\"\n");
		}
		P.Next();
	} else {
		P.Expect(Scanner.IDENT);  // use Expect() error handling
	}

	P.Ecart();
	return x;
}


func (P *Parser) ParseIdentList() *AST.Expr {
	P.Trace("IdentList");

	var last *AST.Expr;
	x := P.ParseIdent(nil);
	for P.tok == Scanner.COMMA {
		pos := P.pos;
		P.Next();
		y := P.ParseIdent(nil);
		if last == nil {
			x = P.NewExpr(pos, Scanner.COMMA, x, y);
			last = x;
		} else {
			last.y = P.NewExpr(pos, Scanner.COMMA, last.y, y);
			last = last.y;
		}
	}

	P.Ecart();
	return x;
}


// ----------------------------------------------------------------------------
// Types

func (P *Parser) ParseType() *AST.Type {
	P.Trace("Type");

	t := P.TryType();
	if t == nil {
		P.Error(P.pos, "type expected");
		t = AST.BadType;
	}

	P.Ecart();
	return t;
}


func (P *Parser) ParseVarType() *AST.Type {
	P.Trace("VarType");

	typ := P.ParseType();

	P.Ecart();
	return typ;
}


func (P *Parser) ParseQualifiedIdent() *AST.Expr {
	P.Trace("QualifiedIdent");

	x := P.ParseIdent(P.top_scope);
	for P.tok == Scanner.PERIOD {
		pos := P.pos;
		P.Next();
		y := P.ParseIdent(nil);
		x = P.NewExpr(pos, Scanner.PERIOD, x, y);
	}

	P.Ecart();
	return x;
}


func (P *Parser) ParseTypeName() *AST.Type {
	P.Trace("TypeName");

	t := AST.NewType(P.pos, AST.TYPENAME);
	t.expr = P.ParseQualifiedIdent();

	P.Ecart();
	return t;
}


func (P *Parser) ParseArrayType() *AST.Type {
	P.Trace("ArrayType");

	t := AST.NewType(P.pos, AST.ARRAY);
	P.Expect(Scanner.LBRACK);
	if P.tok == Scanner.ELLIPSIS {
		t.expr = P.NewExpr(P.pos, Scanner.ELLIPSIS, nil, nil);
		P.Next();
	} else if P.tok != Scanner.RBRACK {
		t.expr = P.ParseExpression(1);
	}
	P.Expect(Scanner.RBRACK);
	t.elt = P.ParseType();

	P.Ecart();
	return t;
}


func (P *Parser) ParseChannelType() *AST.Type {
	P.Trace("ChannelType");

	t := AST.NewType(P.pos, AST.CHANNEL);
	t.mode = AST.FULL;
	if P.tok == Scanner.CHAN {
		P.Next();
		if P.tok == Scanner.ARROW {
			P.Next();
			t.mode = AST.SEND;
		}
	} else {
		P.Expect(Scanner.ARROW);
		P.Expect(Scanner.CHAN);
		t.mode = AST.RECV;
	}
	t.elt = P.ParseVarType();

	P.Ecart();
	return t;
}


func (P *Parser) ParseVar(expect_ident bool) *AST.Type {
	t := AST.BadType;
	if expect_ident {
		x := P.ParseIdent(nil);
		t = AST.NewType(x.pos, AST.TYPENAME);
		t.expr = x;
	} else if P.tok == Scanner.ELLIPSIS {
		t = AST.NewType(P.pos, AST.ELLIPSIS);
		P.Next();
	} else {
		t = P.ParseType();
	}
	return t;
}


func (P *Parser) ParseVarList(list *array.Array, ellipsis_ok bool) {
	P.Trace("VarList");

	// assume a list of types
	// (a list of identifiers looks like a list of type names)
	i0 := list.Len();
	for {
		list.Push(P.ParseVar(ellipsis_ok /* param list */ && i0 > 0));
		if P.tok == Scanner.COMMA {
			P.Next();
		} else {
			break;
		}
	}

	// if we had a list of identifiers, it must be followed by a type
	typ := P.TryType();
	if typ == nil && P.tok == Scanner.ELLIPSIS {
		typ = AST.NewType(P.pos, AST.ELLIPSIS);
		P.Next();
	}

	if ellipsis_ok /* param list */ && i0 > 0 && typ == nil {
		// not the first parameter section; we must have a type
		P.Error(P.pos, "type expected");
		typ = AST.BadType;
	}

	// convert the list into a list of (type) expressions
	if typ != nil {
		// all list entries must be identifiers
		// convert the type entries into identifiers
		for i, n := i0, list.Len(); i < n; i++ {
			t := list.At(i).(*AST.Type);
			if t.form == AST.TYPENAME && t.expr.tok == Scanner.IDENT {
				list.Set(i, t.expr);
			} else {
				list.Set(i, AST.BadExpr);
				P.Error(t.pos, "identifier expected");
			}
		}
		// add type
		list.Push(AST.NewTypeExpr(typ));

	} else {
		// all list entries are types
		// convert all type entries into type expressions
		for i, n := i0, list.Len(); i < n; i++ {
			t := list.At(i).(*AST.Type);
			list.Set(i, AST.NewTypeExpr(t));
		}
	}

	P.Ecart();
}


func (P *Parser) ParseParameterList(ellipsis_ok bool) *array.Array {
	P.Trace("ParameterList");

	list := array.New(0);
	P.ParseVarList(list, ellipsis_ok);
	for P.tok == Scanner.COMMA {
		P.Next();
		P.ParseVarList(list, ellipsis_ok);
	}

	P.Ecart();
	return list;
}


func (P *Parser) ParseParameters(ellipsis_ok bool) *AST.Type {
	P.Trace("Parameters");

	t := AST.NewType(P.pos, AST.STRUCT);
	P.Expect(Scanner.LPAREN);
	if P.tok != Scanner.RPAREN {
		t.list = P.ParseParameterList(ellipsis_ok);
	}
	t.end = P.pos;
	P.Expect(Scanner.RPAREN);

	P.Ecart();
	return t;
}


func (P *Parser) ParseResultList() {
	P.Trace("ResultList");

	P.ParseType();
	for P.tok == Scanner.COMMA {
		P.Next();
		P.ParseType();
	}
	if P.tok != Scanner.RPAREN {
		P.ParseType();
	}

	P.Ecart();
}


func (P *Parser) ParseResult() *AST.Type {
	P.Trace("Result");

	var t *AST.Type;
	if P.tok == Scanner.LPAREN {
		t = P.ParseParameters(false);
	} else {
		typ := P.TryType();
		if typ != nil {
			t = AST.NewType(P.pos, AST.STRUCT);
			t.list = array.New(0);
			t.list.Push(AST.NewTypeExpr(typ));
			t.end = P.pos;
		}
	}

	P.Ecart();
	return t;
}


// Function types
//
// (params)
// (params) type
// (params) (results)

func (P *Parser) ParseFunctionType() *AST.Type {
	P.Trace("FunctionType");

	P.OpenScope();
	P.scope_lev++;

	t := AST.NewType(P.pos, AST.FUNCTION);
	t.list = P.ParseParameters(true).list;  // TODO find better solution
	t.end = P.pos;
	t.elt = P.ParseResult();

	P.scope_lev--;
	P.CloseScope();

	P.Ecart();
	return t;
}


func (P *Parser) ParseMethodSpec(list *array.Array) {
	P.Trace("MethodDecl");

	list.Push(P.ParseIdentList());
	t := AST.BadType;
	if P.sixg {
		t = P.ParseType();
	} else {
		t = P.ParseFunctionType();
	}
	list.Push(AST.NewTypeExpr(t));

	P.Ecart();
}


func (P *Parser) ParseInterfaceType() *AST.Type {
	P.Trace("InterfaceType");

	t := AST.NewType(P.pos, AST.INTERFACE);
	P.Expect(Scanner.INTERFACE);
	if P.tok == Scanner.LBRACE {
		P.Next();
		P.OpenScope();
		P.scope_lev++;

		t.list = array.New(0);
		for P.tok == Scanner.IDENT {
			P.ParseMethodSpec(t.list);
			if P.tok != Scanner.RBRACE {
				P.Expect(Scanner.SEMICOLON);
			}
		}
		t.end = P.pos;

		P.scope_lev--;
		P.CloseScope();
		P.Expect(Scanner.RBRACE);
	}

	P.Ecart();
	return t;
}


func (P *Parser) ParseMapType() *AST.Type {
	P.Trace("MapType");

	t := AST.NewType(P.pos, AST.MAP);
	P.Expect(Scanner.MAP);
	P.Expect(Scanner.LBRACK);
	t.key = P.ParseVarType();
	P.Expect(Scanner.RBRACK);
	t.elt = P.ParseVarType();

	P.Ecart();
	return t;
}


func (P *Parser) ParseOperand() *AST.Expr

func (P *Parser) ParseStructType() *AST.Type {
	P.Trace("StructType");

	t := AST.NewType(P.pos, AST.STRUCT);
	P.Expect(Scanner.STRUCT);
	if P.tok == Scanner.LBRACE {
		P.Next();
		P.OpenScope();
		P.scope_lev++;

		t.list = array.New(0);
		for P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
			P.ParseVarList(t.list, false);
			if P.tok == Scanner.STRING {
				// ParseOperand takes care of string concatenation
				t.list.Push(P.ParseOperand());
			}
			if P.tok == Scanner.SEMICOLON {
				P.Next();
			} else {
				break;
			}
		}
		P.OptSemicolon();
		t.end = P.pos;

		P.scope_lev--;
		P.CloseScope();
		P.Expect(Scanner.RBRACE);
	}

	P.Ecart();
	return t;
}


func (P *Parser) ParsePointerType() *AST.Type {
	P.Trace("PointerType");

	t := AST.NewType(P.pos, AST.POINTER);
	P.Expect(Scanner.MUL);
	t.elt = P.ParseType();

	P.Ecart();
	return t;
}


func (P *Parser) TryType() *AST.Type {
	P.Trace("Type (try)");

	t := AST.BadType;
	switch P.tok {
	case Scanner.IDENT: t = P.ParseTypeName();
	case Scanner.LBRACK: t = P.ParseArrayType();
	case Scanner.CHAN, Scanner.ARROW: t = P.ParseChannelType();
	case Scanner.INTERFACE: t = P.ParseInterfaceType();
	case Scanner.LPAREN: t = P.ParseFunctionType();
	case Scanner.MAP: t = P.ParseMapType();
	case Scanner.STRUCT: t = P.ParseStructType();
	case Scanner.MUL: t = P.ParsePointerType();
	default: t = nil;  // no type found
	}

	P.Ecart();
	return t;
}


// ----------------------------------------------------------------------------
// Blocks

func (P *Parser) ParseStatementList() *array.Array {
	P.Trace("StatementList");

	list := array.New(0);
	for P.tok != Scanner.CASE && P.tok != Scanner.DEFAULT && P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
		s := P.ParseStatement();
		if s != nil {
			// not the empty statement
			list.Push(s);
		}
		if P.tok == Scanner.SEMICOLON {
			P.Next();
		} else if P.opt_semi {
			P.opt_semi = false;  // "consume" optional semicolon
		} else {
			break;
		}
	}

	// Try to provide a good error message
	if P.tok != Scanner.CASE && P.tok != Scanner.DEFAULT && P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
		P.Error(P.pos, "expected end of statement list (semicolon missing?)");
	}

	P.Ecart();
	return list;
}


func (P *Parser) ParseBlock() (slist *array.Array, end int) {
	P.Trace("Block");

	P.Expect(Scanner.LBRACE);
	P.OpenScope();

	slist = P.ParseStatementList();
	end = P.pos;

	P.CloseScope();
	P.Expect(Scanner.RBRACE);
	P.opt_semi = true;

	P.Ecart();
	return slist, end;
}


// ----------------------------------------------------------------------------
// Expressions

func (P *Parser) ParseExpressionList() *AST.Expr {
	P.Trace("ExpressionList");

	x := P.ParseExpression(1);
	for first := true; P.tok == Scanner.COMMA; {
		pos := P.pos;
		P.Next();
		y := P.ParseExpression(1);
		if first {
			x = P.NewExpr(pos, Scanner.COMMA, x, y);
			first = false;
		} else {
			x.y = P.NewExpr(pos, Scanner.COMMA, x.y, y);
		}
	}

	P.Ecart();
	return x;
}


func (P *Parser) ParseFunctionLit() *AST.Expr {
	P.Trace("FunctionLit");

	val := AST.NewObject(P.pos, AST.NONE, "");
	x := AST.NewLit(Scanner.FUNC, val);
	P.Expect(Scanner.FUNC);
	val.typ = P.ParseFunctionType();
	P.expr_lev++;
	P.scope_lev++;
	val.block, val.end = P.ParseBlock();
	P.scope_lev--;
	P.expr_lev--;

	P.Ecart();
	return x;
}


/*
func (P *Parser) ParseNewCall() *AST.Expr {
	P.Trace("NewCall");

	x := AST.NewExpr(P.pos, Scanner.NEW, nil, nil);
	P.Next();
	P.Expect(Scanner.LPAREN);
	P.expr_lev++;
	x.t = P.ParseType();
	if P.tok == Scanner.COMMA {
		P.Next();
		x.y = P.ParseExpressionList();
	}
	P.expr_lev--;
	P.Expect(Scanner.RPAREN);

	P.Ecart();
	return x;
}
*/


func (P *Parser) ParseOperand() *AST.Expr {
	P.Trace("Operand");

	x := AST.BadExpr;
	switch P.tok {
	case Scanner.IDENT:
		x = P.ParseIdent(P.top_scope);

	case Scanner.LPAREN:
		// TODO we could have a function type here as in: new(())
		// (currently not working)
		P.Next();
		P.expr_lev++;
		x = P.ParseExpression(1);
		P.expr_lev--;
		P.Expect(Scanner.RPAREN);

	case Scanner.INT, Scanner.FLOAT, Scanner.STRING:
		val := AST.NewObject(P.pos, AST.NONE, P.val);
		x = AST.NewLit(P.tok, val);
		P.Next();
		if x.tok == Scanner.STRING {
			// TODO should remember the list instead of
			//      concatenate the strings here
			for ; P.tok == Scanner.STRING; P.Next() {
				x.obj.ident += P.val;
			}
		}

	case Scanner.FUNC:
		x = P.ParseFunctionLit();

	default:
		t := P.TryType();
		if t != nil {
			x = AST.NewTypeExpr(t);
		} else {
			P.Error(P.pos, "operand expected");
			P.Next();  // make progress
		}
	}

	P.Ecart();
	return x;
}


func (P *Parser) ParseSelectorOrTypeGuard(x *AST.Expr) *AST.Expr {
	P.Trace("SelectorOrTypeGuard");

	x = P.NewExpr(P.pos, Scanner.PERIOD, x, nil);
	P.Expect(Scanner.PERIOD);

	if P.tok == Scanner.IDENT {
		x.y = P.ParseIdent(nil);

	} else {
		P.Expect(Scanner.LPAREN);
		x.y = AST.NewTypeExpr(P.ParseType());
		P.Expect(Scanner.RPAREN);
	}

	P.Ecart();
	return x;
}


func (P *Parser) ParseIndex(x *AST.Expr) *AST.Expr {
	P.Trace("IndexOrSlice");

	pos := P.pos;
	P.Expect(Scanner.LBRACK);
	P.expr_lev++;
	i := P.ParseExpression(0);
	P.expr_lev--;
	P.Expect(Scanner.RBRACK);

	P.Ecart();
	return P.NewExpr(pos, Scanner.LBRACK, x, i);
}


func (P *Parser) ParseBinaryExpr(prec1 int) *AST.Expr

func (P *Parser) ParseCall(x0 *AST.Expr) *AST.Expr {
	P.Trace("Call");

	x := P.NewExpr(P.pos, Scanner.LPAREN, x0, nil);
	P.Expect(Scanner.LPAREN);
	if P.tok != Scanner.RPAREN {
		P.expr_lev++;
		var t *AST.Type;
		if x0.tok == Scanner.IDENT && (x0.obj.ident == "new" || x0.obj.ident == "make") {
			// heuristic: assume it's a new(T) or make(T, ...) call, try to parse a type
			t = P.TryType();
		}
		if t != nil {
			// we found a type
			x.y = AST.NewTypeExpr(t);
			if P.tok == Scanner.COMMA {
				pos := P.pos;
				P.Next();
				y := P.ParseExpressionList();
				// create list manually because NewExpr checks for type expressions
				z := AST.NewExpr(pos, Scanner.COMMA, nil, y);
				z.x = x.y;
				x.y = z;
			}
		} else {
			// normal argument list
			x.y = P.ParseExpressionList();
		}
		P.expr_lev--;
	}
	P.Expect(Scanner.RPAREN);

	P.Ecart();
	return x;
}


func (P *Parser) ParseCompositeElements() *AST.Expr {
	x := P.ParseExpression(0);
	if P.tok == Scanner.COMMA {
		pos := P.pos;
		P.Next();

		// first element determines mode
		singles := true;
		if x.tok == Scanner.COLON {
			singles = false;
		}

		var last *AST.Expr;
		for P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
			y := P.ParseExpression(0);

			if singles {
				if y.tok == Scanner.COLON {
					P.Error(y.x.pos, "single value expected; found pair");
				}
			} else {
				if y.tok != Scanner.COLON {
					P.Error(y.pos, "key:value pair expected; found single value");
				}
			}

			if last == nil {
				x = P.NewExpr(pos, Scanner.COMMA, x, y);
				last = x;
			} else {
				last.y = P.NewExpr(pos, Scanner.COMMA, last.y, y);
				last = last.y;
			}

			if P.tok == Scanner.COMMA {
				pos = P.pos;
				P.Next();
			} else {
				break;
			}

		}
	}
	return x;
}


func (P *Parser) ParseCompositeLit(t *AST.Type) *AST.Expr {
	P.Trace("CompositeLit");

	x := P.NewExpr(P.pos, Scanner.LBRACE, nil, nil);
	x.obj = AST.NewObject(t.pos, AST.TYPE, "");
	x.obj.typ = t;
	P.Expect(Scanner.LBRACE);
	if P.tok != Scanner.RBRACE {
		x.y = P.ParseCompositeElements();
	}
	P.Expect(Scanner.RBRACE);

	P.Ecart();
	return x;
}


func (P *Parser) ParsePrimaryExpr() *AST.Expr {
	P.Trace("PrimaryExpr");

	x := P.ParseOperand();
	for {
		switch P.tok {
		case Scanner.PERIOD: x = P.ParseSelectorOrTypeGuard(x);
		case Scanner.LBRACK: x = P.ParseIndex(x);
		case Scanner.LPAREN: x = P.ParseCall(x);
		case Scanner.LBRACE:
			// assume a composite literal only if x could be a type
			// and if we are not inside a control clause (expr_lev >= 0)
			// (composites inside control clauses must be parenthesized)
			var t *AST.Type;
			if P.expr_lev >= 0 {
				t = ExprType(x);
			}
			if t != nil {
				x = P.ParseCompositeLit(t);
			} else {
				goto exit;
			}
		default: goto exit;
		}
	}

exit:
	P.Ecart();
	return x;
}


func (P *Parser) ParseUnaryExpr() *AST.Expr {
	P.Trace("UnaryExpr");

	x := AST.BadExpr;
	switch P.tok {
	case Scanner.ADD, Scanner.SUB, Scanner.MUL, Scanner.NOT, Scanner.XOR, Scanner.ARROW, Scanner.AND:
		pos, tok := P.pos, P.tok;
		P.Next();
		y := P.ParseUnaryExpr();
		if tok == Scanner.MUL && y.tok == Scanner.TYPE {
			// pointer type
			t := AST.NewType(pos, AST.POINTER);
			t.elt = y.obj.typ;
			x = AST.NewTypeExpr(t);
		} else {
			x = P.NewExpr(pos, tok, nil, y);
		}

	default:
		x = P.ParsePrimaryExpr();
	}

	P.Ecart();
	return x;
}


func (P *Parser) ParseBinaryExpr(prec1 int) *AST.Expr {
	P.Trace("BinaryExpr");

	x := P.ParseUnaryExpr();
	for prec := Scanner.Precedence(P.tok); prec >= prec1; prec-- {
		for Scanner.Precedence(P.tok) == prec {
			pos, tok := P.pos, P.tok;
			P.Next();
			y := P.ParseBinaryExpr(prec + 1);
			x = P.NewExpr(pos, tok, x, y);
		}
	}

	P.Ecart();
	return x;
}


func (P *Parser) ParseExpression(prec int) *AST.Expr {
	P.Trace("Expression");
	indent := P.indent;

	if prec < 0 {
		panic("precedence must be >= 0");
	}
	x := P.NoType(P.ParseBinaryExpr(prec));

	if indent != P.indent {
		panic("imbalanced tracing code (Expression)");
	}
	P.Ecart();
	return x;
}


// ----------------------------------------------------------------------------
// Statements

func (P *Parser) ParseSimpleStat(range_ok bool) *AST.Stat {
	P.Trace("SimpleStat");

	s := AST.BadStat;
	x := P.ParseExpressionList();

	is_range := false;
	if range_ok && P.tok == Scanner.COLON {
		pos := P.pos;
		P.Next();
		y := P.ParseExpression(1);
		if x.Len() == 1 {
			x = P.NewExpr(pos, Scanner.COLON, x, y);
			is_range = true;
		} else {
			P.Error(pos, "expected initialization, found ':'");
		}
	}

	switch P.tok {
	case Scanner.COLON:
		// label declaration
		s = AST.NewStat(P.pos, Scanner.COLON);
		s.expr = x;
		if x.Len() != 1 {
			P.Error(x.pos, "illegal label declaration");
		}
		P.Next();  // consume ":"
		P.opt_semi = true;

	case
		Scanner.DEFINE, Scanner.ASSIGN, Scanner.ADD_ASSIGN,
		Scanner.SUB_ASSIGN, Scanner.MUL_ASSIGN, Scanner.QUO_ASSIGN,
		Scanner.REM_ASSIGN, Scanner.AND_ASSIGN, Scanner.OR_ASSIGN,
		Scanner.XOR_ASSIGN, Scanner.SHL_ASSIGN, Scanner.SHR_ASSIGN:
		// declaration/assignment
		pos, tok := P.pos, P.tok;
		P.Next();
		y := AST.BadExpr;
		if P.tok == Scanner.RANGE {
			range_pos := P.pos;
			P.Next();
			y = P.ParseExpression(1);
			y = P.NewExpr(range_pos, Scanner.RANGE, nil, y);
			if tok != Scanner.DEFINE && tok != Scanner.ASSIGN {
				P.Error(pos, "expected '=' or ':=', found '" + Scanner.TokenString(tok) + "'");
			}
		} else {
			y = P.ParseExpressionList();
			if is_range {
				P.Error(y.pos, "expected 'range', found expression");
			}
			if xl, yl := x.Len(), y.Len(); xl > 1 && yl > 1 && xl != yl {
				P.Error(x.pos, "arity of lhs doesn't match rhs");
			}
		}
		s = AST.NewStat(x.pos, Scanner.EXPRSTAT);
		s.expr = AST.NewExpr(pos, tok, x, y);

	case Scanner.RANGE:
		pos := P.pos;
		P.Next();
		y := P.ParseExpression(1);
		y = P.NewExpr(pos, Scanner.RANGE, nil, y);
		s = AST.NewStat(x.pos, Scanner.EXPRSTAT);
		s.expr = AST.NewExpr(pos, Scanner.DEFINE, x, y);

	default:
		var pos, tok int;
		if P.tok == Scanner.INC || P.tok == Scanner.DEC {
			pos, tok = P.pos, P.tok;
			P.Next();
		} else {
			pos, tok = x.pos, Scanner.EXPRSTAT;
		}
		s = AST.NewStat(pos, tok);
		s.expr = x;
		if x.Len() != 1 {
			P.Error(x.pos, "only one expression allowed");
		}
	}

	P.Ecart();
	return s;
}


func (P *Parser) ParseGoStat() *AST.Stat {
	P.Trace("GoStat");

	s := AST.NewStat(P.pos, Scanner.GO);
	P.Expect(Scanner.GO);
	s.expr = P.ParseExpression(1);

	P.Ecart();
	return s;
}


func (P *Parser) ParseReturnStat() *AST.Stat {
	P.Trace("ReturnStat");

	s := AST.NewStat(P.pos, Scanner.RETURN);
	P.Expect(Scanner.RETURN);
	if P.tok != Scanner.SEMICOLON && P.tok != Scanner.RBRACE {
		s.expr = P.ParseExpressionList();
	}

	P.Ecart();
	return s;
}


func (P *Parser) ParseControlFlowStat(tok int) *AST.Stat {
	P.Trace("ControlFlowStat");

	s := AST.NewStat(P.pos, tok);
	P.Expect(tok);
	if tok != Scanner.FALLTHROUGH && P.tok == Scanner.IDENT {
		s.expr = P.ParseIdent(P.top_scope);
	}

	P.Ecart();
	return s;
}


func (P *Parser) ParseControlClause(keyword int) *AST.Stat {
	P.Trace("ControlClause");

	s := AST.NewStat(P.pos, keyword);
	P.Expect(keyword);
	if P.tok != Scanner.LBRACE {
		prev_lev := P.expr_lev;
		P.expr_lev = -1;
		if P.tok != Scanner.SEMICOLON {
			s.init = P.ParseSimpleStat(keyword == Scanner.FOR);
			// TODO check for range clause and exit if found
		}
		if P.tok == Scanner.SEMICOLON {
			P.Next();
			if P.tok != Scanner.SEMICOLON && P.tok != Scanner.LBRACE {
				s.expr = P.ParseExpression(1);
			}
			if keyword == Scanner.FOR {
				P.Expect(Scanner.SEMICOLON);
				if P.tok != Scanner.LBRACE {
					s.post = P.ParseSimpleStat(false);
				}
			}
		} else {
			if s.init != nil {  // guard in case of errors
				s.expr, s.init = s.init.expr, nil;
			}
		}
		P.expr_lev = prev_lev;
	}

	P.Ecart();
	return s;
}


func (P *Parser) ParseIfStat() *AST.Stat {
	P.Trace("IfStat");

	P.OpenScope();
	s := P.ParseControlClause(Scanner.IF);
	s.block, s.end = P.ParseBlock();
	if P.tok == Scanner.ELSE {
		P.Next();
		s1 := AST.BadStat;
		if P.tok == Scanner.IF {
			s1 = P.ParseIfStat();
		} else if P.sixg {
			s1 = P.ParseStatement();
			if s1 != nil {
				// not the empty statement
				if s1.tok != Scanner.LBRACE {
					// wrap in a block if we don't have one
					b := AST.NewStat(P.pos, Scanner.LBRACE);
					b.block = array.New(0);
					b.block.Push(s1);
					s1 = b;
				}
				s.post = s1;
			}
		} else {
			s1 = AST.NewStat(P.pos, Scanner.LBRACE);
			s1.block, s1.end = P.ParseBlock();
		}
		s.post = s1;
	}
	P.CloseScope();

	P.Ecart();
	return s;
}


func (P *Parser) ParseForStat() *AST.Stat {
	P.Trace("ForStat");

	P.OpenScope();
	s := P.ParseControlClause(Scanner.FOR);
	s.block, s.end = P.ParseBlock();
	P.CloseScope();

	P.Ecart();
	return s;
}


func (P *Parser) ParseCase() *AST.Stat {
	P.Trace("Case");

	s := AST.NewStat(P.pos, P.tok);
	if P.tok == Scanner.CASE {
		P.Next();
		s.expr = P.ParseExpressionList();
	} else {
		P.Expect(Scanner.DEFAULT);
	}
	P.Expect(Scanner.COLON);

	P.Ecart();
	return s;
}


func (P *Parser) ParseCaseClause() *AST.Stat {
	P.Trace("CaseClause");

	s := P.ParseCase();
	if P.tok != Scanner.CASE && P.tok != Scanner.DEFAULT && P.tok != Scanner.RBRACE {
		s.block = P.ParseStatementList();
	}

	P.Ecart();
	return s;
}


func (P *Parser) ParseSwitchStat() *AST.Stat {
	P.Trace("SwitchStat");

	P.OpenScope();
	s := P.ParseControlClause(Scanner.SWITCH);
	s.block = array.New(0);
	P.Expect(Scanner.LBRACE);
	for P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
		s.block.Push(P.ParseCaseClause());
	}
	s.end = P.pos;
	P.Expect(Scanner.RBRACE);
	P.opt_semi = true;
	P.CloseScope();

	P.Ecart();
	return s;
}


func (P *Parser) ParseCommCase() *AST.Stat {
	P.Trace("CommCase");

	s := AST.NewStat(P.pos, P.tok);
	if P.tok == Scanner.CASE {
		P.Next();
		x := P.ParseExpression(1);
		if P.tok == Scanner.ASSIGN || P.tok == Scanner.DEFINE {
			pos, tok := P.pos, P.tok;
			P.Next();
			if P.tok == Scanner.ARROW {
				y := P.ParseExpression(1);
				x = AST.NewExpr(pos, tok, x, y);
			} else {
				P.Expect(Scanner.ARROW);  // use Expect() error handling
			}
		}
		s.expr = x;
	} else {
		P.Expect(Scanner.DEFAULT);
	}
	P.Expect(Scanner.COLON);

	P.Ecart();
	return s;
}


func (P *Parser) ParseCommClause() *AST.Stat {
	P.Trace("CommClause");

	s := P.ParseCommCase();
	if P.tok != Scanner.CASE && P.tok != Scanner.DEFAULT && P.tok != Scanner.RBRACE {
		s.block = P.ParseStatementList();
	}

	P.Ecart();
	return s;
}


func (P *Parser) ParseSelectStat() *AST.Stat {
	P.Trace("SelectStat");

	s := AST.NewStat(P.pos, Scanner.SELECT);
	s.block = array.New(0);
	P.Expect(Scanner.SELECT);
	P.Expect(Scanner.LBRACE);
	for P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
		s.block.Push(P.ParseCommClause());
	}
	P.Expect(Scanner.RBRACE);
	P.opt_semi = true;

	P.Ecart();
	return s;
}


func (P *Parser) ParseRangeStat() *AST.Stat {
	P.Trace("RangeStat");

	s := AST.NewStat(P.pos, Scanner.RANGE);
	P.Expect(Scanner.RANGE);
	P.ParseIdentList();
	P.Expect(Scanner.DEFINE);
	s.expr = P.ParseExpression(1);
	s.block, s.end = P.ParseBlock();

	P.Ecart();
	return s;
}


func (P *Parser) ParseStatement() *AST.Stat {
	P.Trace("Statement");
	indent := P.indent;

	s := AST.BadStat;
	switch P.tok {
	case Scanner.CONST, Scanner.TYPE, Scanner.VAR:
		s = AST.NewStat(P.pos, P.tok);
		s.decl = P.ParseDeclaration();
	case Scanner.FUNC:
		// for now we do not allow local function declarations,
		// instead we assume this starts a function literal
		fallthrough;
	case
		// only the tokens that are legal top-level expression starts
		Scanner.IDENT, Scanner.INT, Scanner.FLOAT, Scanner.STRING, Scanner.LPAREN,  // operand
		Scanner.LBRACK, Scanner.STRUCT,  // composite type
		Scanner.MUL, Scanner.AND, Scanner.ARROW:  // unary
		s = P.ParseSimpleStat(false);
	case Scanner.GO:
		s = P.ParseGoStat();
	case Scanner.RETURN:
		s = P.ParseReturnStat();
	case Scanner.BREAK, Scanner.CONTINUE, Scanner.GOTO, Scanner.FALLTHROUGH:
		s = P.ParseControlFlowStat(P.tok);
	case Scanner.LBRACE:
		s = AST.NewStat(P.pos, Scanner.LBRACE);
		s.block, s.end = P.ParseBlock();
	case Scanner.IF:
		s = P.ParseIfStat();
	case Scanner.FOR:
		s = P.ParseForStat();
	case Scanner.SWITCH:
		s = P.ParseSwitchStat();
	case Scanner.RANGE:
		s = P.ParseRangeStat();
	case Scanner.SELECT:
		s = P.ParseSelectStat();
	default:
		// empty statement
		s = nil;
	}

	if indent != P.indent {
		panic("imbalanced tracing code (Statement)");
	}
	P.Ecart();
	return s;
}


// ----------------------------------------------------------------------------
// Declarations

func (P *Parser) ParseImportSpec(pos int) *AST.Decl {
	P.Trace("ImportSpec");

	d := AST.NewDecl(pos, Scanner.IMPORT, false);
	if P.tok == Scanner.PERIOD {
		P.Error(P.pos, `"import ." not yet handled properly`);
		P.Next();
	} else if P.tok == Scanner.IDENT {
		d.ident = P.ParseIdent(nil);
	}

	if P.tok == Scanner.STRING {
		// TODO eventually the scanner should strip the quotes
		val := AST.NewObject(P.pos, AST.NONE, P.val);
		d.val = AST.NewLit(Scanner.STRING, val);
		P.Next();
	} else {
		P.Expect(Scanner.STRING);  // use Expect() error handling
	}

	if d.ident != nil {
		P.Declare(d.ident, AST.PACKAGE);
	}

	P.Ecart();
	return d;
}


func (P *Parser) ParseConstSpec(exported bool, pos int) *AST.Decl {
	P.Trace("ConstSpec");

	d := AST.NewDecl(pos, Scanner.CONST, exported);
	d.ident = P.ParseIdentList();
	d.typ = P.TryType();
	if P.tok == Scanner.ASSIGN {
		P.Next();
		d.val = P.ParseExpressionList();
	}
	
	P.Declare(d.ident, AST.CONST);

	P.Ecart();
	return d;
}


func (P *Parser) ParseTypeSpec(exported bool, pos int) *AST.Decl {
	P.Trace("TypeSpec");

	d := AST.NewDecl(pos, Scanner.TYPE, exported);
	d.ident = P.ParseIdent(nil);
	d.typ = P.ParseType();
	P.opt_semi = true;

	P.Ecart();
	return d;
}


func (P *Parser) ParseVarSpec(exported bool, pos int) *AST.Decl {
	P.Trace("VarSpec");

	d := AST.NewDecl(pos, Scanner.VAR, exported);
	d.ident = P.ParseIdentList();
	if P.tok == Scanner.ASSIGN {
		P.Next();
		d.val = P.ParseExpressionList();
	} else {
		d.typ = P.ParseVarType();
		if P.tok == Scanner.ASSIGN {
			P.Next();
			d.val = P.ParseExpressionList();
		}
	}

	P.Declare(d.ident, AST.VAR);

	P.Ecart();
	return d;
}


// TODO replace this by using function pointers derived from methods
func (P *Parser) ParseSpec(exported bool, pos int, keyword int) *AST.Decl {
	switch keyword {
	case Scanner.IMPORT: return P.ParseImportSpec(pos);
	case Scanner.CONST: return P.ParseConstSpec(exported, pos);
	case Scanner.TYPE: return P.ParseTypeSpec(exported, pos);
	case Scanner.VAR: return P.ParseVarSpec(exported, pos);
	}
	panic("UNREACHABLE");
	return nil;
}


func (P *Parser) ParseDecl(exported bool, keyword int) *AST.Decl {
	P.Trace("Decl");

	d := AST.BadDecl;
	pos := P.pos;
	P.Expect(keyword);
	if P.tok == Scanner.LPAREN {
		P.Next();
		d = AST.NewDecl(pos, keyword, exported);
		d.list = array.New(0);
		for P.tok != Scanner.RPAREN && P.tok != Scanner.EOF {
			d.list.Push(P.ParseSpec(exported, pos, keyword));
			if P.tok == Scanner.SEMICOLON {
				P.Next();
			} else {
				break;
			}
		}
		d.end = P.pos;
		P.Expect(Scanner.RPAREN);
		P.opt_semi = true;

	} else {
		d = P.ParseSpec(exported, pos, keyword);
	}

	P.Ecart();
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

func (P *Parser) ParseFunctionDecl(exported bool) *AST.Decl {
	P.Trace("FunctionDecl");

	d := AST.NewDecl(P.pos, Scanner.FUNC, exported);
	P.Expect(Scanner.FUNC);

	var recv *AST.Type;
	if P.tok == Scanner.LPAREN {
		pos := P.pos;
		recv = P.ParseParameters(true);
		if recv.nfields() != 1 {
			P.Error(pos, "must have exactly one receiver");
		}
	}

	d.ident = P.ParseIdent(nil);
	d.typ = P.ParseFunctionType();
	d.typ.key = recv;

	if P.tok == Scanner.LBRACE {
		P.scope_lev++;
		d.list, d.end = P.ParseBlock();
		P.scope_lev--;
	}

	P.Ecart();
	return d;
}


func (P *Parser) ParseExportDecl() *AST.Decl {
	P.Trace("ExportDecl");

	d := AST.NewDecl(P.pos, Scanner.EXPORT, false);
	d.ident = P.ParseIdentList();

	P.Ecart();
	return d;
}


func (P *Parser) ParseDeclaration() *AST.Decl {
	P.Trace("Declaration");
	indent := P.indent;

	d := AST.BadDecl;
	exported := false;
	// TODO don't use bool flag for export
	if P.tok == Scanner.EXPORT || P.tok == Scanner.PACKAGE {
		if P.scope_lev == 0 {
			exported = true;
		} else {
			P.Error(P.pos, "local declarations cannot be exported");
		}
		P.Next();
	}

	switch P.tok {
	case Scanner.CONST, Scanner.TYPE, Scanner.VAR:
		d = P.ParseDecl(exported, P.tok);
	case Scanner.FUNC:
		d = P.ParseFunctionDecl(exported);
	case Scanner.EXPORT:
		if exported {
			P.Error(P.pos, "cannot mark export declaration for export");
		}
		P.Next();
		d = P.ParseExportDecl();
	default:
		if exported && (P.tok == Scanner.IDENT || P.tok == Scanner.LPAREN) {
			d = P.ParseExportDecl();
		} else {
			P.Error(P.pos, "declaration expected");
			P.Next();  // make progress
		}
	}

	if indent != P.indent {
		panic("imbalanced tracing code (Declaration)");
	}
	P.Ecart();
	return d;
}


// ----------------------------------------------------------------------------
// Program

func (P *Parser) ParseProgram() *AST.Program {
	P.Trace("Program");

	P.OpenScope();
	p := AST.NewProgram(P.pos);
	P.Expect(Scanner.PACKAGE);
	p.ident = P.ParseIdent(nil);

	// package body
	{	P.OpenScope();
		p.decls = array.New(0);
		for P.tok == Scanner.IMPORT {
			p.decls.Push(P.ParseDecl(false, Scanner.IMPORT));
			P.OptSemicolon();
		}
		if !P.deps {
			for P.tok != Scanner.EOF {
				p.decls.Push(P.ParseDeclaration());
				P.OptSemicolon();
			}
		}
		P.CloseScope();
	}

	p.comments = P.comments;
	P.CloseScope();

	P.Ecart();
	return p;
}
