// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Parser

import (
	"array";
	Scanner "scanner";
	AST "ast";
)


type Parser struct {
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
		P.tok, P.pos, P.val = t.Tok, t.Pos, t.Val;
	}
	P.opt_semi = false;

	if P.verbose {
		P.PrintIndent();
		s := Scanner.TokenString(P.tok);
		// rewrite "{" and "}" so we don't screw up double-click selection
		// in terminal window (we print scopes using the same characters)
		switch s {
		case "{": s = "LBRACE";
		case "}": s = "RBRACE";
		}
		print("[", P.pos, "] ", s, "\n");
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
	P.top_scope = P.top_scope.Parent;
}


func (P *Parser) DeclareInScope(scope *AST.Scope, x *AST.Expr, kind int) {
	if P.scope_lev < 0 {
		panic("cannot declare objects in other packages");
	}
	if x.Tok != Scanner.ILLEGAL {  // ignore bad exprs
		assert(x.Tok == Scanner.IDENT);
		obj := x.Obj;
		obj.Kind = kind;
		obj.Pnolev = P.scope_lev;
		if scope.LookupLocal(obj.Ident) == nil {
			scope.Insert(obj);
		} else {
			P.Error(obj.Pos, `"` + obj.Ident + `" is declared already`);
		}
	}
}


// Declare a comma-separated list of idents or a single ident.
func (P *Parser) Declare(p *AST.Expr, kind int) {
	for p.Tok == Scanner.COMMA {
		P.DeclareInScope(P.top_scope, p.X, kind);
		p = p.Y;
	}
	P.DeclareInScope(P.top_scope, p, kind);
}


// ----------------------------------------------------------------------------
// AST support

func exprType(x *AST.Expr) *AST.Type {
	var t *AST.Type;
	if x.Tok == Scanner.TYPE {
		t = x.Typ;
	} else if x.Tok == Scanner.IDENT {
		// assume a type name
		t = AST.NewType(x.Pos, AST.TYPENAME);
		t.Expr = x;
	} else if x.Tok == Scanner.PERIOD && x.Y != nil && exprType(x.X) != nil {
		// possibly a qualified (type) identifier
		t = AST.NewType(x.Pos, AST.TYPENAME);
		t.Expr = x;
	}
	return t;
}


func (P *Parser) NoType(x *AST.Expr) *AST.Expr {
	if x != nil && x.Tok == Scanner.TYPE {
		P.Error(x.Pos, "expected expression, found type");
		val := AST.NewObject(x.Pos, AST.NONE, "0");
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
			assert(obj.Kind != AST.NONE);
		}
		x = AST.NewLit(Scanner.IDENT, obj);
		x.Pos = P.pos;  // override obj.pos (incorrect if object was looked up!)
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
			last.Y = P.NewExpr(pos, Scanner.COMMA, last.Y, y);
			last = last.Y;
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
	t.Expr = P.ParseQualifiedIdent();

	P.Ecart();
	return t;
}


func (P *Parser) ParseArrayType() *AST.Type {
	P.Trace("ArrayType");

	t := AST.NewType(P.pos, AST.ARRAY);
	P.Expect(Scanner.LBRACK);
	if P.tok == Scanner.ELLIPSIS {
		t.Expr = P.NewExpr(P.pos, Scanner.ELLIPSIS, nil, nil);
		P.Next();
	} else if P.tok != Scanner.RBRACK {
		t.Expr = P.ParseExpression(1);
	}
	P.Expect(Scanner.RBRACK);
	t.Elt = P.ParseType();

	P.Ecart();
	return t;
}


func (P *Parser) ParseChannelType() *AST.Type {
	P.Trace("ChannelType");

	t := AST.NewType(P.pos, AST.CHANNEL);
	t.Mode = AST.FULL;
	if P.tok == Scanner.CHAN {
		P.Next();
		if P.tok == Scanner.ARROW {
			P.Next();
			t.Mode = AST.SEND;
		}
	} else {
		P.Expect(Scanner.ARROW);
		P.Expect(Scanner.CHAN);
		t.Mode = AST.RECV;
	}
	t.Elt = P.ParseVarType();

	P.Ecart();
	return t;
}


func (P *Parser) ParseVar(expect_ident bool) *AST.Type {
	t := AST.BadType;
	if expect_ident {
		x := P.ParseIdent(nil);
		t = AST.NewType(x.Pos, AST.TYPENAME);
		t.Expr = x;
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
			if t.Form == AST.TYPENAME && t.Expr.Tok == Scanner.IDENT {
				list.Set(i, t.Expr);
			} else {
				list.Set(i, AST.BadExpr);
				P.Error(t.Pos, "identifier expected");
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
		t.List = P.ParseParameterList(ellipsis_ok);
	}
	t.End = P.pos;
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


func (P *Parser) ParseResult(ftyp *AST.Type) *AST.Type {
	P.Trace("Result");

	var t *AST.Type;
	if P.tok == Scanner.LPAREN {
		t = P.ParseParameters(false);
	} else {
		typ := P.TryType();
		if typ != nil {
			t = AST.NewType(P.pos, AST.STRUCT);
			t.List = array.New(0);
			t.List.Push(AST.NewTypeExpr(typ));
			t.End = P.pos;
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
	t.Scope = P.top_scope;
	t.List = P.ParseParameters(true).List;  // TODO find better solution
	t.End = P.pos;
	t.Elt = P.ParseResult(t);

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

		t.List = array.New(0);
		for P.tok == Scanner.IDENT {
			P.ParseMethodSpec(t.List);
			if P.tok != Scanner.RBRACE {
				P.Expect(Scanner.SEMICOLON);
			}
		}
		t.End = P.pos;

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
	t.Key = P.ParseVarType();
	P.Expect(Scanner.RBRACK);
	t.Elt = P.ParseVarType();

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

		t.List = array.New(0);
		t.Scope = AST.NewScope(nil);
		for P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
			P.ParseVarList(t.List, false);
			if P.tok == Scanner.STRING {
				// ParseOperand takes care of string concatenation
				t.List.Push(P.ParseOperand());
			}
			if P.tok == Scanner.SEMICOLON {
				P.Next();
			} else {
				break;
			}
		}
		P.OptSemicolon();
		t.End = P.pos;

		P.Expect(Scanner.RBRACE);
		
		// enter fields into struct scope
		for i, n := 0, t.List.Len(); i < n; i++ {
			x := t.List.At(i).(*AST.Expr);
			if x.Tok == Scanner.IDENT {
				P.DeclareInScope(t.Scope, x, AST.FIELD);
			}
		}
	}

	P.Ecart();
	return t;
}


func (P *Parser) ParsePointerType() *AST.Type {
	P.Trace("PointerType");

	t := AST.NewType(P.pos, AST.POINTER);
	P.Expect(Scanner.MUL);
	t.Elt = P.ParseType();

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

func (P *Parser) ParseStatementList(list *array.Array) {
	P.Trace("StatementList");

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
}


func (P *Parser) ParseBlock(ftyp *AST.Type, tok int) *AST.Block {
	P.Trace("Block");

	b := AST.NewBlock(P.pos, tok);
	P.Expect(tok);

	P.OpenScope();
	// enter recv and parameters into function scope
	if ftyp != nil {
		assert(ftyp.Form == AST.FUNCTION);
		if ftyp.Key != nil {
		}
		if ftyp.List != nil {
			for i, n := 0, ftyp.List.Len(); i < n; i++ {
				x := ftyp.List.At(i).(*AST.Expr);
				if x.Tok == Scanner.IDENT {
					P.DeclareInScope(P.top_scope, x, AST.VAR);
				}
			}
		}
	}

	P.ParseStatementList(b.List);
	P.CloseScope();

	if tok == Scanner.LBRACE {
		b.End = P.pos;
		P.Expect(Scanner.RBRACE);
		P.opt_semi = true;
	}

	P.Ecart();
	return b;
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
			x.Y = P.NewExpr(pos, Scanner.COMMA, x.Y, y);
		}
	}

	P.Ecart();
	return x;
}


func (P *Parser) ParseFunctionLit() *AST.Expr {
	P.Trace("FunctionLit");

	f := AST.NewObject(P.pos, AST.FUNC, "");
	P.Expect(Scanner.FUNC);
	f.Typ = P.ParseFunctionType();
	P.expr_lev++;
	P.scope_lev++;
	f.Body = P.ParseBlock(f.Typ, Scanner.LBRACE);
	P.scope_lev--;
	P.expr_lev--;

	P.Ecart();
	return AST.NewLit(Scanner.FUNC, f);
}


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
		if x.Tok == Scanner.STRING {
			// TODO should remember the list instead of
			//      concatenate the strings here
			for ; P.tok == Scanner.STRING; P.Next() {
				x.Obj.Ident += P.val;
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
		// TODO should always guarantee x.Typ != nil
		var scope *AST.Scope;
		if x.Typ != nil {
			scope = x.Typ.Scope;
		}
		x.Y = P.ParseIdent(scope);
		x.Typ = x.Y.Obj.Typ;

	} else {
		P.Expect(Scanner.LPAREN);
		x.Y = AST.NewTypeExpr(P.ParseType());
		x.Typ = x.Y.Typ;
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
		if x0.Tok == Scanner.IDENT && (x0.Obj.Ident == "new" || x0.Obj.Ident == "make") {
			// heuristic: assume it's a new(T) or make(T, ...) call, try to parse a type
			t = P.TryType();
		}
		if t != nil {
			// we found a type
			x.Y = AST.NewTypeExpr(t);
			if P.tok == Scanner.COMMA {
				pos := P.pos;
				P.Next();
				y := P.ParseExpressionList();
				// create list manually because NewExpr checks for type expressions
				z := AST.NewExpr(pos, Scanner.COMMA, nil, y);
				z.X = x.Y;
				x.Y = z;
			}
		} else {
			// normal argument list
			x.Y = P.ParseExpressionList();
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
		if x.Tok == Scanner.COLON {
			singles = false;
		}

		var last *AST.Expr;
		for P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
			y := P.ParseExpression(0);

			if singles {
				if y.Tok == Scanner.COLON {
					P.Error(y.X.Pos, "single value expected; found pair");
				}
			} else {
				if y.Tok != Scanner.COLON {
					P.Error(y.Pos, "key:value pair expected; found single value");
				}
			}

			if last == nil {
				x = P.NewExpr(pos, Scanner.COMMA, x, y);
				last = x;
			} else {
				last.Y = P.NewExpr(pos, Scanner.COMMA, last.Y, y);
				last = last.Y;
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
	x.Obj = AST.NewObject(t.Pos, AST.TYPE, "");
	x.Obj.Typ = t;
	P.Expect(Scanner.LBRACE);
	if P.tok != Scanner.RBRACE {
		x.Y = P.ParseCompositeElements();
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
				t = exprType(x);
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
		if tok == Scanner.MUL && y.Tok == Scanner.TYPE {
			// pointer type
			t := AST.NewType(pos, AST.POINTER);
			t.Elt = y.Obj.Typ;
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
		s.Expr = x;
		if x.Len() != 1 {
			P.Error(x.Pos, "illegal label declaration");
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
				P.Error(y.Pos, "expected 'range', found expression");
			}
			if xl, yl := x.Len(), y.Len(); xl > 1 && yl > 1 && xl != yl {
				P.Error(x.Pos, "arity of lhs doesn't match rhs");
			}
		}
		s = AST.NewStat(x.Pos, Scanner.EXPRSTAT);
		s.Expr = AST.NewExpr(pos, tok, x, y);

	case Scanner.RANGE:
		pos := P.pos;
		P.Next();
		y := P.ParseExpression(1);
		y = P.NewExpr(pos, Scanner.RANGE, nil, y);
		s = AST.NewStat(x.Pos, Scanner.EXPRSTAT);
		s.Expr = AST.NewExpr(pos, Scanner.DEFINE, x, y);

	default:
		var pos, tok int;
		if P.tok == Scanner.INC || P.tok == Scanner.DEC {
			pos, tok = P.pos, P.tok;
			P.Next();
		} else {
			pos, tok = x.Pos, Scanner.EXPRSTAT;
		}
		s = AST.NewStat(pos, tok);
		s.Expr = x;
		if x.Len() != 1 {
			P.Error(x.Pos, "only one expression allowed");
		}
	}

	P.Ecart();
	return s;
}


func (P *Parser) ParseGoStat() *AST.Stat {
	P.Trace("GoStat");

	s := AST.NewStat(P.pos, Scanner.GO);
	P.Expect(Scanner.GO);
	s.Expr = P.ParseExpression(1);

	P.Ecart();
	return s;
}


func (P *Parser) ParseReturnStat() *AST.Stat {
	P.Trace("ReturnStat");

	s := AST.NewStat(P.pos, Scanner.RETURN);
	P.Expect(Scanner.RETURN);
	if P.tok != Scanner.SEMICOLON && P.tok != Scanner.RBRACE {
		s.Expr = P.ParseExpressionList();
	}

	P.Ecart();
	return s;
}


func (P *Parser) ParseControlFlowStat(tok int) *AST.Stat {
	P.Trace("ControlFlowStat");

	s := AST.NewStat(P.pos, tok);
	P.Expect(tok);
	if tok != Scanner.FALLTHROUGH && P.tok == Scanner.IDENT {
		s.Expr = P.ParseIdent(P.top_scope);
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
			s.Init = P.ParseSimpleStat(keyword == Scanner.FOR);
			// TODO check for range clause and exit if found
		}
		if P.tok == Scanner.SEMICOLON {
			P.Next();
			if P.tok != Scanner.SEMICOLON && P.tok != Scanner.LBRACE {
				s.Expr = P.ParseExpression(1);
			}
			if keyword == Scanner.FOR {
				P.Expect(Scanner.SEMICOLON);
				if P.tok != Scanner.LBRACE {
					s.Post = P.ParseSimpleStat(false);
				}
			}
		} else {
			if s.Init != nil {  // guard in case of errors
				s.Expr, s.Init = s.Init.Expr, nil;
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
	s.Body = P.ParseBlock(nil, Scanner.LBRACE);
	if P.tok == Scanner.ELSE {
		P.Next();
		s1 := AST.BadStat;
		if P.tok == Scanner.IF || P.tok == Scanner.LBRACE {
			s1 = P.ParseStatement();
		} else if P.sixg {
			s1 = P.ParseStatement();
			if s1 != nil {
				// not the empty statement
				assert(s1.Tok != Scanner.LBRACE);
				// wrap in a block since we don't have one
				b := AST.NewStat(s1.Pos, Scanner.LBRACE);
				b.Body = AST.NewBlock(s1.Pos, Scanner.LBRACE);
				b.Body.List.Push(s1);
				s1 = b;
			}
		} else {
			P.Error(P.pos, "'if' or '{' expected - illegal 'else' branch");
		}
		s.Post = s1;
	}
	P.CloseScope();

	P.Ecart();
	return s;
}


func (P *Parser) ParseForStat() *AST.Stat {
	P.Trace("ForStat");

	P.OpenScope();
	s := P.ParseControlClause(Scanner.FOR);
	s.Body = P.ParseBlock(nil, Scanner.LBRACE);
	P.CloseScope();

	P.Ecart();
	return s;
}


func (P *Parser) ParseSwitchCase() *AST.Stat {
	P.Trace("SwitchCase");

	s := AST.NewStat(P.pos, P.tok);
	if P.tok == Scanner.CASE {
		P.Next();
		s.Expr = P.ParseExpressionList();
	} else {
		P.Expect(Scanner.DEFAULT);
	}

	P.Ecart();
	return s;
}


func (P *Parser) ParseCaseClause() *AST.Stat {
	P.Trace("CaseClause");

	s := P.ParseSwitchCase();
	s.Body = P.ParseBlock(nil, Scanner.COLON);

	P.Ecart();
	return s;
}


func (P *Parser) ParseSwitchStat() *AST.Stat {
	P.Trace("SwitchStat");

	P.OpenScope();
	s := P.ParseControlClause(Scanner.SWITCH);
	b := AST.NewBlock(P.pos, Scanner.LBRACE);
	P.Expect(Scanner.LBRACE);
	for P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
		b.List.Push(P.ParseCaseClause());
	}
	b.End = P.pos;
	P.Expect(Scanner.RBRACE);
	P.opt_semi = true;
	P.CloseScope();
	s.Body = b;

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
		s.Expr = x;
	} else {
		P.Expect(Scanner.DEFAULT);
	}

	P.Ecart();
	return s;
}


func (P *Parser) ParseCommClause() *AST.Stat {
	P.Trace("CommClause");

	s := P.ParseCommCase();
	s.Body = P.ParseBlock(nil, Scanner.COLON);

	P.Ecart();
	return s;
}


func (P *Parser) ParseSelectStat() *AST.Stat {
	P.Trace("SelectStat");

	s := AST.NewStat(P.pos, Scanner.SELECT);
	P.Expect(Scanner.SELECT);
	b := AST.NewBlock(P.pos, Scanner.LBRACE);
	P.Expect(Scanner.LBRACE);
	for P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
		b.List.Push(P.ParseCommClause());
	}
	b.End = P.pos;
	P.Expect(Scanner.RBRACE);
	P.opt_semi = true;
	s.Body = b;

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
		s.Decl = P.ParseDeclaration();
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
		s.Body = P.ParseBlock(nil, Scanner.LBRACE);
	case Scanner.IF:
		s = P.ParseIfStat();
	case Scanner.FOR:
		s = P.ParseForStat();
	case Scanner.SWITCH:
		s = P.ParseSwitchStat();
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

func (P *Parser) ParseImportSpec(d *AST.Decl) {
	P.Trace("ImportSpec");

	if P.tok == Scanner.PERIOD {
		P.Error(P.pos, `"import ." not yet handled properly`);
		P.Next();
	} else if P.tok == Scanner.IDENT {
		d.Ident = P.ParseIdent(nil);
	}

	if P.tok == Scanner.STRING {
		// TODO eventually the scanner should strip the quotes
		val := AST.NewObject(P.pos, AST.NONE, P.val);
		d.Val = AST.NewLit(Scanner.STRING, val);
		P.Next();
	} else {
		P.Expect(Scanner.STRING);  // use Expect() error handling
	}

	if d.Ident != nil {
		P.Declare(d.Ident, AST.PACKAGE);
	}

	P.Ecart();
}


func (P *Parser) ParseConstSpec(d *AST.Decl) {
	P.Trace("ConstSpec");

	d.Ident = P.ParseIdentList();
	d.Typ = P.TryType();
	if P.tok == Scanner.ASSIGN {
		P.Next();
		d.Val = P.ParseExpressionList();
	}

	P.Declare(d.Ident, AST.CONST);

	P.Ecart();
}


func (P *Parser) ParseTypeSpec(d *AST.Decl) {
	P.Trace("TypeSpec");

	d.Ident = P.ParseIdent(nil);
	d.Typ = P.ParseType();
	P.opt_semi = true;

	P.Ecart();
}


func (P *Parser) ParseVarSpec(d *AST.Decl) {
	P.Trace("VarSpec");

	d.Ident = P.ParseIdentList();
	if P.tok == Scanner.ASSIGN {
		P.Next();
		d.Val = P.ParseExpressionList();
	} else {
		d.Typ = P.ParseVarType();
		if P.tok == Scanner.ASSIGN {
			P.Next();
			d.Val = P.ParseExpressionList();
		}
	}

	P.Declare(d.Ident, AST.VAR);

	P.Ecart();
}


func (P *Parser) ParseSpec(d *AST.Decl) {
	switch d.Tok {
	case Scanner.IMPORT: P.ParseImportSpec(d);
	case Scanner.CONST: P.ParseConstSpec(d);
	case Scanner.TYPE: P.ParseTypeSpec(d);
	case Scanner.VAR: P.ParseVarSpec(d);
	default: unreachable();
	}
	
	// semantic checks
	if d.Tok == Scanner.IMPORT {
		// TODO
	} else {
		if d.Typ != nil {
			// apply type to all variables
		}
		if d.Val != nil {
			// initialization/assignment
			llen := d.Ident.Len();
			rlen := d.Val.Len();
			if llen == rlen {
				// TODO
			} else if rlen == 1 {
				// TODO
			} else {
				if llen < rlen {
					P.Error(d.Val.At(llen).Pos, "more expressions than variables");
				} else {
					P.Error(d.Ident.At(rlen).Pos, "more variables than expressions");
				}
			}
		} else {
			// TODO
		}
	}
}


func (P *Parser) ParseDecl(keyword int) *AST.Decl {
	P.Trace("Decl");

	d := AST.NewDecl(P.pos, keyword);
	P.Expect(keyword);
	if P.tok == Scanner.LPAREN {
		P.Next();
		d.List = array.New(0);
		for P.tok != Scanner.RPAREN && P.tok != Scanner.EOF {
			d1 := AST.NewDecl(P.pos, keyword);
			P.ParseSpec(d1);
			d.List.Push(d1);
			if P.tok == Scanner.SEMICOLON {
				P.Next();
			} else {
				break;
			}
		}
		d.End = P.pos;
		P.Expect(Scanner.RPAREN);
		P.opt_semi = true;

	} else {
		P.ParseSpec(d);
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

func (P *Parser) ParseFunctionDecl() *AST.Decl {
	P.Trace("FunctionDecl");

	d := AST.NewDecl(P.pos, Scanner.FUNC);
	P.Expect(Scanner.FUNC);

	var recv *AST.Type;
	if P.tok == Scanner.LPAREN {
		pos := P.pos;
		recv = P.ParseParameters(true);
		if recv.Nfields() != 1 {
			P.Error(pos, "must have exactly one receiver");
		}
	}

	d.Ident = P.ParseIdent(nil);
	d.Typ = P.ParseFunctionType();
	d.Typ.Key = recv;

	if P.tok == Scanner.LBRACE {
		f := AST.NewObject(d.Pos, AST.FUNC, d.Ident.Obj.Ident);
		f.Typ = d.Typ;
		f.Body = P.ParseBlock(d.Typ, Scanner.LBRACE);
		d.Val = AST.NewLit(Scanner.FUNC, f);
	}

	P.Ecart();
	return d;
}


func (P *Parser) ParseDeclaration() *AST.Decl {
	P.Trace("Declaration");
	indent := P.indent;

	d := AST.BadDecl;

	switch P.tok {
	case Scanner.CONST, Scanner.TYPE, Scanner.VAR:
		d = P.ParseDecl(P.tok);
	case Scanner.FUNC:
		d = P.ParseFunctionDecl();
	default:
		P.Error(P.pos, "declaration expected");
		P.Next();  // make progress
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
	p.Ident = P.ParseIdent(nil);

	// package body
	{	P.OpenScope();
		p.Decls = array.New(0);
		for P.tok == Scanner.IMPORT {
			p.Decls.Push(P.ParseDecl(Scanner.IMPORT));
			P.OptSemicolon();
		}
		if !P.deps {
			for P.tok != Scanner.EOF {
				p.Decls.Push(P.ParseDeclaration());
				P.OptSemicolon();
			}
		}
		P.CloseScope();
	}

	p.Comments = P.comments;
	P.CloseScope();

	P.Ecart();
	return p;
}
