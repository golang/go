// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Parser

import (
	"flag";
	"fmt";
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
		fmt.Printf(". ");
	}
}


func (P *Parser) Trace(msg string) {
	P.PrintIndent();
	fmt.Printf("%s {\n", msg);
	P.indent++;
}


func (P *Parser) Ecart() {
	P.indent--;
	P.PrintIndent();
	fmt.Printf("}\n");
}


func (P *Parser) VerifyIndent(indent uint) {
	if indent != P.indent {
		panic("imbalanced tracing code");
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
		fmt.Printf("[%d] %s\n", P.pos, s);
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


func (P *Parser) DeclareInScope(scope *AST.Scope, x AST.Expr, kind int, typ *AST.Type) {
	if P.scope_lev < 0 {
		panic("cannot declare objects in other packages");
	}
	if ident, ok := x.(*AST.Ident); ok {  // ignore bad exprs
		obj := ident.Obj;
		obj.Kind = kind;
		obj.Typ = typ;
		obj.Pnolev = P.scope_lev;
		switch {
		case scope.LookupLocal(obj.Ident) == nil:
			scope.Insert(obj);
		case kind == AST.TYPE:
			// possibly a forward declaration
		case kind == AST.FUNC:
			// possibly a forward declaration
		default:
			P.Error(obj.Pos, `"` + obj.Ident + `" is declared already`);
		}
	}
}


// Declare a comma-separated list of idents or a single ident.
func (P *Parser) Declare(x AST.Expr, kind int, typ *AST.Type) {
	for {
		p, ok := x.(*AST.BinaryExpr);
		if ok && p.Tok == Scanner.COMMA {
			P.DeclareInScope(P.top_scope, p.X, kind, typ);
			x = p.Y;
		} else {
			break;
		}
	}
	P.DeclareInScope(P.top_scope, x, kind, typ);
}


// ----------------------------------------------------------------------------
// AST support

func exprType(x AST.Expr) *AST.Type {
	var typ *AST.Type;
	if t, is_type := x.(*AST.TypeLit); is_type {
		typ = t.Typ
	} else if t, is_ident := x.(*AST.Ident); is_ident {
		// assume a type name
		typ = AST.NewType(t.Pos(), AST.TYPENAME);
		typ.Expr = x;
	} else if t, is_selector := x.(*AST.Selector); is_selector && exprType(t.Sel) != nil {
		// possibly a qualified (type) identifier
		typ = AST.NewType(t.Pos(), AST.TYPENAME);
		typ.Expr = x;
	}
	return typ;
}


func (P *Parser) NoType(x AST.Expr) AST.Expr {
	if x != nil {
		lit, ok := x.(*AST.TypeLit);
		if ok {
			P.Error(lit.Typ.Pos, "expected expression, found type");
			x = &AST.BasicLit{lit.Typ.Pos, Scanner.STRING, ""};
		}
	}
	return x;
}


func (P *Parser) NewBinary(pos, tok int, x, y AST.Expr) *AST.BinaryExpr {
	return &AST.BinaryExpr{pos, tok, P.NoType(x), P.NoType(y)};
}


// ----------------------------------------------------------------------------
// Common productions

func (P *Parser) TryType() *AST.Type;
func (P *Parser) ParseExpression(prec int) AST.Expr;
func (P *Parser) ParseStatement() AST.Stat;
func (P *Parser) OldParseStatement() *AST.StatImpl;
func (P *Parser) ParseDeclaration() *AST.Decl;


// If scope != nil, lookup identifier in scope. Otherwise create one.
func (P *Parser) ParseIdent(scope *AST.Scope) *AST.Ident {
	if P.verbose {
		P.Trace("Ident");
		defer P.Ecart();
	}

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
		x := &AST.Ident{P.pos, obj};
		if P.verbose {
			P.PrintIndent();
			fmt.Printf("ident = \"%s\"\n", P.val);
		}
		P.Next();
		return x;
	}
	
	P.Expect(Scanner.IDENT);  // use Expect() error handling
	return &AST.Ident{P.pos, nil};
}


func (P *Parser) ParseIdentList() AST.Expr {
	if P.verbose {
		P.Trace("IdentList");
		defer P.Ecart();
	}

	var last *AST.BinaryExpr;
	var x AST.Expr = P.ParseIdent(nil);
	for P.tok == Scanner.COMMA {
		pos := P.pos;
		P.Next();
		y := P.ParseIdent(nil);
		if last == nil {
			last = P.NewBinary(pos, Scanner.COMMA, x, y);
			x = last;
		} else {
			last.Y = P.NewBinary(pos, Scanner.COMMA, last.Y, y);
			last = last.Y;
		}
	}

	return x;
}


// ----------------------------------------------------------------------------
// Types

func (P *Parser) ParseType() *AST.Type {
	if P.verbose {
		P.Trace("Type");
		defer P.Ecart();
	}

	t := P.TryType();
	if t == nil {
		P.Error(P.pos, "type expected");
		t = AST.BadType;
	}

	return t;
}


func (P *Parser) ParseVarType() *AST.Type {
	if P.verbose {
		P.Trace("VarType");
		defer P.Ecart();
	}

	return P.ParseType();
}


func (P *Parser) ParseQualifiedIdent() AST.Expr {
	if P.verbose {
		P.Trace("QualifiedIdent");
		defer P.Ecart();
	}

	var x AST.Expr = P.ParseIdent(P.top_scope);
	for P.tok == Scanner.PERIOD {
		pos := P.pos;
		P.Next();
		y := P.ParseIdent(nil);
		x = &AST.Selector{pos, x, y};
	}

	return x;
}


func (P *Parser) ParseTypeName() *AST.Type {
	if P.verbose {
		P.Trace("TypeName");
		defer P.Ecart();
	}

	t := AST.NewType(P.pos, AST.TYPENAME);
	t.Expr = P.ParseQualifiedIdent();

	return t;
}


func (P *Parser) ParseArrayType() *AST.Type {
	if P.verbose {
		P.Trace("ArrayType");
		defer P.Ecart();
	}

	t := AST.NewType(P.pos, AST.ARRAY);
	P.Expect(Scanner.LBRACK);
	if P.tok == Scanner.ELLIPSIS {
		t.Expr = P.NewBinary(P.pos, Scanner.ELLIPSIS, nil, nil);
		P.Next();
	} else if P.tok != Scanner.RBRACK {
		t.Expr = P.ParseExpression(1);
	}
	P.Expect(Scanner.RBRACK);
	t.Elt = P.ParseType();

	return t;
}


func (P *Parser) ParseChannelType() *AST.Type {
	if P.verbose {
		P.Trace("ChannelType");
		defer P.Ecart();
	}

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

	return t;
}


func (P *Parser) ParseVar(expect_ident bool) *AST.Type {
	t := AST.BadType;
	if expect_ident {
		x := P.ParseIdent(nil);
		t = AST.NewType(x.Pos(), AST.TYPENAME);
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
	if P.verbose {
		P.Trace("VarList");
		defer P.Ecart();
	}

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
			if t.Form == AST.TYPENAME {
				if ident, ok := t.Expr.(*AST.Ident); ok {
					list.Set(i, ident);
					continue;
				}
			}
			list.Set(i, &AST.BadExpr{0});
			P.Error(t.Pos, "identifier expected");
		}
		// add type
		list.Push(&AST.TypeLit{typ});

	} else {
		// all list entries are types
		// convert all type entries into type expressions
		for i, n := i0, list.Len(); i < n; i++ {
			t := list.At(i).(*AST.Type);
			list.Set(i, &AST.TypeLit{t});
		}
	}
}


func (P *Parser) ParseParameterList(ellipsis_ok bool) *array.Array {
	if P.verbose {
		P.Trace("ParameterList");
		defer P.Ecart();
	}

	list := array.New(0);
	P.ParseVarList(list, ellipsis_ok);
	for P.tok == Scanner.COMMA {
		P.Next();
		P.ParseVarList(list, ellipsis_ok);
	}

	return list;
}


func (P *Parser) ParseParameters(ellipsis_ok bool) *AST.Type {
	if P.verbose {
		P.Trace("Parameters");
		defer P.Ecart();
	}

	t := AST.NewType(P.pos, AST.STRUCT);
	P.Expect(Scanner.LPAREN);
	if P.tok != Scanner.RPAREN {
		t.List = P.ParseParameterList(ellipsis_ok);
	}
	t.End = P.pos;
	P.Expect(Scanner.RPAREN);

	return t;
}


func (P *Parser) ParseResultList() {
	if P.verbose {
		P.Trace("ResultList");
		defer P.Ecart();
	}

	P.ParseType();
	for P.tok == Scanner.COMMA {
		P.Next();
		P.ParseType();
	}
	if P.tok != Scanner.RPAREN {
		P.ParseType();
	}
}


func (P *Parser) ParseResult(ftyp *AST.Type) *AST.Type {
	if P.verbose {
		P.Trace("Result");
		defer P.Ecart();
	}

	var t *AST.Type;
	if P.tok == Scanner.LPAREN {
		t = P.ParseParameters(false);
	} else if P.tok != Scanner.FUNC {
		typ := P.TryType();
		if typ != nil {
			t = AST.NewType(P.pos, AST.STRUCT);
			t.List = array.New(0);
			t.List.Push(&AST.TypeLit{typ});
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

func (P *Parser) ParseSignature() *AST.Type {
	if P.verbose {
		P.Trace("Signature");
		defer P.Ecart();
	}

	P.OpenScope();
	P.scope_lev++;

	t := AST.NewType(P.pos, AST.FUNCTION);
	t.Scope = P.top_scope;
	t.List = P.ParseParameters(true).List;  // TODO find better solution
	t.End = P.pos;
	t.Elt = P.ParseResult(t);

	P.scope_lev--;
	P.CloseScope();

	return t;
}


func (P *Parser) ParseFunctionType() *AST.Type {
	if P.verbose {
		P.Trace("FunctionType");
		defer P.Ecart();
	}

	P.Expect(Scanner.FUNC);
	return P.ParseSignature();
}


func (P *Parser) ParseMethodSpec(list *array.Array) {
	if P.verbose {
		P.Trace("MethodDecl");
		defer P.Ecart();
	}

	list.Push(P.ParseIdentList());
	t := P.ParseSignature();
	list.Push(&AST.TypeLit{t});
}


func (P *Parser) ParseInterfaceType() *AST.Type {
	if P.verbose {
		P.Trace("InterfaceType");
		defer P.Ecart();
	}

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

	return t;
}


func (P *Parser) ParseMapType() *AST.Type {
	if P.verbose {
		P.Trace("MapType");
		defer P.Ecart();
	}

	t := AST.NewType(P.pos, AST.MAP);
	P.Expect(Scanner.MAP);
	P.Expect(Scanner.LBRACK);
	t.Key = P.ParseVarType();
	P.Expect(Scanner.RBRACK);
	t.Elt = P.ParseVarType();

	return t;
}


func (P *Parser) ParseOperand() AST.Expr

func (P *Parser) ParseStructType() *AST.Type {
	if P.verbose {
		P.Trace("StructType");
		defer P.Ecart();
	}

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
			if x, ok := t.List.At(i).(*AST.Ident); ok {
				P.DeclareInScope(t.Scope, x, AST.FIELD, nil);
			}
		}
	}

	return t;
}


func (P *Parser) ParsePointerType() *AST.Type {
	if P.verbose {
		P.Trace("PointerType");
		defer P.Ecart();
	}

	t := AST.NewType(P.pos, AST.POINTER);
	P.Expect(Scanner.MUL);
	t.Elt = P.ParseType();

	return t;
}


func (P *Parser) TryType() *AST.Type {
	if P.verbose {
		P.Trace("Type (try)");
		defer P.Ecart();
	}

	t := AST.BadType;
	switch P.tok {
	case Scanner.IDENT: t = P.ParseTypeName();
	case Scanner.LBRACK: t = P.ParseArrayType();
	case Scanner.CHAN, Scanner.ARROW: t = P.ParseChannelType();
	case Scanner.INTERFACE: t = P.ParseInterfaceType();
	case Scanner.FUNC: t = P.ParseFunctionType();
	case Scanner.MAP: t = P.ParseMapType();
	case Scanner.STRUCT: t = P.ParseStructType();
	case Scanner.MUL: t = P.ParsePointerType();
	default: t = nil;  // no type found
	}
	return t;
}


// ----------------------------------------------------------------------------
// Blocks


var newstat = flag.Bool("newstat", false, "use new statement parsing - work in progress");


func (P *Parser) ParseStatementList(list *array.Array) {
	if P.verbose {
		P.Trace("StatementList");
		defer P.Ecart();
		defer P.VerifyIndent(P.indent);
	}

	for P.tok != Scanner.CASE && P.tok != Scanner.DEFAULT && P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
		var s interface{};
		if *newstat {
			s = P.ParseStatement();
		} else {
			s = P.OldParseStatement();
		}
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
}


func (P *Parser) ParseBlock(ftyp *AST.Type, tok int) *AST.Block {
	if P.verbose {
		P.Trace("Block");
		defer P.Ecart();
	}

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
				if x, ok := ftyp.List.At(i).(*AST.Ident); ok {
					P.DeclareInScope(P.top_scope, x, AST.VAR, nil);
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

	return b;
}


// ----------------------------------------------------------------------------
// Expressions

func (P *Parser) ParseExpressionList() AST.Expr {
	if P.verbose {
		P.Trace("ExpressionList");
		defer P.Ecart();
	}

	x := P.ParseExpression(1);
	for first := true; P.tok == Scanner.COMMA; {
		pos := P.pos;
		P.Next();
		y := P.ParseExpression(1);
		if first {
			x = P.NewBinary(pos, Scanner.COMMA, x, y);
			first = false;
		} else {
			x.(*AST.BinaryExpr).Y = P.NewBinary(pos, Scanner.COMMA, x.(*AST.BinaryExpr).Y, y);
		}
	}

	return x;
}


func (P *Parser) ParseFunctionLit() AST.Expr {
	if P.verbose {
		P.Trace("FunctionLit");
		defer P.Ecart();
	}

	pos := P.pos;
	P.Expect(Scanner.FUNC);
	typ := P.ParseSignature();
	P.expr_lev++;
	P.scope_lev++;
	body := P.ParseBlock(typ, Scanner.LBRACE);
	P.scope_lev--;
	P.expr_lev--;

	return &AST.FunctionLit{pos, typ, body};
}


func (P *Parser) ParseOperand() AST.Expr {
	if P.verbose {
		P.Trace("Operand");
		defer P.Ecart();
	}

	switch P.tok {
	case Scanner.IDENT:
		return P.ParseIdent(P.top_scope);

	case Scanner.LPAREN:
		// TODO we could have a function type here as in: new(())
		// (currently not working)
		P.Next();
		P.expr_lev++;
		x := P.ParseExpression(1);
		P.expr_lev--;
		P.Expect(Scanner.RPAREN);
		return x;

	case Scanner.INT, Scanner.FLOAT, Scanner.STRING:
		x := &AST.BasicLit{P.pos, P.tok, P.val};
		P.Next();
		if x.Tok == Scanner.STRING {
			// TODO should remember the list instead of
			//      concatenate the strings here
			for ; P.tok == Scanner.STRING; P.Next() {
				x.Val += P.val;
			}
		}
		return x;

	case Scanner.FUNC:
		return P.ParseFunctionLit();

	default:
		t := P.TryType();
		if t != nil {
			return &AST.TypeLit{t};
		} else {
			P.Error(P.pos, "operand expected");
			P.Next();  // make progress
		}
	}

	return &AST.BadExpr{P.pos};
}


func (P *Parser) ParseSelectorOrTypeGuard(x AST.Expr) AST.Expr {
	if P.verbose {
		P.Trace("SelectorOrTypeGuard");
		defer P.Ecart();
	}

	pos := P.pos;
	P.Expect(Scanner.PERIOD);

	if P.tok == Scanner.IDENT {
		x = &AST.Selector{pos, x, P.ParseIdent(nil)};

	} else {
		P.Expect(Scanner.LPAREN);
		x = &AST.TypeGuard{pos, x, P.ParseType()};
		P.Expect(Scanner.RPAREN);
	}

	return x;
}


func (P *Parser) ParseIndex(x AST.Expr) AST.Expr {
	if P.verbose {
		P.Trace("IndexOrSlice");
		defer P.Ecart();
	}

	pos := P.pos;
	P.Expect(Scanner.LBRACK);
	P.expr_lev++;
	i := P.ParseExpression(0);
	P.expr_lev--;
	P.Expect(Scanner.RBRACK);

	return &AST.Index{pos, x, i};
}


func (P *Parser) ParseBinaryExpr(prec1 int) AST.Expr

func (P *Parser) ParseCall(f AST.Expr) AST.Expr {
	if P.verbose {
		P.Trace("Call");
		defer P.Ecart();
	}

	call := &AST.Call{P.pos, f, nil};
	P.Expect(Scanner.LPAREN);
	if P.tok != Scanner.RPAREN {
		P.expr_lev++;
		var t *AST.Type;
		if x0, ok := f.(*AST.Ident); ok && (x0.Obj.Ident == "new" || x0.Obj.Ident == "make") {
			// heuristic: assume it's a new(T) or make(T, ...) call, try to parse a type
			t = P.TryType();
		}
		if t != nil {
			// we found a type
			args := &AST.TypeLit{t};
			if P.tok == Scanner.COMMA {
				pos := P.pos;
				P.Next();
				y := P.ParseExpressionList();
				// create list manually because NewExpr checks for type expressions
				args := &AST.BinaryExpr{pos, Scanner.COMMA, args, y};
			}
			call.Args = args;
		} else {
			// normal argument list
			call.Args = P.ParseExpressionList();
		}
		P.expr_lev--;
	}
	P.Expect(Scanner.RPAREN);

	return call;
}


func (P *Parser) ParseCompositeElements() AST.Expr {
	x := P.ParseExpression(0);
	if P.tok == Scanner.COMMA {
		pos := P.pos;
		P.Next();

		// first element determines mode
		singles := true;
		if t, is_binary := x.(*AST.BinaryExpr); is_binary && t.Tok == Scanner.COLON {
			singles = false;
		}

		var last *AST.BinaryExpr;
		for P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
			y := P.ParseExpression(0);

			if singles {
				if t, is_binary := y.(*AST.BinaryExpr); is_binary && t.Tok == Scanner.COLON {
					P.Error(t.X.Pos(), "single value expected; found pair");
				}
			} else {
				if t, is_binary := y.(*AST.BinaryExpr); !is_binary || t.Tok != Scanner.COLON {
					P.Error(y.Pos(), "key:value pair expected; found single value");
				}
			}

			if last == nil {
				last = P.NewBinary(pos, Scanner.COMMA, x, y);
				x = last;
			} else {
				last.Y = P.NewBinary(pos, Scanner.COMMA, last.Y, y);
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


func (P *Parser) ParseCompositeLit(t *AST.Type) AST.Expr {
	if P.verbose {
		P.Trace("CompositeLit");
		defer P.Ecart();
	}

	pos := P.pos;
	P.Expect(Scanner.LBRACE);
	var elts AST.Expr;
	if P.tok != Scanner.RBRACE {
		elts = P.ParseCompositeElements();
	}
	P.Expect(Scanner.RBRACE);

	return &AST.CompositeLit{pos, t, elts};
}


func (P *Parser) ParsePrimaryExpr() AST.Expr {
	if P.verbose {
		P.Trace("PrimaryExpr");
		defer P.Ecart();
	}

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
				return x;
			}
		default:
			return x;
		}
	}

	unreachable();
	return nil;
}


func (P *Parser) ParseUnaryExpr() AST.Expr {
	if P.verbose {
		P.Trace("UnaryExpr");
		defer P.Ecart();
	}

	switch P.tok {
	case Scanner.ADD, Scanner.SUB, Scanner.MUL, Scanner.NOT, Scanner.XOR, Scanner.ARROW, Scanner.AND:
		pos, tok := P.pos, P.tok;
		P.Next();
		y := P.ParseUnaryExpr();
		if lit, ok := y.(*AST.TypeLit); ok && tok == Scanner.MUL {
			// pointer type
			t := AST.NewType(pos, AST.POINTER);
			t.Elt = lit.Typ;
			return &AST.TypeLit{t};
		} else {
			return &AST.UnaryExpr{pos, tok, y};
		}
	}

	return P.ParsePrimaryExpr();
}


func (P *Parser) ParseBinaryExpr(prec1 int) AST.Expr {
	if P.verbose {
		P.Trace("BinaryExpr");
		defer P.Ecart();
	}

	x := P.ParseUnaryExpr();
	for prec := Scanner.Precedence(P.tok); prec >= prec1; prec-- {
		for Scanner.Precedence(P.tok) == prec {
			pos, tok := P.pos, P.tok;
			P.Next();
			y := P.ParseBinaryExpr(prec + 1);
			x = P.NewBinary(pos, tok, x, y);
		}
	}

	return x;
}


func (P *Parser) ParseExpression(prec int) AST.Expr {
	if P.verbose {
		P.Trace("Expression");
		defer P.Ecart();
		defer P.VerifyIndent(P.indent);
	}

	if prec < 0 {
		panic("precedence must be >= 0");
	}

	return P.NoType(P.ParseBinaryExpr(prec));
}


// ----------------------------------------------------------------------------
// Statements

func (P *Parser) ParseSimpleStat(range_ok bool) AST.Stat {
	if P.verbose {
		P.Trace("SimpleStat");
		defer P.Ecart();
	}

	x := P.ParseExpressionList();

	switch P.tok {
	case Scanner.COLON:
		// label declaration
		pos := P.pos;
		P.Next();  // consume ":"
		if AST.ExprLen(x) == 1 {
			if label, is_ident := x.(*AST.Ident); is_ident {
				return &AST.LabelDecl{pos, label};
			}
		}
		P.Error(x.Pos(), "illegal label declaration");
		return nil;
		
	case
		Scanner.DEFINE, Scanner.ASSIGN, Scanner.ADD_ASSIGN,
		Scanner.SUB_ASSIGN, Scanner.MUL_ASSIGN, Scanner.QUO_ASSIGN,
		Scanner.REM_ASSIGN, Scanner.AND_ASSIGN, Scanner.OR_ASSIGN,
		Scanner.XOR_ASSIGN, Scanner.SHL_ASSIGN, Scanner.SHR_ASSIGN:
		// declaration/assignment
		pos, tok := P.pos, P.tok;
		P.Next();
		var y AST.Expr;
		if range_ok && P.tok == Scanner.RANGE {
			range_pos := P.pos;
			P.Next();
			y = &AST.UnaryExpr{range_pos, Scanner.RANGE, P.ParseExpression(1)};
			if tok != Scanner.DEFINE && tok != Scanner.ASSIGN {
				P.Error(pos, "expected '=' or ':=', found '" + Scanner.TokenString(tok) + "'");
			}
		} else {
			y = P.ParseExpressionList();
			if xl, yl := AST.ExprLen(x), AST.ExprLen(y); xl > 1 && yl > 1 && xl != yl {
				P.Error(x.Pos(), "arity of lhs doesn't match rhs");
			}
		}
		// TODO changed ILLEGAL -> NONE
		return &AST.ExpressionStat{x.Pos(), Scanner.ILLEGAL, P.NewBinary(pos, tok, x, y)};
		
	default:
		if AST.ExprLen(x) != 1 {
			P.Error(x.Pos(), "only one expression allowed");
		}
		
		if P.tok == Scanner.INC || P.tok == Scanner.DEC {
			s := &AST.ExpressionStat{P.pos, P.tok, x};
			P.Next();  // consume "++" or "--"
			return s;
		}
		
		// TODO changed ILLEGAL -> NONE
		return &AST.ExpressionStat{x.Pos(), Scanner.ILLEGAL, x};
	}

	unreachable();
	return nil;
}


func (P *Parser) OldParseSimpleStat(range_ok bool) *AST.StatImpl {
	if P.verbose {
		P.Trace("SimpleStat");
		defer P.Ecart();
	}

	s := AST.OldBadStat;
	x := P.ParseExpressionList();

	switch P.tok {
	case Scanner.COLON:
		// label declaration
		s = AST.NewStat(P.pos, Scanner.COLON);
		s.Expr = x;
		if AST.ExprLen(x) != 1 {
			P.Error(x.Pos(), "illegal label declaration");
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
		var y AST.Expr = &AST.BadExpr{pos};
		if range_ok && P.tok == Scanner.RANGE {
			range_pos := P.pos;
			P.Next();
			y = P.ParseExpression(1);
			y = P.NewBinary(range_pos, Scanner.RANGE, nil, y);
			if tok != Scanner.DEFINE && tok != Scanner.ASSIGN {
				P.Error(pos, "expected '=' or ':=', found '" + Scanner.TokenString(tok) + "'");
			}
		} else {
			y = P.ParseExpressionList();
			if xl, yl := AST.ExprLen(x), AST.ExprLen(y); xl > 1 && yl > 1 && xl != yl {
				P.Error(x.Pos(), "arity of lhs doesn't match rhs");
			}
		}
		s = AST.NewStat(x.Pos(), Scanner.EXPRSTAT);
		s.Expr = P.NewBinary(pos, tok, x, y);
		
	default:
		var pos, tok int;
		if P.tok == Scanner.INC || P.tok == Scanner.DEC {
			pos, tok = P.pos, P.tok;
			P.Next();
		} else {
			pos, tok = x.Pos(), Scanner.EXPRSTAT;
		}
		s = AST.NewStat(pos, tok);
		s.Expr = x;
		if AST.ExprLen(x) != 1 {
			P.Error(pos, "only one expression allowed");
			panic();  // fix position
		}
	}

	return s;
}


func (P *Parser) ParseInvocationStat(keyword int) *AST.ExpressionStat {
	if P.verbose {
		P.Trace("InvocationStat");
		defer P.Ecart();
	}

	pos := P.pos;
	P.Expect(keyword);
	return &AST.ExpressionStat{pos, keyword, P.ParseExpression(1)};
}


func (P *Parser) OldParseInvocationStat(keyword int) *AST.StatImpl {
	if P.verbose {
		P.Trace("InvocationStat");
		defer P.Ecart();
	}

	s := AST.NewStat(P.pos, keyword);
	P.Expect(keyword);
	s.Expr = P.ParseExpression(1);

	return s;
}


func (P *Parser) ParseReturnStat() *AST.ExpressionStat {
	if P.verbose {
		P.Trace("ReturnStat");
		defer P.Ecart();
	}

	pos := P.pos;
	P.Expect(Scanner.RETURN);
	var x AST.Expr;
	if P.tok != Scanner.SEMICOLON && P.tok != Scanner.RBRACE {
		x = P.ParseExpressionList();
	}

	return &AST.ExpressionStat{pos, Scanner.RETURN, x};
}


func (P *Parser) OldParseReturnStat() *AST.StatImpl {
	if P.verbose {
		P.Trace("ReturnStat");
		defer P.Ecart();
	}

	s := AST.NewStat(P.pos, Scanner.RETURN);
	P.Expect(Scanner.RETURN);
	if P.tok != Scanner.SEMICOLON && P.tok != Scanner.RBRACE {
		s.Expr = P.ParseExpressionList();
	}

	return s;
}


func (P *Parser) ParseControlFlowStat(tok int) *AST.StatImpl {
	if P.verbose {
		P.Trace("ControlFlowStat");
		defer P.Ecart();
	}

	s := AST.NewStat(P.pos, tok);
	P.Expect(tok);
	if tok != Scanner.FALLTHROUGH && P.tok == Scanner.IDENT {
		s.Expr = P.ParseIdent(P.top_scope);
	}

	return s;
}


func (P *Parser) ParseControlClause(isForStat bool) (init AST.Stat, expr AST.Expr, post AST.Stat) {
	if P.verbose {
		P.Trace("ControlClause");
		defer P.Ecart();
	}

	if P.tok != Scanner.LBRACE {
		prev_lev := P.expr_lev;
		P.expr_lev = -1;
		if P.tok != Scanner.SEMICOLON {
			init = P.ParseSimpleStat(isForStat);
			// TODO check for range clause and exit if found
		}
		if P.tok == Scanner.SEMICOLON {
			P.Next();
			if P.tok != Scanner.SEMICOLON && P.tok != Scanner.LBRACE {
				expr = P.ParseExpression(1);
			}
			if isForStat {
				P.Expect(Scanner.SEMICOLON);
				if P.tok != Scanner.LBRACE {
					post = P.ParseSimpleStat(false);
				}
			}
		} else {
			if init != nil {  // guard in case of errors
				if s, is_expr_stat := init.(*AST.ExpressionStat); is_expr_stat {
					expr, init = s.Expr, nil;
				} else {
					P.Error(0, "illegal control clause");
				}
			}
		}
		P.expr_lev = prev_lev;
	}

	return init, expr, post;
}


func (P *Parser) OldParseControlClause(keyword int) *AST.StatImpl {
	if P.verbose {
		P.Trace("ControlClause");
		defer P.Ecart();
	}

	s := AST.NewStat(P.pos, keyword);
	P.Expect(keyword);
	if P.tok != Scanner.LBRACE {
		prev_lev := P.expr_lev;
		P.expr_lev = -1;
		if P.tok != Scanner.SEMICOLON {
			s.Init = P.OldParseSimpleStat(keyword == Scanner.FOR);
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
					s.Post = P.OldParseSimpleStat(false);
				}
			}
		} else {
			if s.Init != nil {  // guard in case of errors
				s.Expr, s.Init = s.Init.Expr, nil;
			}
		}
		P.expr_lev = prev_lev;
	}

	return s;
}


func (P *Parser) ParseIfStat() *AST.IfStat {
	if P.verbose {
		P.Trace("IfStat");
		defer P.Ecart();
	}

	P.OpenScope();
	pos := P.pos;
	P.Expect(Scanner.IF);
	init, cond, dummy := P.ParseControlClause(false);
	body := P.ParseBlock(nil, Scanner.LBRACE);
	var else_ AST.Stat;
	if P.tok == Scanner.ELSE {
		P.Next();
		if P.tok == Scanner.IF || P.tok == Scanner.LBRACE {
			else_ = P.ParseStatement();
		} else if P.sixg {
			else_ = P.ParseStatement();
			if else_ != nil {
				// not the empty statement
				// wrap in a block since we don't have one
				panic();
				/*
				b := AST.NewStat(s1.Pos, Scanner.LBRACE);
				b.Body = AST.NewBlock(s1.Pos, Scanner.LBRACE);
				b.Body.List.Push(s1);
				s1 = b;
				*/
			}
		} else {
			P.Error(P.pos, "'if' or '{' expected - illegal 'else' branch");
		}
	}
	P.CloseScope();

	return &AST.IfStat{pos, init, cond, body, else_ };
}


func (P *Parser) OldParseIfStat() *AST.StatImpl {
	if P.verbose {
		P.Trace("IfStat");
		defer P.Ecart();
	}

	P.OpenScope();
	s := P.OldParseControlClause(Scanner.IF);
	s.Body = P.ParseBlock(nil, Scanner.LBRACE);
	if P.tok == Scanner.ELSE {
		P.Next();
		s1 := AST.OldBadStat;
		if P.tok == Scanner.IF || P.tok == Scanner.LBRACE {
			s1 = P.OldParseStatement();
		} else if P.sixg {
			s1 = P.OldParseStatement();
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

	return s;
}


func (P *Parser) ParseForStat() *AST.ForStat {
	if P.verbose {
		P.Trace("ForStat");
		defer P.Ecart();
	}

	P.OpenScope();
	pos := P.pos;
	P.Expect(Scanner.FOR);
	init, cond, post := P.ParseControlClause(true);
	body := P.ParseBlock(nil, Scanner.LBRACE);
	P.CloseScope();

	return &AST.ForStat{pos, init, cond, post, body};
}


func (P *Parser) OldParseForStat() *AST.StatImpl {
	if P.verbose {
		P.Trace("ForStat");
		defer P.Ecart();
	}

	P.OpenScope();
	s := P.OldParseControlClause(Scanner.FOR);
	s.Body = P.ParseBlock(nil, Scanner.LBRACE);
	P.CloseScope();

	return s;
}


func (P *Parser) ParseSwitchCase() *AST.StatImpl {
	if P.verbose {
		P.Trace("SwitchCase");
		defer P.Ecart();
	}

	s := AST.NewStat(P.pos, P.tok);
	if P.tok == Scanner.CASE {
		P.Next();
		s.Expr = P.ParseExpressionList();
	} else {
		P.Expect(Scanner.DEFAULT);
	}

	return s;
}


func (P *Parser) ParseCaseClause() *AST.StatImpl {
	if P.verbose {
		P.Trace("CaseClause");
		defer P.Ecart();
	}

	s := P.ParseSwitchCase();
	s.Body = P.ParseBlock(nil, Scanner.COLON);

	return s;
}


func (P *Parser) ParseSwitchStat() *AST.SwitchStat {
	if P.verbose {
		P.Trace("SwitchStat");
		defer P.Ecart();
	}

	P.OpenScope();
	pos := P.pos;
	P.Expect(Scanner.SWITCH);
	init, tag, post := P.ParseControlClause(false);
	body := AST.NewBlock(P.pos, Scanner.LBRACE);
	P.Expect(Scanner.LBRACE);
	for P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
		body.List.Push(P.ParseCaseClause());
	}
	body.End = P.pos;
	P.Expect(Scanner.RBRACE);
	P.opt_semi = true;
	P.CloseScope();

	return &AST.SwitchStat{pos, init, tag, body};
}


func (P *Parser) OldParseSwitchStat() *AST.StatImpl {
	if P.verbose {
		P.Trace("SwitchStat");
		defer P.Ecart();
	}

	P.OpenScope();
	s := P.OldParseControlClause(Scanner.SWITCH);
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

	return s;
}


func (P *Parser) ParseCommCase() *AST.StatImpl {
	if P.verbose {
		P.Trace("CommCase");
		defer P.Ecart();
	}

	s := AST.NewStat(P.pos, P.tok);
	if P.tok == Scanner.CASE {
		P.Next();
		x := P.ParseExpression(1);
		if P.tok == Scanner.ASSIGN || P.tok == Scanner.DEFINE {
			pos, tok := P.pos, P.tok;
			P.Next();
			if P.tok == Scanner.ARROW {
				y := P.ParseExpression(1);
				x = P.NewBinary(pos, tok, x, y);
			} else {
				P.Expect(Scanner.ARROW);  // use Expect() error handling
			}
		}
		s.Expr = x;
	} else {
		P.Expect(Scanner.DEFAULT);
	}

	return s;
}


func (P *Parser) ParseCommClause() *AST.StatImpl {
	if P.verbose {
		P.Trace("CommClause");
		defer P.Ecart();
	}

	s := P.ParseCommCase();
	s.Body = P.ParseBlock(nil, Scanner.COLON);

	return s;
}


func (P *Parser) ParseSelectStat() *AST.SelectStat {
	if P.verbose {
		P.Trace("SelectStat");
		defer P.Ecart();
	}

	P.OpenScope();
	pos := P.pos;
	P.Expect(Scanner.SELECT);
	body := AST.NewBlock(P.pos, Scanner.LBRACE);
	P.Expect(Scanner.LBRACE);
	for P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
		body.List.Push(P.ParseCommClause());
	}
	body.End = P.pos;
	P.Expect(Scanner.RBRACE);
	P.opt_semi = true;
	P.CloseScope();

	return &AST.SelectStat{pos, body};
}


func (P *Parser) OldParseSelectStat() *AST.StatImpl {
	if P.verbose {
		P.Trace("SelectStat");
		defer P.Ecart();
	}

	P.OpenScope();
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
	P.CloseScope();
	s.Body = b;

	return s;
}


func (P *Parser) ParseStatement() AST.Stat {
	if P.verbose {
		P.Trace("Statement");
		defer P.Ecart();
		defer P.VerifyIndent(P.indent);
	}

	s := AST.OldBadStat;
	switch P.tok {
	case Scanner.CONST, Scanner.TYPE, Scanner.VAR:
		return &AST.DeclarationStat{P.ParseDeclaration()};
	case Scanner.FUNC:
		// for now we do not allow local function declarations,
		// instead we assume this starts a function literal
		fallthrough;
	case
		// only the tokens that are legal top-level expression starts
		Scanner.IDENT, Scanner.INT, Scanner.FLOAT, Scanner.STRING, Scanner.LPAREN,  // operand
		Scanner.LBRACK, Scanner.STRUCT,  // composite type
		Scanner.MUL, Scanner.AND, Scanner.ARROW:  // unary
		return P.ParseSimpleStat(false);
	case Scanner.GO, Scanner.DEFER:
		return P.ParseInvocationStat(P.tok);
	case Scanner.RETURN:
		return P.ParseReturnStat();
	case Scanner.BREAK, Scanner.CONTINUE, Scanner.GOTO, Scanner.FALLTHROUGH:
		s = P.ParseControlFlowStat(P.tok);
	case Scanner.LBRACE:
		s = AST.NewStat(P.pos, Scanner.LBRACE);
		s.Body = P.ParseBlock(nil, Scanner.LBRACE);
	case Scanner.IF:
		return P.ParseIfStat();
	case Scanner.FOR:
		return P.ParseForStat();
	case Scanner.SWITCH:
		return P.ParseSwitchStat();
	case Scanner.SELECT:
		return P.ParseSelectStat();
	}

	// empty statement
	return nil;
}


func (P *Parser) OldParseStatement() *AST.StatImpl {
	if P.verbose {
		P.Trace("Statement");
		defer P.Ecart();
		defer P.VerifyIndent(P.indent);
	}

	s := AST.OldBadStat;
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
		s = P.OldParseSimpleStat(false);
	case Scanner.GO, Scanner.DEFER:
		s = P.OldParseInvocationStat(P.tok);
	case Scanner.RETURN:
		s = P.OldParseReturnStat();
	case Scanner.BREAK, Scanner.CONTINUE, Scanner.GOTO, Scanner.FALLTHROUGH:
		s = P.ParseControlFlowStat(P.tok);
	case Scanner.LBRACE:
		s = AST.NewStat(P.pos, Scanner.LBRACE);
		s.Body = P.ParseBlock(nil, Scanner.LBRACE);
	case Scanner.IF:
		s = P.OldParseIfStat();
	case Scanner.FOR:
		s = P.OldParseForStat();
	case Scanner.SWITCH:
		s = P.OldParseSwitchStat();
	case Scanner.SELECT:
		s = P.OldParseSelectStat();
	default:
		// empty statement
		s = nil;
	}

	return s;
}


// ----------------------------------------------------------------------------
// Declarations

func (P *Parser) ParseImportSpec(d *AST.Decl) {
	if P.verbose {
		P.Trace("ImportSpec");
		defer P.Ecart();
	}

	if P.tok == Scanner.PERIOD {
		P.Error(P.pos, `"import ." not yet handled properly`);
		P.Next();
	} else if P.tok == Scanner.IDENT {
		d.Ident = P.ParseIdent(nil);
	}

	if P.tok == Scanner.STRING {
		// TODO eventually the scanner should strip the quotes
		d.Val = &AST.BasicLit{P.pos, Scanner.STRING, P.val};
		P.Next();
	} else {
		P.Expect(Scanner.STRING);  // use Expect() error handling
	}
}


func (P *Parser) ParseConstSpec(d *AST.Decl) {
	if P.verbose {
		P.Trace("ConstSpec");
		defer P.Ecart();
	}

	d.Ident = P.ParseIdentList();
	d.Typ = P.TryType();
	if P.tok == Scanner.ASSIGN {
		P.Next();
		d.Val = P.ParseExpressionList();
	}
}


func (P *Parser) ParseTypeSpec(d *AST.Decl) {
	if P.verbose {
		P.Trace("TypeSpec");
		defer P.Ecart();
	}

	d.Ident = P.ParseIdent(nil);
	d.Typ = P.ParseType();
	P.opt_semi = true;
}


func (P *Parser) ParseVarSpec(d *AST.Decl) {
	if P.verbose {
		P.Trace("VarSpec");
		defer P.Ecart();
	}

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
}


func (P *Parser) ParseSpec(d *AST.Decl) {
	kind := AST.NONE;
	
	switch d.Tok {
	case Scanner.IMPORT: P.ParseImportSpec(d); kind = AST.PACKAGE;
	case Scanner.CONST: P.ParseConstSpec(d); kind = AST.CONST;
	case Scanner.TYPE: P.ParseTypeSpec(d); kind = AST.TYPE;
	case Scanner.VAR: P.ParseVarSpec(d); kind = AST.VAR;
	default: unreachable();
	}

	// semantic checks
	if d.Tok == Scanner.IMPORT {
		if d.Ident != nil {
			P.Declare(d.Ident, kind, nil);
		}
	} else {
		P.Declare(d.Ident, kind, d.Typ);
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
					P.Error(AST.ExprAt(d.Val, llen).Pos(), "more expressions than variables");
				} else {
					P.Error(AST.ExprAt(d.Ident, rlen).Pos(), "more variables than expressions");
				}
			}
		} else {
			// TODO
		}
	}
}


func (P *Parser) ParseDecl(keyword int) *AST.Decl {
	if P.verbose {
		P.Trace("Decl");
		defer P.Ecart();
	}

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
	if P.verbose {
		P.Trace("FunctionDecl");
		defer P.Ecart();
	}

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

	ident := P.ParseIdent(nil);
	d.Ident = ident;
	d.Typ = P.ParseSignature();
	d.Typ.Key = recv;

	if P.tok == Scanner.LBRACE {
		d.Body = P.ParseBlock(d.Typ, Scanner.LBRACE);
	}

	return d;
}


func (P *Parser) ParseDeclaration() *AST.Decl {
	if P.verbose {
		P.Trace("Declaration");
		defer P.Ecart();
		defer P.VerifyIndent(P.indent);
	}
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

	return d;
}


// ----------------------------------------------------------------------------
// Program

func (P *Parser) ParseProgram() *AST.Program {
	if P.verbose {
		P.Trace("Program");
		defer P.Ecart();
	}

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

	return p;
}
