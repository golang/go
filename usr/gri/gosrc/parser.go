// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Parser

import Utils "utils"
import Scanner "scanner"
import Globals "globals"
import Object "object"
import Type "type"
import Universe "universe"
import Import "import"
import AST "ast"
import Expr "expr"


export type Parser struct {
	comp *Globals.Compilation;
	verbose bool;
	indent uint;
	scanner *Scanner.Scanner;
	tokchan chan *Scanner.Token;
	
	// Token
	tok int;  // one token look-ahead
	pos int;  // token source position
	val string;  // token value (for IDENT, NUMBER, STRING only)

	// Semantic analysis
	level int;  // 0 = global scope, -1 = function/struct scope of global functions/structs, etc.
	top_scope *Globals.Scope;
	forward_types *Globals.List;
	exports *Globals.List;
}


// ----------------------------------------------------------------------------
// Support functions

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
	P.indent++;  // always, so proper identation is always checked
}


func (P *Parser) Ecart() {
	P.indent--;  // always, so proper identation is always checked
	if P.verbose {
		P.PrintIndent();
		print("}\n");
	}
}


func (P *Parser) Next() {
	if P.tokchan == nil {
		P.tok, P.pos, P.val = P.scanner.Scan();
	} else {
		t := <- P.tokchan;
		P.tok, P.pos, P.val = t.tok, t.pos, t.val;
	}
	if P.verbose {
		P.PrintIndent();
		print("[", P.pos, "] ", Scanner.TokenName(P.tok), "\n");
	}
}


func (P *Parser) Open(comp *Globals.Compilation, scanner *Scanner.Scanner, tokchan chan *Scanner.Token) {
	P.comp = comp;
	P.verbose = comp.flags.verbosity > 2;
	P.indent = 0;
	P.scanner = scanner;
	P.tokchan = tokchan;
	P.Next();
	P.level = 0;
	P.top_scope = Universe.scope;
	P.forward_types = Globals.NewList();
	P.exports = Globals.NewList();
}


func (P *Parser) Error(pos int, msg string) {
	P.scanner.Error(pos, msg);
}


func (P *Parser) Expect(tok int) {
	if P.tok != tok {
		P.Error(P.pos, "expected '" + Scanner.TokenName(tok) + "', found '" + Scanner.TokenName(P.tok) + "'");
	}
	P.Next();  // make progress in any case
}


func (P *Parser) Optional(tok int) {
	if P.tok == tok {
		P.Next();
	}
}


// ----------------------------------------------------------------------------
// Scopes

func (P *Parser) OpenScope() {
	P.top_scope = Globals.NewScope(P.top_scope);
}


func (P *Parser) CloseScope() {
	P.top_scope = P.top_scope.parent;
}


func (P *Parser) Lookup(ident string) *Globals.Object {
	for scope := P.top_scope; scope != nil; scope = scope.parent {
		obj := scope.Lookup(ident);
		if obj != nil {
			return obj;
		}
	}
	return nil;
}


func (P *Parser) DeclareInScope(scope *Globals.Scope, obj *Globals.Object) {
	if P.level > 0 {
		panic("cannot declare objects in other packages");
	}
	obj.pnolev = P.level;
	if scope.Lookup(obj.ident) != nil {
		P.Error(obj.pos, `"` + obj.ident + `" is declared already`);
		return;  // don't insert it into the scope
	}
	scope.Insert(obj);
}


func (P *Parser) Declare(obj *Globals.Object) {
	P.DeclareInScope(P.top_scope, obj);
}


func MakeFunctionType(sig *Globals.Scope, p0, r0 int) *Globals.Type {
	form := Type.FUNCTION;
	if p0 == 1 {
		form = Type.METHOD;
	} else {
		if p0 != 0 {
			panic("incorrect p0");
		}
	}
	typ := Globals.NewType(form);
	typ.len = r0 - p0;
	typ.scope = sig;

	// set result type
	if sig.entries.len - r0 == 1 {
		// exactly one result value
		typ.elt = sig.entries.last.obj.typ;
	} else {
		// 0 or >1 result values - create a tuple referring to this type
		tup := Globals.NewType(Type.TUPLE);
		tup.elt = typ;
		typ.elt = tup;
	}

	// parameters/results are always exported (they can't be accessed
	// w/o the function or function type being exported)
	for p := sig.entries.first; p != nil; p = p.next {
		p.obj.exported = true;
	}

	return typ;
}


func (P *Parser) DeclareFunc(pos int, ident string, typ *Globals.Type) *Globals.Object {
	// determine scope
	scope := P.top_scope;
	if typ.form == Type.METHOD {
		// declare in corresponding struct
		if typ.scope.entries.len < 1 {
			panic("no recv in signature?");
		}
		recv_typ := typ.scope.entries.first.obj.typ;
		if recv_typ.form == Type.POINTER {
			recv_typ = recv_typ.elt;
		}
		scope = recv_typ.scope;
	}

	// declare the function
	obj := scope.Lookup(ident);
	if obj == nil {
		obj = Globals.NewObject(pos, Object.FUNC, ident);
		obj.typ = typ;
		// TODO do we need to set the primary type? probably...
		P.DeclareInScope(scope, obj);
		return obj;
	}

	// obj != NULL: possibly a forward declaration
	if obj.kind != Object.FUNC {
		P.Error(pos, `"` + ident + `" is declared already`);
		// continue but do not insert this function into the scope
		obj = Globals.NewObject(-1, Object.FUNC, ident);
		obj.typ = typ;
		// TODO do we need to set the primary type? probably...
		return obj;
	}

	// we have a function with the same name
	if !Type.Equal(typ, obj.typ) {
		P.Error(-1, `type of "` + ident + `" does not match its forward declaration`);
		// continue but do not insert this function into the scope
		obj = Globals.NewObject(-1, Object.FUNC, ident);
		obj.typ = typ;
		// TODO do we need to set the primary type? probably...
		return obj;
	}

	// We have a matching forward declaration. Use it.
	return obj;
}


// ----------------------------------------------------------------------------
// Common productions


func (P *Parser) TryType() *Globals.Type;
func (P *Parser) ParseExpression() Globals.Expr;
func (P *Parser) TryStatement() bool;
func (P *Parser) ParseDeclaration();


func (P *Parser) ParseIdent(allow_keyword bool) (pos int, ident string) {
	P.Trace("Ident");

	pos, ident = P.pos, "";
	// NOTE Can make this faster by not doing the keyword lookup in the
	// scanner if we don't care about keywords.
	if P.tok == Scanner.IDENT || allow_keyword && P.tok > Scanner.IDENT {
		ident = P.val;
		if P.verbose {
			P.PrintIndent();
			print("Ident = \"", ident, "\"\n");
		}
		P.Next();
	} else {
		P.Expect(Scanner.IDENT);  // use Expect() error handling
	}

	P.Ecart();
	return pos, ident;
}


func (P *Parser) ParseIdentDecl(kind int) *Globals.Object {
	P.Trace("IdentDecl");

	pos, ident := P.ParseIdent(kind == Object.FIELD);
	obj := Globals.NewObject(pos, kind, ident);
	P.Declare(obj);

	P.Ecart();
	return obj;
}


func (P *Parser) ParseIdentDeclList(kind int) *Globals.List {
	P.Trace("IdentDeclList");

	list := Globals.NewList();
	list.AddObj(P.ParseIdentDecl(kind));
	for P.tok == Scanner.COMMA {
		P.Next();
		list.AddObj(P.ParseIdentDecl(kind));
	}

	P.Ecart();
	return list;
}


func (P *Parser) ParseIdentList() {
	P.Trace("IdentList");
	P.ParseIdent(false);
	for P.tok == Scanner.COMMA {
		P.Next();
		P.ParseIdent(false);
	}
	P.Ecart();
}


func (P *Parser) ParseQualifiedIdent(pos int, ident string) *Globals.Object {
	P.Trace("QualifiedIdent");

	if pos < 0 {
		pos, ident = P.ParseIdent(false);
	}

	obj := P.Lookup(ident);
	if obj == nil {
		P.Error(pos, `"` + ident + `" is not declared`);
		obj = Globals.NewObject(pos, Object.BAD, ident);
	}

	if obj.kind == Object.PACKAGE && P.tok == Scanner.PERIOD {
		if obj.pnolev < 0 {
			panic("obj.pnolev < 0");
		}
		pkg := P.comp.pkg_list[obj.pnolev];
		//if pkg.obj.ident != ident {
		//	panic("pkg.obj.ident != ident");
		//}
		P.Next();  // consume "."
		pos, ident = P.ParseIdent(false);
		obj = pkg.scope.Lookup(ident);
		if obj == nil {
			P.Error(pos, `"` + ident + `" is not declared in package "` + pkg.obj.ident + `"`);
			obj = Globals.NewObject(pos, Object.BAD, ident);
		}
	}

	P.Ecart();
	return obj;
}


// ----------------------------------------------------------------------------
// Types

func (P *Parser) ParseType() *Globals.Type {
	P.Trace("Type");

	typ := P.TryType();
	if typ == nil {
		P.Error(P.pos, "type expected");
		typ = Universe.bad_t;
	}

	P.Ecart();
	return typ;
}


func (P *Parser) ParseVarType() *Globals.Type {
	P.Trace("VarType");

	pos := P.pos;
	typ := P.ParseType();

	switch typ.form {
	case Type.ARRAY:
		if P.comp.flags.sixg || typ.len >= 0 {
			break;
		}
		// open arrays must be pointers
		fallthrough;
		
	case Type.MAP, Type.CHANNEL, Type.FUNCTION:
		P.Error(pos, "must be pointer to this type");
		typ = Universe.bad_t;
	}

	P.Ecart();
	return typ;
}


func (P *Parser) ParseTypeName() *Globals.Type {
	P.Trace("TypeName");

	pos := P.pos;
	obj := P.ParseQualifiedIdent(-1, "");
	typ := obj.typ;
	if obj.kind != Object.TYPE {
		P.Error(pos, "qualified identifier does not denote a type");
		typ = Universe.bad_t;
	}

	P.Ecart();
	return typ;
}


func (P *Parser) ParseArrayType() *Globals.Type {
	P.Trace("ArrayType");

	P.Expect(Scanner.LBRACK);
	typ := Globals.NewType(Type.ARRAY);
	if P.tok != Scanner.RBRACK {
		// TODO set typ.len
		P.ParseExpression();
	}
	P.Expect(Scanner.RBRACK);
	typ.elt = P.ParseVarType();

	P.Ecart();
	return typ;
}


func (P *Parser) ParseChannelType() *Globals.Type {
	P.Trace("ChannelType");

	typ := Globals.NewType(Type.CHANNEL);
	if P.tok == Scanner.CHAN {
		P.Next();
		if P.tok == Scanner.ARROW {
			typ.aux = Type.SEND;
			P.Next();
		} else {
			typ.aux = Type.SEND + Type.RECV;
		}
	} else {
		P.Expect(Scanner.ARROW);
		P.Expect(Scanner.CHAN);
		typ.aux = Type.RECV;
	}
	typ.elt = P.ParseVarType();

	P.Ecart();
	return typ;
}


func (P *Parser) ParseVarDeclList(kind int) {
	P.Trace("VarDeclList");

	list := P.ParseIdentDeclList(kind);
	typ := P.ParseVarType();
	for p := list.first; p != nil; p = p.next {
		p.obj.typ = typ;  // TODO should use/have set_type()
	}

	P.Ecart();
}


func (P *Parser) ParseParameterList() {
	P.Trace("ParameterList");

	P.ParseVarDeclList(Object.VAR);
	for P.tok == Scanner.COMMA {
		P.Next();
		P.ParseVarDeclList(Object.VAR);
	}

	P.Ecart();
}


func (P *Parser) ParseParameters() {
	P.Trace("Parameters");

	P.Expect(Scanner.LPAREN);
	if P.tok != Scanner.RPAREN {
		P.ParseParameterList();
	}
	P.Expect(Scanner.RPAREN);

	P.Ecart();
}


func (P *Parser) ParseResult() {
	P.Trace("Result");

	if P.tok == Scanner.LPAREN {
		// one or more named results
		// TODO: here we allow empty returns - should proably fix this
		P.ParseParameters();

	} else {
		// anonymous result
		pos := P.pos;
		typ := P.TryType();
		if typ != nil {
			obj := Globals.NewObject(pos, Object.VAR, ".res");
			obj.typ = typ;
			P.Declare(obj);
		}
	}

	P.Ecart();
}


// Signatures
//
// (params)
// (params) type
// (params) (results)

func (P *Parser) ParseSignature() *Globals.Type {
	P.Trace("Signature");

	P.OpenScope();
	P.level--;
	sig := P.top_scope;

	P.ParseParameters();
	r0 := sig.entries.len;
	P.ParseResult();

	P.level++;
	P.CloseScope();

	P.Ecart();
	return MakeFunctionType(sig, 0, r0);
}


// Named signatures
//
//        ident (params)
//        ident (params) type
//        ident (params) (results)
// (recv) ident (params)
// (recv) ident (params) type
// (recv) ident (params) (results)

func (P *Parser) ParseNamedSignature() (pos int, ident string, typ *Globals.Type) {
	P.Trace("NamedSignature");

	P.OpenScope();
	P.level--;
	sig := P.top_scope;
	p0 := 0;

	if P.tok == Scanner.LPAREN {
		recv_pos := P.pos;
		P.ParseParameters();
		p0 = sig.entries.len;
		if p0 != 1 {
			print("p0 = ", p0, "\n");
			P.Error(recv_pos, "must have exactly one receiver");
			panic("UNIMPLEMENTED (ParseNamedSignature)");
			// TODO do something useful here
		}
	}

	pos, ident = P.ParseIdent(true);

	P.ParseParameters();

	r0 := sig.entries.len;
	P.ParseResult();
	P.level++;
	P.CloseScope();

	P.Ecart();
	return pos, ident, MakeFunctionType(sig, p0, r0);
}


func (P *Parser) ParseFunctionType() *Globals.Type {
	P.Trace("FunctionType");

	typ := P.ParseSignature();

	P.Ecart();
	return typ;
}


func (P *Parser) ParseMethodDecl(recv_typ *Globals.Type) {
	P.Trace("MethodDecl");

	pos, ident := P.ParseIdent(true);
	P.OpenScope();
	P.level--;
	sig := P.top_scope;

	// dummy receiver (give it a name so it won't conflict with unnamed result)
	recv := Globals.NewObject(pos, Object.VAR, ".recv");
	recv.typ = recv_typ;
	sig.Insert(recv);

	P.ParseParameters();

	r0 := sig.entries.len;
	P.ParseResult();
	P.level++;
	P.CloseScope();
	P.Optional(Scanner.SEMICOLON);

	obj := Globals.NewObject(pos, Object.FUNC, ident);
	obj.typ = MakeFunctionType(sig, 1, r0);
	P.Declare(obj);

	P.Ecart();
}


func (P *Parser) ParseInterfaceType() *Globals.Type {
	P.Trace("InterfaceType");

	P.Expect(Scanner.INTERFACE);
	P.Expect(Scanner.LBRACE);
	P.OpenScope();
	P.level--;
	typ := Globals.NewType(Type.INTERFACE);
	typ.scope = P.top_scope;
	for P.tok >= Scanner.IDENT {
		P.ParseMethodDecl(typ);
	}
	P.level++;
	P.CloseScope();
	P.Expect(Scanner.RBRACE);

	P.Ecart();
	return typ;
}


func (P *Parser) ParseMapType() *Globals.Type {
	P.Trace("MapType");

	P.Expect(Scanner.MAP);
	P.Expect(Scanner.LBRACK);
	typ := Globals.NewType(Type.MAP);
	typ.key = P.ParseVarType();
	P.Expect(Scanner.RBRACK);
	typ.elt = P.ParseVarType();
	P.Ecart();

	return typ;
}


func (P *Parser) ParseStructType() *Globals.Type {
	P.Trace("StructType");

	P.Expect(Scanner.STRUCT);
	P.Expect(Scanner.LBRACE);
	P.OpenScope();
	P.level--;
	typ := Globals.NewType(Type.STRUCT);
	typ.scope = P.top_scope;
	for P.tok >= Scanner.IDENT {
		P.ParseVarDeclList(Object.FIELD);
		if P.tok != Scanner.RBRACE {
			P.Expect(Scanner.SEMICOLON);
		}
	}
	P.Optional(Scanner.SEMICOLON);
	P.level++;
	P.CloseScope();
	P.Expect(Scanner.RBRACE);

	P.Ecart();
	return typ;
}


func (P *Parser) ParsePointerType() *Globals.Type {
	P.Trace("PointerType");

	P.Expect(Scanner.MUL);
	typ := Globals.NewType(Type.POINTER);

	var elt *Globals.Type;
	if P.tok == Scanner.STRING && !P.comp.flags.sixg {
		// implicit package.type forward declaration
		// TODO eventually the scanner should strip the quotes
		pkg_name := P.val[1 : len(P.val) - 1];  // strip quotes
		pkg := P.comp.Lookup(pkg_name);
		if pkg == nil {
			// package doesn't exist yet - add it to the package list
			obj := Globals.NewObject(P.pos, Object.PACKAGE, ".pkg");
			pkg = Globals.NewPackage(pkg_name, obj, Globals.NewScope(nil));
			pkg.key = "";  // mark as forward-declared package
			P.comp.Insert(pkg);
		} else {
			// package exists already - must be forward declaration
			if pkg.key != "" {
				P.Error(P.pos, `cannot use implicit package forward declaration for imported package "` + P.val + `"`);
				panic("wrong package forward decl");
				// TODO introduce dummy package so we can continue safely
			}
		}
		
		P.Next();  // consume package name
		P.Expect(Scanner.PERIOD);
		pos, ident := P.ParseIdent(false);
		obj := pkg.scope.Lookup(ident);
		if obj == nil {
			elt = Globals.NewType(Type.FORWARD);
			elt.scope = P.top_scope;  // not really needed here, but for consistency
			obj = Globals.NewObject(pos, Object.TYPE, ident);
			obj.exported = true;  // the type name must be visible
			obj.typ = elt;
			elt.obj = obj;  // primary type object;
			pkg.scope.Insert(obj);
			obj.pnolev = pkg.obj.pnolev;
		} else {
			if obj.kind != Object.TYPE || obj.typ.form != Type.FORWARD {
				panic("inconsistency in package.type forward declaration");
			}
			elt = obj.typ;
		}
		
	} else if P.tok == Scanner.IDENT {
		if P.Lookup(P.val) == nil {
			// implicit type forward declaration
			// create a named forward type 
			pos, ident := P.ParseIdent(false);
			obj := Globals.NewObject(pos, Object.TYPE, ident);
			elt = Globals.NewType(Type.FORWARD);
			obj.typ = elt;
			elt.obj = obj;  // primary type object;
			// remember the current scope - resolving the forward
			// type must find a matching declaration in this or a less nested scope
			elt.scope = P.top_scope;

				// create a named forward type

		} else {
			// type name
			// (ParseType() (via TryType()) checks for forward types and complains,
			// so call ParseTypeName() directly)
			// we can only have a foward type here if we refer to the name of a
			// yet incomplete type (i.e. if we are in the middle of a type's declaration)
			elt = P.ParseTypeName();
		}

		// collect uses of pointer types referring to forward types
		if elt.form == Type.FORWARD {
			P.forward_types.AddTyp(typ);
		}

	} else {
		elt = P.ParseType();
	}


	typ.elt = elt;

	P.Ecart();
	return typ;
}


// Returns nil if no type was found.
func (P *Parser) TryType() *Globals.Type {
	P.Trace("Type (try)");

	pos := P.pos;
	var typ *Globals.Type = nil;
	switch P.tok {
	case Scanner.IDENT: typ = P.ParseTypeName();
	case Scanner.LBRACK: typ = P.ParseArrayType();
	case Scanner.CHAN, Scanner.ARROW: typ = P.ParseChannelType();
	case Scanner.INTERFACE: typ = P.ParseInterfaceType();
	case Scanner.LPAREN: typ = P.ParseFunctionType();
	case Scanner.MAP: typ = P.ParseMapType();
	case Scanner.STRUCT: typ = P.ParseStructType();
	case Scanner.MUL: typ = P.ParsePointerType();
	}

	if typ != nil && typ.form == Type.FORWARD {
		P.Error(pos, "incomplete type");
	}

	P.Ecart();
	return typ;
}


// ----------------------------------------------------------------------------
// Blocks

func (P *Parser) ParseStatement() {
	P.Trace("Statement");
	if !P.TryStatement() {
		P.Error(P.pos, "statement expected");
		P.Next();  // make progress
	}
	P.Ecart();
}


func (P *Parser) ParseStatementList() {
	P.Trace("StatementList");
	for P.TryStatement() {
		P.Optional(Scanner.SEMICOLON);
	}
	P.Ecart();
}


func (P *Parser) ParseBlock(sig *Globals.Scope) {
	P.Trace("Block");

	P.Expect(Scanner.LBRACE);
	P.OpenScope();
	if sig != nil {
		P.level--;
		// add copies of the formal parameters to the function scope
		scope := P.top_scope;
		for p := sig.entries.first; p != nil; p = p.next {
			scope.Insert(p.obj.Copy())
		}
	}
	if P.tok != Scanner.RBRACE && P.tok != Scanner.SEMICOLON {
		P.ParseStatementList();
	}
	P.Optional(Scanner.SEMICOLON);
	if sig != nil {
		P.level++;
	}
	P.CloseScope();
	P.Expect(Scanner.RBRACE);

	P.Ecart();
}


// ----------------------------------------------------------------------------
// Expressions

func (P *Parser) ParseExpressionList(list *Globals.List) {
	P.Trace("ExpressionList");

	list.AddExpr(P.ParseExpression());
	for P.tok == Scanner.COMMA {
		P.Next();
		list.AddExpr(P.ParseExpression());
	}

	P.Ecart();
}


func (P *Parser) ParseNewExpressionList() *Globals.List {
	P.Trace("NewExpressionList");

	list := Globals.NewList();
	P.ParseExpressionList(list);

	P.Ecart();
	return list;
}


func (P *Parser) ParseFunctionLit() Globals.Expr {
	P.Trace("FunctionLit");

	P.Expect(Scanner.FUNC);
	typ := P.ParseFunctionType();
	P.ParseBlock(typ.scope);

	P.Ecart();
	return nil;
}


func (P *Parser) ParseExpressionPair(list *Globals.List) {
	P.Trace("ExpressionPair");

	list.AddExpr(P.ParseExpression());
	P.Expect(Scanner.COLON);
	list.AddExpr(P.ParseExpression());

	P.Ecart();
}


func (P *Parser) ParseExpressionPairList(list *Globals.List) {
	P.Trace("ExpressionPairList");

	P.ParseExpressionPair(list);
	for (P.tok == Scanner.COMMA) {
		P.ParseExpressionPair(list);
	}

	P.Ecart();
}


func (P *Parser) ParseCompositeLit(typ *Globals.Type) Globals.Expr {
	P.Trace("CompositeLit");

	P.Expect(Scanner.LBRACE);
	// TODO: should allow trailing ','
	list := Globals.NewList();
	if P.tok != Scanner.RBRACE {
		list.AddExpr(P.ParseExpression());
		if P.tok == Scanner.COMMA {
			P.Next();
			if P.tok != Scanner.RBRACE {
				P.ParseExpressionList(list);
			}
		} else if P.tok == Scanner.COLON {
			P.Next();
			list.AddExpr(P.ParseExpression());
			if P.tok == Scanner.COMMA {
				P.Next();
				if P.tok != Scanner.RBRACE {
					P.ParseExpressionPairList(list);
				}
			}
		}
	}
	P.Expect(Scanner.RBRACE);

	P.Ecart();
	return nil;
}


func (P *Parser) ParseOperand(pos int, ident string) Globals.Expr {
	P.Trace("Operand");

	if pos < 0 && P.tok == Scanner.IDENT {
		// no look-ahead yet
		pos = P.pos;
		ident = P.val;
		P.Next();
	}

	var res Globals.Expr = AST.Bad;

	if pos >= 0 {
		// we have an identifier
		obj := P.ParseQualifiedIdent(pos, ident);
		if obj.kind == Object.TYPE && P.tok == Scanner.LBRACE {
			res = P.ParseCompositeLit(obj.typ);
		} else {
			res = AST.NewObject(pos, obj);
		}

	} else {

		switch P.tok {
		case Scanner.IDENT:
			panic("UNREACHABLE");

		case Scanner.LPAREN:
			P.Next();
			res = P.ParseExpression();
			P.Expect(Scanner.RPAREN);

		case Scanner.INT:
			x := AST.NewLiteral(P.pos, Universe.int_t);
			x.i = 42;  // TODO set the right value
			res = x;
			P.Next();

		case Scanner.FLOAT:
			x := AST.NewLiteral(P.pos, Universe.float_t);
			x.f = 42.0;  // TODO set the right value
			res = x;
			P.Next();

		case Scanner.STRING:
			x := AST.NewLiteral(P.pos, Universe.string_t);
			x.s = P.val;  // TODO need to strip quotes, interpret string properly
			res = x;
			P.Next();

		case Scanner.FUNC:
			res = P.ParseFunctionLit();
		default:
			typ := P.TryType();
			if typ != nil {
				res = P.ParseCompositeLit(typ);
			} else {
				P.Error(P.pos, "operand expected");
				P.Next();  // make progress
			}
		}

	}

	P.Ecart();
	return res;
}


func (P *Parser) ParseSelectorOrTypeAssertion(x Globals.Expr) Globals.Expr {
	P.Trace("SelectorOrTypeAssertion");

	P.Expect(Scanner.PERIOD);
	pos := P.pos;
	
	if P.tok >= Scanner.IDENT {
		pos, selector := P.ParseIdent(true);
		x = Expr.Select(P.comp, x, pos, selector);
	} else {
		P.Expect(Scanner.LPAREN);
		typ := P.ParseType();
		P.Expect(Scanner.RPAREN);
		x = Expr.AssertType(P.comp, x, pos, typ);
	}

	P.Ecart();
	return x;
}


func (P *Parser) ParseIndexOrSlice(x Globals.Expr) Globals.Expr {
	P.Trace("IndexOrSlice");

	P.Expect(Scanner.LBRACK);
	i := P.ParseExpression();
	if P.tok == Scanner.COLON {
		P.Next();
		j := P.ParseExpression();
		x = Expr.Slice(P.comp, x, i, j);
	} else {
		x = Expr.Index(P.comp, x, i);
	}
	P.Expect(Scanner.RBRACK);

	P.Ecart();
	return x;
}


func (P *Parser) ParseCall(x Globals.Expr) Globals.Expr {
	P.Trace("Call");

	P.Expect(Scanner.LPAREN);
	args := Globals.NewList();
	if P.tok != Scanner.RPAREN {
		P.ParseExpressionList(args);
	}
	P.Expect(Scanner.RPAREN);
	x = Expr.Call(P.comp, x, args);

	P.Ecart();
	return x;
}


func (P *Parser) ParsePrimaryExpr(pos int, ident string) Globals.Expr {
	P.Trace("PrimaryExpr");

	x := P.ParseOperand(pos, ident);
	for {
		switch P.tok {
		case Scanner.PERIOD: x = P.ParseSelectorOrTypeAssertion(x);
		case Scanner.LBRACK: x = P.ParseIndexOrSlice(x);
		case Scanner.LPAREN: x = P.ParseCall(x);
		default: goto exit;
		}
	}

exit:
	P.Ecart();
	return x;
}


// TODO is this function needed?
func (P *Parser) ParsePrimaryExprList() *Globals.List {
	P.Trace("PrimaryExprList");

	list := Globals.NewList();
	list.AddExpr(P.ParsePrimaryExpr(-1, ""));
	for P.tok == Scanner.COMMA {
		P.Next();
		list.AddExpr(P.ParsePrimaryExpr(-1, ""));
	}

	P.Ecart();
	return list;
}


func (P *Parser) ParseUnaryExpr() Globals.Expr {
	P.Trace("UnaryExpr");

	switch P.tok {
	case Scanner.ADD: fallthrough;
	case Scanner.SUB: fallthrough;
	case Scanner.NOT: fallthrough;
	case Scanner.XOR: fallthrough;
	case Scanner.MUL: fallthrough;
	case Scanner.ARROW: fallthrough;
	case Scanner.AND:
		P.Next();
		x := P.ParseUnaryExpr();
		P.Ecart();
		return x;  // TODO fix this
	}
	
	x := P.ParsePrimaryExpr(-1, "");

	P.Ecart();
	return x;  // TODO fix this
}


func Precedence(tok int) int {
	// TODO should use a map or array here for lookup
	switch tok {
	case Scanner.LOR:
		return 1;
	case Scanner.LAND:
		return 2;
	case Scanner.ARROW:
		return 3;
	case Scanner.EQL, Scanner.NEQ, Scanner.LSS, Scanner.LEQ, Scanner.GTR, Scanner.GEQ:
		return 4;
	case Scanner.ADD, Scanner.SUB, Scanner.OR, Scanner.XOR:
		return 5;
	case Scanner.MUL, Scanner.QUO, Scanner.REM, Scanner.SHL, Scanner.SHR, Scanner.AND:
		return 6;
	}
	return 0;
}


func (P *Parser) ParseBinaryExpr(pos int, ident string, prec1 int) Globals.Expr {
	P.Trace("BinaryExpr");

	var x Globals.Expr;
	if pos >= 0 {
		x = P.ParsePrimaryExpr(pos, ident);
	} else {
		x = P.ParseUnaryExpr();
	}

	for prec := Precedence(P.tok); prec >= prec1; prec-- {
		for Precedence(P.tok) == prec {
			P.Next();
			y := P.ParseBinaryExpr(-1, "", prec + 1);
			x = Expr.BinaryExpr(P.comp, x, y);
		}
	}

	P.Ecart();
	return x;
}


// Expressions where the first token may be an identifier which has already
// been consumed. If the identifier is present, pos is the identifier position,
// otherwise pos must be < 0 (and ident is ignored).
func (P *Parser) ParseIdentExpression(pos int, ident string) Globals.Expr {
	P.Trace("IdentExpression");
	indent := P.indent;

	x := P.ParseBinaryExpr(pos, ident, 1);

	if indent != P.indent {
		panic("imbalanced tracing code (Expression)");
	}
	P.Ecart();
	return x;
}


func (P *Parser) ParseExpression() Globals.Expr {
	P.Trace("Expression");

	x := P.ParseIdentExpression(-1, "");

	P.Ecart();
	return x;
}


// ----------------------------------------------------------------------------
// Statements

func (P *Parser) ConvertToExprList(pos_list, ident_list, expr_list *Globals.List) {
	if pos_list.len != ident_list.len {
		panic("inconsistent lists");
	}
	for p, q := pos_list.first, ident_list.first; q != nil; p, q = p.next, q.next {
		pos, ident := p.val, q.str;
		obj := P.Lookup(ident);
		if obj == nil {
			P.Error(pos, `"` + ident + `" is not declared`);
			obj = Globals.NewObject(pos, Object.BAD, ident);
		}
		expr_list.AddExpr(AST.NewObject(pos, obj));
	}
	pos_list.Clear();
	ident_list.Clear();
}


func (P *Parser) ParseIdentOrExpr(pos_list, ident_list, expr_list *Globals.List) {
	P.Trace("IdentOrExpr");

	pos, ident := -1, "";
	just_ident := false;
	if expr_list.len == 0 /* only idents so far */ && P.tok == Scanner.IDENT {
		pos, ident = P.pos, P.val;
		P.Next();
		switch P.tok {
		case Scanner.COMMA,
			Scanner.COLON,
			Scanner.DEFINE,
			Scanner.ASSIGN,
			Scanner.ADD_ASSIGN,
			Scanner.SUB_ASSIGN,
			Scanner.MUL_ASSIGN,
			Scanner.QUO_ASSIGN,
			Scanner.REM_ASSIGN,
			Scanner.AND_ASSIGN,
			Scanner.OR_ASSIGN,
			Scanner.XOR_ASSIGN,
			Scanner.SHL_ASSIGN,
			Scanner.SHR_ASSIGN:
			// identifier is *not* part of a more complicated expression
			just_ident = true;
		}
	}

	if just_ident {
		pos_list.AddInt(pos);
		ident_list.AddStr(ident);
	} else {
		P.ConvertToExprList(pos_list, ident_list, expr_list);
		expr_list.AddExpr(P.ParseIdentExpression(pos, ident));
	}

	P.Ecart();
}


func (P *Parser) ParseIdentOrExprList() (pos_list, ident_list, expr_list *Globals.List) {
	P.Trace("IdentOrExprList");

	pos_list, ident_list = Globals.NewList(), Globals.NewList();  // "pairs" of (pos, ident)
	expr_list = Globals.NewList();
	
	P.ParseIdentOrExpr(pos_list, ident_list, expr_list);
	for P.tok == Scanner.COMMA {
		P.Next();
		P.ParseIdentOrExpr(pos_list, ident_list, expr_list);
	}

	P.Ecart();
	return pos_list, ident_list, expr_list;
}


// Compute the number of individual values provided by the expression list.
func (P *Parser) ListArity(list *Globals.List) int {
	if list.len == 1 {
		x := list.ExprAt(0);
		if x.op() == AST.CALL {
			panic("UNIMPLEMENTED");
		}
		return 1;
	} else {
		for p := list.first; p != nil; p = p.next {
			x := p.expr;
			if x.op() == AST.CALL {
				panic("UNIMPLEMENTED");
			}
		}
	}
	panic("UNREACHABLE");
}


func (P *Parser) ParseSimpleStat() {
	P.Trace("SimpleStat");

	// If we see an identifier, we don't know if it's part of a
	// label declaration, (multiple) variable declaration, assignment,
	// or simply an expression, without looking ahead.
	// Strategy: We parse an expression list, but simultaneously, as
	// long as possible, maintain a list of identifiers which is converted
	// into an expression list only if neccessary. The result of
	// ParseIdentOrExprList is a pair of non-empty lists of identfiers and
	// their respective source positions, or a non-empty list of expressions
	// (but not both).
	pos_list, ident_list, expr_list := P.ParseIdentOrExprList();

	switch P.tok {
	case Scanner.COLON:
		// label declaration
		if ident_list.len == 1 {
			obj := Globals.NewObject(pos_list.first.val, Object.LABEL, ident_list.first.str);
			P.Declare(obj);
		} else {
			P.Error(P.pos, "illegal label declaration");
		}
		P.Next();  // consume ":"

	case Scanner.DEFINE:
		// variable declaration
		if ident_list.len == 0 {
			P.Error(P.pos, "illegal left-hand side for declaration");
		}
		P.Next();  // consume ":="
		val_list := P.ParseNewExpressionList();
		if val_list.len != ident_list.len {
			P.Error(val_list.first.expr.pos(), "number of expressions does not match number of variables");
		}
		// declare variables
		for p, q := pos_list.first, ident_list.first; q != nil; p, q = p.next, q.next {
			obj := Globals.NewObject(p.val, Object.VAR, q.str);
			P.Declare(obj);
			// TODO set correct types
			obj.typ = Universe.bad_t;  // for now
		}

	case Scanner.ASSIGN: fallthrough;
	case Scanner.ADD_ASSIGN: fallthrough;
	case Scanner.SUB_ASSIGN: fallthrough;
	case Scanner.MUL_ASSIGN: fallthrough;
	case Scanner.QUO_ASSIGN: fallthrough;
	case Scanner.REM_ASSIGN: fallthrough;
	case Scanner.AND_ASSIGN: fallthrough;
	case Scanner.OR_ASSIGN: fallthrough;
	case Scanner.XOR_ASSIGN: fallthrough;
	case Scanner.SHL_ASSIGN: fallthrough;
	case Scanner.SHR_ASSIGN:
		P.ConvertToExprList(pos_list, ident_list, expr_list);
		P.Next();
		pos := P.pos;
		val_list := P.ParseNewExpressionList();
		
		// assign variables
		if val_list.len == 1 && val_list.first.expr.typ().form == Type.TUPLE {
			panic("UNIMPLEMENTED");
		} else {
			var p, q *Globals.Elem;
			for p, q = expr_list.first, val_list.first; p != nil && q != nil; p, q = p.next, q.next {
				
			}
			if p != nil || q != nil {
				P.Error(pos, "number of expressions does not match number of variables");
			}
		}

	default:
		P.ConvertToExprList(pos_list, ident_list, expr_list);
		if expr_list.len != 1 {
			P.Error(P.pos, "no expression list allowed");
		}
		if P.tok == Scanner.INC || P.tok == Scanner.DEC {
			P.Next();
		}
	}

	P.Ecart();
}


func (P *Parser) ParseGoStat() {
	P.Trace("GoStat");

	P.Expect(Scanner.GO);
	P.ParseExpression();

	P.Ecart();
}


func (P *Parser) ParseReturnStat() {
	P.Trace("ReturnStat");

	P.Expect(Scanner.RETURN);
	res := Globals.NewList();
	if P.tok != Scanner.SEMICOLON && P.tok != Scanner.RBRACE {
		P.ParseExpressionList(res);
	}

	P.Ecart();
}


func (P *Parser) ParseControlFlowStat(tok int) {
	P.Trace("ControlFlowStat");

	P.Expect(tok);
	if P.tok == Scanner.IDENT {
		P.ParseIdent(false);
	}

	P.Ecart();
}


func (P *Parser) ParseIfStat() *AST.IfStat {
	P.Trace("IfStat");

	P.Expect(Scanner.IF);
	P.OpenScope();
	if P.tok != Scanner.LBRACE {
		if P.tok != Scanner.SEMICOLON {
			P.ParseSimpleStat();
		}
		if P.tok == Scanner.SEMICOLON {
			P.Next();
			if P.tok != Scanner.LBRACE {
				P.ParseExpression();
			}
		}
	}
	P.ParseBlock(nil);
	if P.tok == Scanner.ELSE {
		P.Next();
		if P.tok == Scanner.IF {
			P.ParseIfStat();
		} else {
			// TODO should be P.ParseBlock()
			P.ParseStatement();
		}
	}
	P.CloseScope();

	P.Ecart();
	return nil;
}


func (P *Parser) ParseForStat() {
	P.Trace("ForStat");

	P.Expect(Scanner.FOR);
	P.OpenScope();
	if P.tok != Scanner.LBRACE {
		if P.tok != Scanner.SEMICOLON {
			P.ParseSimpleStat();
		}
		if P.tok == Scanner.SEMICOLON {
			P.Next();
			if P.tok != Scanner.SEMICOLON {
				P.ParseExpression();
			}
			P.Expect(Scanner.SEMICOLON);
			if P.tok != Scanner.LBRACE {
				P.ParseSimpleStat();
			}
		}
	}
	P.ParseBlock(nil);
	P.CloseScope();

	P.Ecart();
}


func (P *Parser) ParseCase() {
	P.Trace("Case");

	if P.tok == Scanner.CASE {
		P.Next();
		list := Globals.NewList();
		P.ParseExpressionList(list);
	} else {
		P.Expect(Scanner.DEFAULT);
	}
	P.Expect(Scanner.COLON);

	P.Ecart();
}


func (P *Parser) ParseCaseClause() {
	P.Trace("CaseClause");

	P.ParseCase();
	if P.tok != Scanner.FALLTHROUGH && P.tok != Scanner.RBRACE {
		P.ParseStatementList();
		P.Optional(Scanner.SEMICOLON);
	}
	if P.tok == Scanner.FALLTHROUGH {
		P.Next();
		P.Optional(Scanner.SEMICOLON);
	}

	P.Ecart();
}


func (P *Parser) ParseSwitchStat() {
	P.Trace("SwitchStat");

	P.Expect(Scanner.SWITCH);
	P.OpenScope();
	if P.tok != Scanner.LBRACE {
		if P.tok != Scanner.SEMICOLON {
			P.ParseSimpleStat();
		}
		if P.tok == Scanner.SEMICOLON {
			P.Next();
			if P.tok != Scanner.LBRACE {
				P.ParseExpression();
			}
		}
	}
	P.Expect(Scanner.LBRACE);
	for P.tok == Scanner.CASE || P.tok == Scanner.DEFAULT {
		P.ParseCaseClause();
	}
	P.Expect(Scanner.RBRACE);
	P.CloseScope();

	P.Ecart();
}


func (P *Parser) ParseCommCase() {
  P.Trace("CommCase");

  if P.tok == Scanner.CASE {
	P.Next();
	if P.tok == Scanner.GTR {
		// send
		P.Next();
		P.ParseExpression();
		P.Expect(Scanner.EQL);
		P.ParseExpression();
	} else {
		// receive
		if P.tok != Scanner.LSS {
			P.ParseIdent(false);
			P.Expect(Scanner.ASSIGN);
		}
		P.Expect(Scanner.LSS);
		P.ParseExpression();
	}
  } else {
	P.Expect(Scanner.DEFAULT);
  }
  P.Expect(Scanner.COLON);

  P.Ecart();
}


func (P *Parser) ParseCommClause() {
	P.Trace("CommClause");

	P.ParseCommCase();
	if P.tok != Scanner.CASE && P.tok != Scanner.DEFAULT && P.tok != Scanner.RBRACE {
		P.ParseStatementList();
		P.Optional(Scanner.SEMICOLON);
	}

	P.Ecart();
}


func (P *Parser) ParseRangeStat() {
	P.Trace("RangeStat");

	P.Expect(Scanner.RANGE);
	P.ParseIdentList();
	P.Expect(Scanner.DEFINE);
	P.ParseExpression();
	P.ParseBlock(nil);

	P.Ecart();
}


func (P *Parser) ParseSelectStat() {
	P.Trace("SelectStat");

	P.Expect(Scanner.SELECT);
	P.Expect(Scanner.LBRACE);
	for P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
		P.ParseCommClause();
	}
	P.Next();

	P.Ecart();
}


func (P *Parser) TryStatement() bool {
	P.Trace("Statement (try)");
	indent := P.indent;

	res := true;
	switch P.tok {
	case Scanner.CONST: fallthrough;
	case Scanner.TYPE: fallthrough;
	case Scanner.VAR:
		P.ParseDeclaration();
	case Scanner.FUNC:
		// for now we do not allow local function declarations
		fallthrough;
	case Scanner.MUL, Scanner.ARROW, Scanner.IDENT, Scanner.LPAREN:
		P.ParseSimpleStat();
	case Scanner.GO:
		P.ParseGoStat();
	case Scanner.RETURN:
		P.ParseReturnStat();
	case Scanner.BREAK, Scanner.CONTINUE, Scanner.GOTO:
		P.ParseControlFlowStat(P.tok);
	case Scanner.LBRACE:
		P.ParseBlock(nil);
	case Scanner.IF:
		P.ParseIfStat();
	case Scanner.FOR:
		P.ParseForStat();
	case Scanner.SWITCH:
		P.ParseSwitchStat();
	case Scanner.RANGE:
		P.ParseRangeStat();
	case Scanner.SELECT:
		P.ParseSelectStat();
	default:
		// no statement found
		res = false;
	}

	if indent != P.indent {
		panic("imbalanced tracing code (Statement)");
	}
	P.Ecart();
	return res;
}


// ----------------------------------------------------------------------------
// Declarations

func (P *Parser) ParseImportSpec() {
	P.Trace("ImportSpec");

	var obj *Globals.Object = nil;
	if P.tok == Scanner.PERIOD {
		P.Error(P.pos, `"import ." not yet handled properly`);
		P.Next();
	} else if P.tok == Scanner.IDENT {
		obj = P.ParseIdentDecl(Object.PACKAGE);
	}

	if P.tok == Scanner.STRING {
		// TODO eventually the scanner should strip the quotes
		pkg_name := P.val[1 : len(P.val) - 1];  // strip quotes
		pkg := P.comp.env.Import(P.comp, pkg_name);
		if pkg != nil {
			pno := pkg.obj.pnolev;  // preserve pno
			if obj == nil {
				// use original package name
				obj = pkg.obj;
				P.Declare(obj);  // this changes (pkg.)obj.pnolev!
			}
			obj.pnolev = pno;  // reset pno
		} else {
			P.Error(P.pos, `import of "` + pkg_name + `" failed`);
		}
		P.Next();
	} else {
		P.Expect(Scanner.STRING);  // use Expect() error handling
	}

	P.Ecart();
}


func (P *Parser) ParseConstSpec(exported bool) {
	P.Trace("ConstSpec");

	list := P.ParseIdentDeclList(Object.CONST);
	typ := P.TryType();
	if typ != nil {
		for p := list.first; p != nil; p = p.next {
			p.obj.typ = typ;
		}
	}

	if P.tok == Scanner.ASSIGN {
		P.Next();
		P.ParseNewExpressionList();
	}

	if exported {
		for p := list.first; p != nil; p = p.next {
			p.obj.exported = true;
		}
	}

	P.Ecart();
}


func (P *Parser) ParseTypeSpec(exported bool) {
	P.Trace("TypeSpec");

	var typ *Globals.Type;

	pos, ident := P.ParseIdent(false);
	obj := P.Lookup(ident);

	if !P.comp.flags.sixg && obj != nil {
		if obj.typ.form == Type.FORWARD {
			// imported forward-declared type
			if !exported {
				panic("foo");
			}
		} else {
			panic("bar");
		}

	} else {
		// Immediately after declaration of the type name, the type is
		// considered forward-declared. It may be referred to from inside
		// the type specification only via a pointer type.
		typ = Globals.NewType(Type.FORWARD);
		typ.scope = P.top_scope;  // not really needed here, but for consistency

		obj = Globals.NewObject(pos, Object.TYPE, ident);
		obj.exported = exported;
		obj.typ = typ;
		typ.obj = obj;  // primary type object
		P.Declare(obj);
	}

	// If the next token is an identifier and we have a legal program,
	// it must be a typename. In that case this declaration introduces
	// an alias type.
	if P.tok == Scanner.IDENT {
		typ = Globals.NewType(Type.ALIAS);
		elt := P.ParseType();  // we want a complete type - don't shortcut to ParseTypeName()
		typ.elt = elt;
		if elt.form == Type.ALIAS {
			typ.key = elt.key;  // the base type
		} else {
			typ.key = elt;
		}
	} else {
		typ = P.ParseType();
	}

	obj.typ = typ;
	if typ.obj == nil {
		typ.obj = obj;  // primary type object
	}

	// if the type is exported, for now we export all fields
	// of structs and interfaces by default
	// TODO this needs to change eventually
	// Actually in 6g even types referred to are exported - sigh...
	if exported && (typ.form == Type.STRUCT || typ.form == Type.INTERFACE) {
		for p := typ.scope.entries.first; p != nil; p = p.next {
			p.obj.exported = true;
		}
	}

	P.Ecart();
}


func (P *Parser) ParseVarSpec(exported bool) {
	P.Trace("VarSpec");

	list := P.ParseIdentDeclList(Object.VAR);
	if P.tok == Scanner.ASSIGN {
		P.Next();
		P.ParseNewExpressionList();
	} else {
		typ := P.ParseVarType();
		for p := list.first; p != nil; p = p.next {
			p.obj.typ = typ;
		}
		if P.tok == Scanner.ASSIGN {
			P.Next();
			P.ParseNewExpressionList();
		}
	}

	if exported {
		for p := list.first; p != nil; p = p.next {
			p.obj.exported = true;
		}
	}

	P.Ecart();
}


// TODO With method variables, we wouldn't need this dispatch function.
func (P *Parser) ParseSpec(exported bool, keyword int) {
	switch keyword {
	case Scanner.IMPORT: P.ParseImportSpec();
	case Scanner.CONST: P.ParseConstSpec(exported);
	case Scanner.TYPE: P.ParseTypeSpec(exported);
	case Scanner.VAR: P.ParseVarSpec(exported);
	default: panic("UNREACHABLE");
	}
}


func (P *Parser) ParseDecl(exported bool, keyword int) {
	P.Trace("Decl");

	P.Expect(keyword);
	if P.tok == Scanner.LPAREN {
		P.Next();
		for P.tok == Scanner.IDENT {
			P.ParseSpec(exported, keyword);
			if P.tok != Scanner.RPAREN {
				// P.Expect(Scanner.SEMICOLON);
				P.Optional(Scanner.SEMICOLON);  // TODO this seems wrong! (needed for math.go)
			}
		}
		P.Next();
	} else {
		P.ParseSpec(exported, keyword);
	}

	P.Ecart();
}


func (P *Parser) ParseFuncDecl(exported bool) {
	P.Trace("FuncDecl");

	P.Expect(Scanner.FUNC);
	pos, ident, typ := P.ParseNamedSignature();
	obj := P.DeclareFunc(pos, ident, typ);  // need obj later for statements
	obj.exported = exported;
	if P.tok == Scanner.SEMICOLON {
		// forward declaration
		P.Next();
	} else {
		P.ParseBlock(typ.scope);
	}

	P.Ecart();
}


func (P *Parser) ParseExportDecl() {
	P.Trace("ExportDecl");

	// TODO This is deprecated syntax and should go away eventually.
	// (Also at the moment the syntax is everything goes...)
	//P.Expect(Scanner.EXPORT);

	if !P.comp.flags.sixg {
		P.Error(P.pos, "deprecated export syntax (use -6g to enable)");
	}

	has_paren := false;
	if P.tok == Scanner.LPAREN {
		P.Next();
		has_paren = true;
	}
	for P.tok == Scanner.IDENT {
		pos, ident := P.ParseIdent(false);
		P.exports.AddStr(ident);
		P.Optional(Scanner.COMMA);  // TODO this seems wrong
	}
	if has_paren {
		P.Expect(Scanner.RPAREN)
	}

	P.Ecart();
}


func (P *Parser) ParseDeclaration() {
	P.Trace("Declaration");
	indent := P.indent;

	exported := false;
	if P.tok == Scanner.EXPORT {
		if P.level == 0 {
			exported = true;
		} else {
			P.Error(P.pos, "local declarations cannot be exported");
		}
		P.Next();
	}

	switch P.tok {
	case Scanner.CONST, Scanner.TYPE, Scanner.VAR:
		P.ParseDecl(exported, P.tok);
	case Scanner.FUNC:
		P.ParseFuncDecl(exported);
	case Scanner.EXPORT:
		if exported {
			P.Error(P.pos, "cannot mark export declaration for export");
		}
		P.Next();
		P.ParseExportDecl();
	default:
		if exported && (P.tok == Scanner.IDENT || P.tok == Scanner.LPAREN) {
			P.ParseExportDecl();
		} else {
			P.Error(P.pos, "declaration expected");
			P.Next();  // make progress
		}
	}

	if indent != P.indent {
		panic("imbalanced tracing code (Declaration)");
	}
	P.Ecart();
}


// ----------------------------------------------------------------------------
// Program

func (P *Parser) ResolveForwardTypes() {
	for p := P.forward_types.first; p != nil; p = p.next {
		typ := p.typ;
		if typ.form != Type.POINTER {
			panic("unresolved types should be pointers only");
		}

		elt := typ.elt;
		if typ.elt.form != Type.FORWARD {
			panic("unresolved pointer should point to forward type");
		}

		obj := elt.obj;
		if obj.typ == elt {
			// actual forward declaration (as opposed to forward types introduced
			// during type declaration) - need to lookup the actual type object
			var elt_obj *Globals.Object;
			for scope := elt.scope; scope != nil && elt_obj == nil; scope = scope.parent {
				elt_obj = scope.Lookup(obj.ident);
			}
			// update the type object if we found one
			if elt_obj != nil {
				if elt_obj.kind == Object.TYPE {
					obj = elt_obj;
				} else {
					P.Error(obj.pos, `"` + obj.ident + `" does not denote a type`);
				}
			}
		}

		// update the pointer type
		typ.elt = obj.typ;

		// TODO as long as we don't *use* a forward type, we are ok
		// => consider not reporting this as an error
		// (in a real forward declaration, the corresponding objects are not in a scope
		// and have incorrect pnolev)
		if typ.elt.form == Type.FORWARD {
			P.Error(obj.pos, `"` + obj.ident + `" is not declared after forward declaration`);
		}
	}
}


func (P *Parser) MarkExports() {
	scope := P.top_scope;
	for p := P.exports.first; p != nil; p = p.next {
		obj := scope.Lookup(p.str);
		if obj != nil {
			obj.exported = true;
			// For now we export deep
			// TODO this should change eventually - we need selective export
			if obj.kind == Object.TYPE {
				typ := obj.typ;
				if typ.form == Type.STRUCT || typ.form == Type.INTERFACE {
					scope := typ.scope;
					for p := scope.entries.first; p != nil; p = p.next {
						p.obj.exported = true;
					}
				}
			}
		} else {
			// TODO need to report proper src position
			P.Error(-1, `"` + p.str + `" is not declared - cannot be exported`);
		}
	}
}


func (P *Parser) ParseProgram() {
	P.Trace("Program");

	P.OpenScope();
	P.Expect(Scanner.PACKAGE);
	obj := P.ParseIdentDecl(Object.PACKAGE);
	P.Optional(Scanner.SEMICOLON);

	{	P.OpenScope();
		if P.level != 0 {
			panic("incorrect scope level");
		}

		P.comp.Insert(Globals.NewPackage(P.scanner.filename, obj, P.top_scope));
		if P.comp.pkg_ref != 1 {
			panic("should have exactly one package now");
		}

		if P.comp.flags.sixg {
			// automatically import package sys
			pkg := P.comp.env.Import(P.comp, "sys");
			if pkg != nil {
				pno := pkg.obj.pnolev;  // preserve pno
				P.Declare(pkg.obj);  // this changes pkg.obj.pnolev!
				pkg.obj.pnolev = pno;  // reset pno
			} else {
				P.Error(P.pos, `pre-import of package "sys" failed`);
			}
		}
		
		for P.tok == Scanner.IMPORT {
			P.ParseDecl(false, Scanner.IMPORT);
			P.Optional(Scanner.SEMICOLON);
		}

		for P.tok != Scanner.EOF {
			P.ParseDeclaration();
			P.Optional(Scanner.SEMICOLON);
		}

		P.ResolveForwardTypes();
		P.MarkExports();

		if P.level != 0 {
			panic("incorrect scope level");
		}
		P.CloseScope();
	}

	P.CloseScope();
	P.Ecart();
}
