// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Parser

import Scanner "scanner"
import Globals "globals"
import Object "object"
import Type "type"
import Universe "universe"


// So I can submit and have a running parser for now...
const EnableSemanticTests = false;


export Parser
type Parser struct {
	comp *Globals.Compilation;
	verbose, indent int;
	S *Scanner.Scanner;
	tok int;  // one token look-ahead
	beg, end int;  // token position
	ident string;  // last ident seen
	top_scope *Globals.Scope;
	exports *Globals.List;
}


// ----------------------------------------------------------------------------
// Support functions

func (P *Parser) PrintIndent() {
	for i := P.indent; i > 0; i-- {
		print ". ";
	}
}


func (P *Parser) Trace(msg string) {
	if P.verbose > 0 {
		P.PrintIndent();
		print msg, " {\n";
		P.indent++;
	}
}


func (P *Parser) Ecart() {
	if P.verbose > 0 {
		P.indent--;
		P.PrintIndent();
		print "}\n";
	}
}


func (P *Parser) Next() {
	P.tok, P.beg, P.end = P.S.Scan();
	if P.tok == Scanner.IDENT {
		P.ident = P.S.src[P.beg : P.end];
	}
	if P.verbose > 1 {
		P.PrintIndent();
		print "[", P.beg, "] ", Scanner.TokenName(P.tok), "\n";
	}
}


func (P *Parser) Open(comp *Globals.Compilation, S *Scanner.Scanner, verbose int) {
	P.comp = comp;
	P.verbose = verbose;
	P.indent = 0;
	P.S = S;
	P.Next();
	P.top_scope = Universe.scope;
	P.exports = Globals.NewList();
}


func (P *Parser) Error(pos int, msg string) {
	P.S.Error(pos, msg);
}


func (P *Parser) Expect(tok int) {
	if P.tok != tok {
		P.Error(P.beg, "expected '" + Scanner.TokenName(tok) + "', found '" + Scanner.TokenName(P.tok) + "'");
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
	if !EnableSemanticTests {
		return;
	}
	
	if scope.Lookup(obj.ident) != nil {
		// TODO is this the correct error position?
		P.Error(obj.pos, `"` + obj.ident + `" is declared already`);
		return;  // don't insert it into the scope
	}
	scope.Insert(obj);
}


func (P *Parser) Declare(obj *Globals.Object) {
	P.DeclareInScope(P.top_scope, obj);
}


// ----------------------------------------------------------------------------
// Common productions


func (P *Parser) TryType() *Globals.Type;
func (P *Parser) ParseExpression();
func (P *Parser) TryStatement() bool;
func (P *Parser) ParseDeclaration();


func (P *Parser) ParseIdent() string {
	P.Trace("Ident");

	ident := "";
	if P.tok == Scanner.IDENT {
		ident = P.ident;
		if P.verbose > 0 {
			P.PrintIndent();
			print "Ident = \"", ident, "\"\n";
		}
		P.Next();
	} else {
		P.Expect(Scanner.IDENT);  // use Expect() error handling
	}
	
	P.Ecart();
	return ident;
}


func (P *Parser) ParseIdentDecl(kind int) *Globals.Object {
	P.Trace("IdentDecl");
	
	pos := P.beg;
	obj := Globals.NewObject(pos, kind, P.ParseIdent());
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
	P.ParseIdent();
	for P.tok == Scanner.COMMA {
		P.Next();
		P.ParseIdent();
	}
	P.Ecart();
}


func (P *Parser) ParseQualifiedIdent() *Globals.Object {
	P.Trace("QualifiedIdent");

	if EnableSemanticTests {
		pos := P.beg;
		ident := P.ParseIdent();
		obj := P.Lookup(ident);
		if obj == nil {
			P.Error(pos, `"` + ident + `" is not declared`);
			obj = Globals.NewObject(pos, Object.BAD, ident);
		}

		if obj.kind == Object.PACKAGE && P.tok == Scanner.PERIOD {
			panic "Qualified ident not complete yet";
			P.Next();
			P.ParseIdent();
		}
		P.Ecart();
		return obj;
		
	} else {
		P.ParseIdent();
		if P.tok == Scanner.PERIOD {
			P.Next();
			P.ParseIdent();
		}
		P.Ecart();
		return nil;
	}
}


// ----------------------------------------------------------------------------
// Types

func (P *Parser) ParseType() *Globals.Type{
	P.Trace("Type");
	
	typ := P.TryType();
	if typ == nil {
		P.Error(P.beg, "type expected");
		typ = Universe.bad_t;
	}
	
	P.Ecart();
	return typ;
}


func (P *Parser) ParseTypeName() *Globals.Type {
	P.Trace("TypeName");
	
	if EnableSemanticTests {
		obj := P.ParseQualifiedIdent();
		typ := obj.typ;
		if obj.kind != Object.TYPE {
			P.Error(obj.pos, `"` + obj.ident + `" is not a type`);
			typ = Universe.bad_t;
		}
		P.Ecart();
		return typ;
	} else {
		P.ParseQualifiedIdent();
		P.Ecart();
		return Universe.bad_t;
	}
}


func (P *Parser) ParseArrayType() *Globals.Type {
	P.Trace("ArrayType");
	P.Expect(Scanner.LBRACK);
	if P.tok != Scanner.RBRACK {
		P.ParseExpression();
	}
	P.Expect(Scanner.RBRACK);
	P.ParseType();
	P.Ecart();
	return Universe.bad_t;
}


func (P *Parser) ParseChannelType() *Globals.Type {
	P.Trace("ChannelType");
	P.Expect(Scanner.CHAN);
	switch P.tok {
	case Scanner.SEND: fallthrough
	case Scanner.RECV:
		P.Next();
	}
	P.ParseType();
	P.Ecart();
	return Universe.bad_t;
}


func (P *Parser) ParseParameterSection() {
	P.Trace("ParameterSection");
	P.ParseIdentList();
	P.ParseType();
	P.Ecart();
}


func (P *Parser) ParseParameterList() {
	P.Trace("ParameterList");
	P.ParseParameterSection();
	for P.tok == Scanner.COMMA {
		P.Next();
		P.ParseParameterSection();
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


func (P *Parser) TryResult() bool {
	P.Trace("Result (try)");
	res := false;
	if P.tok == Scanner.LPAREN {
		// TODO: here we allow empty returns - should proably fix this
		P.ParseParameters();
		res = true;
	} else {
		res = P.TryType() != nil;
	}
	P.Ecart();
	return res;
}


// Anonymous signatures
//
//          (params)
//          (params) type
//          (params) (results)
// (recv) . (params)
// (recv) . (params) type
// (recv) . (params) (results)

func (P *Parser) ParseAnonymousSignature() {
	P.Trace("AnonymousSignature");
	P.OpenScope();
	P.ParseParameters();
	if P.tok == Scanner.PERIOD {
		P.Next();
		P.ParseParameters();
	}
	P.TryResult();
	P.CloseScope();
	P.Ecart();
}


// Named signatures
//
//        name (params)
//        name (params) type
//        name (params) (results)
// (recv) name (params)
// (recv) name (params) type
// (recv) name (params) (results)

func (P *Parser) ParseNamedSignature() {
	P.Trace("NamedSignature");
	P.OpenScope();
	if P.tok == Scanner.LPAREN {
		P.ParseParameters();
	}
	P.ParseIdent();  // function name
	P.ParseParameters();
	P.TryResult();
	P.CloseScope();
	P.Ecart();
}


func (P *Parser) ParseFunctionType() *Globals.Type {
	P.Trace("FunctionType");
	P.Expect(Scanner.FUNC);
	P.ParseAnonymousSignature();
	P.Ecart();
	return Universe.bad_t;
}


func (P *Parser) ParseMethodDecl() {
	P.Trace("MethodDecl");
	P.ParseIdent();
	P.ParseParameters();
	P.TryResult();
	P.Optional(Scanner.SEMICOLON);
	P.Ecart();
}


func (P *Parser) ParseInterfaceType() *Globals.Type {
	P.Trace("InterfaceType");
	P.Expect(Scanner.INTERFACE);
	P.Expect(Scanner.LBRACE);
	P.OpenScope();
	for P.tok != Scanner.RBRACE {
		P.ParseMethodDecl();
	}
	P.CloseScope();
	P.Next();
	P.Ecart();
	return Universe.bad_t;
}


func (P *Parser) ParseMapType() *Globals.Type {
	P.Trace("MapType");
	P.Expect(Scanner.MAP);
	P.Expect(Scanner.LBRACK);
	P.ParseType();
	P.Expect(Scanner.RBRACK);
	P.ParseType();
	P.Ecart();
	return Universe.bad_t;
}


func (P *Parser) ParseFieldDecl() {
	P.Trace("FieldDecl");
	
	list := P.ParseIdentDeclList(Object.VAR);
	typ := P.ParseType();  // TODO should check completeness of types
	for p := list.first; p != nil; p = p.next {
		p.obj.typ = typ;  // TODO should use/have set_type()
	}
	
	P.Ecart();
}


func (P *Parser) ParseStructType() *Globals.Type {
	P.Trace("StructType");
	
	P.Expect(Scanner.STRUCT);
	P.Expect(Scanner.LBRACE);
	P.OpenScope();
	typ := Globals.NewType(Type.STRUCT);
	typ.scope = P.top_scope;
	for P.tok == Scanner.IDENT {
		P.ParseFieldDecl();
		if P.tok != Scanner.RBRACE {
			P.Expect(Scanner.SEMICOLON);
		}
	}
	P.Optional(Scanner.SEMICOLON);
	P.CloseScope();
	P.Expect(Scanner.RBRACE);
	
	P.Ecart();
	return typ;
}


func (P *Parser) ParsePointerType() *Globals.Type {
	P.Trace("PointerType");
	P.Expect(Scanner.MUL);
	P.ParseType();
	P.Ecart();
	return Universe.bad_t;
}


// Returns nil if no type was found.
func (P *Parser) TryType() *Globals.Type {
	P.Trace("Type (try)");
	
	var typ *Globals.Type = nil;
	switch P.tok {
	case Scanner.IDENT: typ = P.ParseTypeName();
	case Scanner.LBRACK: typ = P.ParseArrayType();
	case Scanner.CHAN: typ = P.ParseChannelType();
	case Scanner.INTERFACE: typ = P.ParseInterfaceType();
	case Scanner.FUNC: typ = P.ParseFunctionType();
	case Scanner.MAP: typ = P.ParseMapType();
	case Scanner.STRUCT: typ = P.ParseStructType();
	case Scanner.MUL: typ = P.ParsePointerType();
	}

	P.Ecart();
	return typ;
}


// ----------------------------------------------------------------------------
// Blocks

func (P *Parser) ParseStatement() {
	P.Trace("Statement");
	if !P.TryStatement() {
		P.Error(P.beg, "statement expected");
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


func (P *Parser) ParseBlock() {
	P.Trace("Block");
	P.Expect(Scanner.LBRACE);
	P.OpenScope();
	if P.tok != Scanner.RBRACE && P.tok != Scanner.SEMICOLON {
		P.ParseStatementList();
	}
	P.Optional(Scanner.SEMICOLON);
	P.CloseScope();
	P.Expect(Scanner.RBRACE);
	P.Ecart();
}


// ----------------------------------------------------------------------------
// Expressions

func (P *Parser) ParseExpressionList() {
	P.Trace("ExpressionList");
	P.ParseExpression();
	for P.tok == Scanner.COMMA {
		P.Next();
		P.ParseExpression();
	}
	P.Ecart();
}


func (P *Parser) ParseNew() {
	P.Trace("New");
	P.Expect(Scanner.NEW);
	P.Expect(Scanner.LPAREN);
	P.ParseType();
	if P.tok == Scanner.COMMA {
		P.Next();
		P.ParseExpressionList()
	}
	P.Expect(Scanner.RPAREN);
	P.Ecart();
}


func (P *Parser) ParseFunctionLit() {
	P.Trace("FunctionLit");
	P.ParseFunctionType();
	P.ParseBlock();
	P.Ecart();
}


func (P *Parser) ParseOperand() {
	P.Trace("Operand");
	switch P.tok {
	case Scanner.IDENT:
		P.ParseQualifiedIdent();
	case Scanner.LPAREN:
		P.Next();
		P.ParseExpression();
		P.Expect(Scanner.RPAREN);
	case Scanner.STRING: fallthrough;
	case Scanner.NUMBER: fallthrough;
	case Scanner.NIL: fallthrough;
	case Scanner.IOTA: fallthrough;
	case Scanner.TRUE: fallthrough;
	case Scanner.FALSE:
		P.Next();
	case Scanner.FUNC:
		P.ParseFunctionLit();
	case Scanner.NEW:
		P.ParseNew();
	default:
		P.Error(P.beg, "operand expected");
		P.Next();  // make progress
	}
	P.Ecart();
}


func (P *Parser) ParseSelectorOrTypeAssertion() {
	P.Trace("SelectorOrTypeAssertion");
	P.Expect(Scanner.PERIOD);
	if P.tok == Scanner.IDENT {
		P.ParseIdent();
	} else {
		P.Expect(Scanner.LPAREN);
		P.ParseType();
		P.Expect(Scanner.RPAREN);
	}
	P.Ecart();
}


func (P *Parser) ParseIndexOrSlice() {
	P.Trace("IndexOrSlice");
	P.Expect(Scanner.LBRACK);
	P.ParseExpression();
	if P.tok == Scanner.COLON {
		P.Next();
		P.ParseExpression();
	}
	P.Expect(Scanner.RBRACK);
	P.Ecart();
}


func (P *Parser) ParseInvocation() {
	P.Trace("Invocation");
	P.Expect(Scanner.LPAREN);
	if P.tok != Scanner.RPAREN {
		P.ParseExpressionList();
	}
	P.Expect(Scanner.RPAREN);
	P.Ecart();
}


func (P *Parser) ParsePrimaryExpr() {
	P.Trace("PrimaryExpr");
	P.ParseOperand();
	for {
		switch P.tok {
		case Scanner.PERIOD:
			P.ParseSelectorOrTypeAssertion();
		case Scanner.LBRACK:
			P.ParseIndexOrSlice();
		case Scanner.LPAREN:
			P.ParseInvocation();
		default:
			P.Ecart();
			return;
		}
	}
	P.Ecart();
}


func (P *Parser) ParsePrimaryExprList() {
	P.Trace("PrimaryExprList");
	P.ParsePrimaryExpr();
	for P.tok == Scanner.COMMA {
		P.Next();
		P.ParsePrimaryExpr();
	}
	P.Ecart();
}


func (P *Parser) ParseUnaryExpr() {
	P.Trace("UnaryExpr");
	switch P.tok {
	case Scanner.ADD: fallthrough;
	case Scanner.SUB: fallthrough;
	case Scanner.NOT: fallthrough;
	case Scanner.XOR: fallthrough;
	case Scanner.MUL: fallthrough;
	case Scanner.RECV: fallthrough;
	case Scanner.AND:
		P.Next();
		P.ParseUnaryExpr();
		P.Ecart();
		return;
	}
	P.ParsePrimaryExpr();
	P.Ecart();
}


func Precedence(tok int) int {
	// TODO should use a map or array here for lookup
	switch tok {
	case Scanner.LOR:
		return 1;
	case Scanner.LAND:
		return 2;
	case Scanner.SEND, Scanner.RECV:
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


func (P *Parser) ParseBinaryExpr(prec1 int) {
	P.Trace("BinaryExpr");
	P.ParseUnaryExpr();
	for prec := Precedence(P.tok); prec >= prec1; prec-- {
		for Precedence(P.tok) == prec {
			P.Next();
			P.ParseBinaryExpr(prec + 1);
		}
	}
	P.Ecart();
}


func (P *Parser) ParseExpression() {
	P.Trace("Expression");
	indent := P.indent;
	P.ParseBinaryExpr(1);
	if indent != P.indent {
		panic "imbalanced tracing code";
	}
	P.Ecart();
}


// ----------------------------------------------------------------------------
// Statements

func (P *Parser) ParseBuiltinStat() {
	P.Trace("BuiltinStat");
	P.Expect(Scanner.IDENT);
	P.ParseExpressionList();  // TODO should be optional
	P.Ecart();
}


func (P *Parser) ParseSimpleStat() {
	P.Trace("SimpleStat");
	P.ParseExpression();
	if P.tok == Scanner.COLON {
		P.Next();
		P.Ecart();
		return;
	}
	if P.tok == Scanner.COMMA {
		P.Next();
		P.ParsePrimaryExprList();
	}
	switch P.tok {
	case Scanner.ASSIGN: fallthrough;
	case Scanner.DEFINE: fallthrough;
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
		P.Next();
		P.ParseExpressionList();
	case Scanner.INC:
		P.Next();
	case Scanner.DEC:
		P.Next();
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
	if P.tok != Scanner.SEMICOLON && P.tok != Scanner.RBRACE {
		P.ParseExpressionList();
	}
	P.Ecart();
}


func (P *Parser) ParseControlFlowStat(tok int) {
	P.Trace("ControlFlowStat");
	P.Expect(tok);
	if P.tok == Scanner.IDENT {
		P.ParseIdent();
	}
	P.Ecart();
}


func (P *Parser) ParseIfStat() {
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
	P.ParseBlock();
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
	P.ParseBlock();
	P.CloseScope();
	P.Ecart();
}


func (P *Parser) ParseCase() {
	P.Trace("Case");
	if P.tok == Scanner.CASE {
		P.Next();
		P.ParseExpressionList();
	} else {
		P.Expect(Scanner.DEFAULT);
	}
	P.Expect(Scanner.COLON);
	P.Ecart();
}


func (P *Parser) ParseCaseList() {
	P.Trace("CaseList");
	P.ParseCase();
	for P.tok == Scanner.CASE || P.tok == Scanner.DEFAULT {
		P.ParseCase();
	}
	P.Ecart();
}


func (P *Parser) ParseCaseClause() {
	P.Trace("CaseClause");
	P.ParseCaseList();
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
	for P.tok != Scanner.RBRACE {
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
			P.ParseIdent();
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


func (P *Parser) ParseRangeStat() bool {
	P.Trace("RangeStat");
	P.Expect(Scanner.RANGE);
	P.ParseIdentList();
	P.Expect(Scanner.DEFINE);
	P.ParseExpression();
	P.ParseBlock();
	P.Ecart();
}


func (P *Parser) ParseSelectStat() bool {
	P.Trace("SelectStat");
	P.Expect(Scanner.SELECT);
	P.Expect(Scanner.LBRACE);
	for P.tok != Scanner.RBRACE {
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
	case Scanner.SEND: fallthrough;
	case Scanner.RECV:
		P.ParseSimpleStat();  // send or receive
	case Scanner.IDENT:
		switch P.ident {
		case "print", "panic":
			P.ParseBuiltinStat();
		default:
			P.ParseSimpleStat();
		}
	case Scanner.GO:
		P.ParseGoStat();
	case Scanner.RETURN:
		P.ParseReturnStat();
	case Scanner.BREAK, Scanner.CONTINUE, Scanner.GOTO:
		P.ParseControlFlowStat(P.tok);
	case Scanner.LBRACE:
		P.ParseBlock();
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
		panic "imbalanced tracing code"
	}
	P.Ecart();
	return res;
}


// ----------------------------------------------------------------------------
// Declarations

func (P *Parser) ParseImportSpec() {
	P.Trace("ImportSpec");
	
	if P.tok == Scanner.PERIOD {
		P.Next();
	} else if P.tok == Scanner.IDENT {
		P.Next();
	}
	P.Expect(Scanner.STRING);
	
	P.Ecart();
}


func (P *Parser) ParseImportDecl() {
	P.Trace("ImportDecl");
	
	P.Expect(Scanner.IMPORT);
	if P.tok == Scanner.LPAREN {
		P.Next();
		for P.tok != Scanner.RPAREN {
			P.ParseImportSpec();
			P.Optional(Scanner.SEMICOLON);  // TODO this seems wrong
		}
		P.Next();
	} else {
		P.ParseImportSpec();
	}
	
	P.Ecart();
}


func (P *Parser) ParseConstSpec() {
	P.Trace("ConstSpec");
	
	list := P.ParseIdentDeclList(Object.CONST);
	typ := P.TryType();
	if typ != nil {
		for p := list.first; p != nil; p = p.next {
			p.obj.typ = typ;  // TODO should use/have set_type()!
		}
	}
	if P.tok == Scanner.ASSIGN {
		P.Next();
		P.ParseExpressionList();
	}
	
	P.Ecart();
}


func (P *Parser) ParseConstDecl() {
	P.Trace("ConstDecl");
	
	P.Expect(Scanner.CONST);
	if P.tok == Scanner.LPAREN {
		P.Next();
		for P.tok != Scanner.RPAREN {
			P.ParseConstSpec();
			if P.tok != Scanner.RPAREN {
				P.Expect(Scanner.SEMICOLON);
			}
		}
		P.Next();
	} else {
		P.ParseConstSpec();
	}
	
	P.Ecart();
}


func (P *Parser) ParseTypeSpec() {
	P.Trace("TypeSpec");
	
	pos := P.beg;
	ident := P.ParseIdent();
	obj := P.top_scope.Lookup(ident);  // only lookup in top scope!
	if obj != nil {
		// ok if forward declared type
		if obj.kind != Object.TYPE || obj.typ.form != Type.UNDEF {
			// TODO use obj.pos to refer to decl pos in error msg!
			P.Error(pos, `"` + ident + `" is declared already`);
		}
	} else {
		obj = Globals.NewObject(pos, Object.TYPE, ident);
		obj.typ = Universe.undef_t;  // TODO fix this
		P.top_scope.Insert(obj);
	}
	
	typ := P.TryType();  // no type if we have a forward decl
	if typ != nil {
		// TODO what about the name of incomplete types?
		obj.typ = typ;  // TODO should use/have set_typ()!
		if typ.obj == nil {
			typ.obj = obj;  // primary type object
		}
	}
	
	P.Ecart();
}


func (P *Parser) ParseTypeDecl() {
	P.Trace("TypeDecl");
	
	P.Expect(Scanner.TYPE);
	if P.tok == Scanner.LPAREN {
		P.Next();
		for P.tok != Scanner.RPAREN {
			P.ParseTypeSpec();
			if P.tok != Scanner.RPAREN {
				P.Expect(Scanner.SEMICOLON);
			}
		}
		P.Next();
	} else {
		P.ParseTypeSpec();
	}
	
	P.Ecart();
}


func (P *Parser) ParseVarSpec() {
	P.Trace("VarSpec");
	
	list := P.ParseIdentDeclList(Object.VAR);
	if P.tok == Scanner.ASSIGN {
		P.Next();
		P.ParseExpressionList();
	} else {
		typ := P.ParseType();
		for p := list.first; p != nil; p = p.next {
			p.obj.typ = typ;  // TODO should use/have set_type()!
		}
		if P.tok == Scanner.ASSIGN {
			P.Next();
			P.ParseExpressionList();
		}
	}
	
	P.Ecart();
}


func (P *Parser) ParseVarDecl() {
	P.Trace("VarDecl");
	
	P.Expect(Scanner.VAR);
	if P.tok == Scanner.LPAREN {
		P.Next();
		for P.tok != Scanner.RPAREN {
			P.ParseVarSpec();
			if P.tok != Scanner.RPAREN {
				P.Expect(Scanner.SEMICOLON);
			}
		}
		P.Next();
	} else {
		P.ParseVarSpec();
	}
	
	P.Ecart();
}


func (P *Parser) ParseFuncDecl() {
	P.Trace("FuncDecl");
	
	P.Expect(Scanner.FUNC);
	P.ParseNamedSignature();
	if P.tok == Scanner.SEMICOLON {
		// forward declaration
		P.Next();
	} else {
		P.ParseBlock();
	}
	
	P.Ecart();
}


func (P *Parser) ParseExportDecl() {
	P.Trace("ExportDecl");
	
	P.Expect(Scanner.EXPORT);
	if P.tok == Scanner.LPAREN {
		P.Next();
		for P.tok != Scanner.RPAREN {
			P.exports.AddStr(P.ParseIdent());
			P.Optional(Scanner.COMMA);  // TODO this seems wrong
		}
		P.Next();
	} else {
		P.exports.AddStr(P.ParseIdent());
		for P.tok == Scanner.COMMA {
			P.Next();
			P.exports.AddStr(P.ParseIdent());
		}
	}
	
	P.Ecart();
}


func (P *Parser) ParseDeclaration() {
	P.Trace("Declaration");
	
	indent := P.indent;
	switch P.tok {
	case Scanner.CONST:
		P.ParseConstDecl();
	case Scanner.TYPE:
		P.ParseTypeDecl();
	case Scanner.VAR:
		P.ParseVarDecl();
	case Scanner.FUNC:
		P.ParseFuncDecl();
	case Scanner.EXPORT:
		P.ParseExportDecl();
	default:
		P.Error(P.beg, "declaration expected");
		P.Next();  // make progress
	}
	if indent != P.indent {
		panic "imbalanced tracing code"
	}
	
	P.Ecart();
}


// ----------------------------------------------------------------------------
// Program

func (P *Parser) MarkExports() {
	if !EnableSemanticTests {
		return;
	}
	
	scope := P.top_scope;
	for p := P.exports.first; p != nil; p = p.next {
		obj := scope.Lookup(p.str);
		if obj != nil {
			obj.mark = true;
			// For now we export deep
			// TODO this should change eventually - we need selective export
			if obj.kind == Object.TYPE {
				typ := obj.typ;
				if typ.form == Type.STRUCT || typ.form == Type.INTERFACE {
					scope := typ.scope;
					for p := scope.entries.first; p != nil; p = p.next {
						p.obj.mark = true;
					}
				}
			}
		} else {
			// TODO need to report proper src position
			P.Error(0, `"` + p.str + `" is not declared - cannot be exported`);
		}
	}
}


func (P *Parser) ParseProgram() {
	P.Trace("Program");
	
	P.OpenScope();
	P.Expect(Scanner.PACKAGE);
	pkg := P.comp.pkgs[0];
	pkg.obj = P.ParseIdentDecl(Object.PACKAGE);
	P.Optional(Scanner.SEMICOLON);
	
	{	P.OpenScope();
		pkg.scope = P.top_scope;
		for P.tok == Scanner.IMPORT {
			P.ParseImportDecl();
			P.Optional(Scanner.SEMICOLON);
		}
		
		for P.tok != Scanner.EOF {
			P.ParseDeclaration();
			P.Optional(Scanner.SEMICOLON);
		}
		
		P.MarkExports();
		P.CloseScope();
	}
	
	P.CloseScope();
	P.Ecart();
}
