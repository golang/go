// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Parser

import Scanner "scanner"
import AST "ast"


export type Parser struct {
	verbose bool;
	indent uint;
	scanner *Scanner.Scanner;
	tokchan *<-chan *Scanner.Token;
	
	// Scanner.Token
	pos int;  // token source position
	tok int;  // one token look-ahead
	val string;  // token value (for IDENT, NUMBER, STRING only)
	semi bool;  // true if a semicolon was inserted by the previous statement

	// Nesting level
	level int;  // 0 = global scope, -1 = function scope of global functions, etc.
};


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
		P.pos, P.tok, P.val = P.scanner.Scan();
	} else {
		t := <-P.tokchan;
		P.tok, P.pos, P.val = t.tok, t.pos, t.val;
	}
	P.semi = false;
	if P.verbose {
		P.PrintIndent();
		print("[", P.pos, "] ", Scanner.TokenName(P.tok), "\n");
	}
}


func (P *Parser) Open(verbose bool, scanner *Scanner.Scanner, tokchan *<-chan *Scanner.Token) {
	P.verbose = verbose;
	P.indent = 0;
	P.scanner = scanner;
	P.tokchan = tokchan;
	P.Next();
	P.level = 0;
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


func (P *Parser) OptSemicolon() {
	if P.tok == Scanner.SEMICOLON {
		P.Next();
	}
}


// ----------------------------------------------------------------------------
// Scopes

func (P *Parser) OpenScope() {
}


func (P *Parser) CloseScope() {
}


// ----------------------------------------------------------------------------
// Common productions

func (P *Parser) TryType() (typ AST.Type, ok bool);
func (P *Parser) ParseExpression() AST.Expr;
func (P *Parser) ParseStatement() AST.Stat;
func (P *Parser) ParseDeclaration() AST.Node;


func (P *Parser) ParseIdent() *AST.Ident {
	P.Trace("Ident");

	ident := new(AST.Ident);
	ident.pos, ident.val = P.pos, "";
	if P.tok == Scanner.IDENT {
		ident.val = P.val;
		if P.verbose {
			P.PrintIndent();
			print("Ident = \"", ident.val, "\"\n");
		}
		P.Next();
	} else {
		P.Expect(Scanner.IDENT);  // use Expect() error handling
	}
	
	P.Ecart();
	return ident;
}


func (P *Parser) ParseIdentList() *AST.List {
	P.Trace("IdentList");

	list := AST.NewList();
	list.Add(P.ParseIdent());
	for P.tok == Scanner.COMMA {
		P.Next();
		list.Add(P.ParseIdent());
	}

	P.Ecart();
	return list;
}


func (P *Parser) ParseQualifiedIdent() AST.Expr {
	P.Trace("QualifiedIdent");

	ident := P.ParseIdent();
	var qident AST.Expr = ident;

	for P.tok == Scanner.PERIOD {
		pos := P.pos;
		P.Next();
		y := P.ParseIdent();

		z := new(AST.Selector);
		z.pos, z.x, z.field = pos, qident, y.val;
		qident = z;
	}
	
	P.Ecart();
	return qident;
}


// ----------------------------------------------------------------------------
// Types

func (P *Parser) ParseType() AST.Type {
	P.Trace("Type");
	
	typ, ok := P.TryType();
	if !ok {
		P.Error(P.pos, "type expected");
	}
	
	P.Ecart();
	return typ;
}


func (P *Parser) ParseVarType() AST.Type {
	P.Trace("VarType");
	
	typ := P.ParseType();
	
	P.Ecart();
	return typ;
}


func (P *Parser) ParseTypeName() AST.Type {
	P.Trace("TypeName");
	
	typ := P.ParseQualifiedIdent();

	P.Ecart();
	return typ;
}


func (P *Parser) ParseArrayType() *AST.ArrayType {
	P.Trace("ArrayType");
	
	typ := new(AST.ArrayType);
	typ.pos = P.pos;
	typ.len_ = AST.NIL;
	
	P.Expect(Scanner.LBRACK);
	if P.tok != Scanner.RBRACK {
		// TODO set typ.len
		typ.len_ = P.ParseExpression();
	}
	P.Expect(Scanner.RBRACK);
	typ.elt = P.ParseType();

	P.Ecart();
	return typ;
}


func (P *Parser) ParseChannelType() *AST.ChannelType {
	P.Trace("ChannelType");
	
	typ := new(AST.ChannelType);
	typ.pos = P.pos;
	typ.mode = AST.FULL;
	
	if P.tok == Scanner.CHAN {
		P.Next();
		if P.tok == Scanner.ARROW {
			P.Next();
			typ.mode = AST.SEND;
		}
	} else {
		P.Expect(Scanner.ARROW);
		P.Expect(Scanner.CHAN);
		typ.mode = AST.RECV;
	}
	typ.elt = P.ParseVarType();

	P.Ecart();
	return typ;
}


func (P *Parser) ParseVarDeclList() *AST.VarDeclList {
	P.Trace("VarDeclList");

	vars := new(AST.VarDeclList);
	vars.idents = AST.NewList();
	vars.typ = AST.NIL;
	
	vars.idents.Add(P.ParseType());
	for P.tok == Scanner.COMMA {
		P.Next();
		vars.idents.Add(P.ParseType());
	}
	
	var ok bool;
	vars.typ, ok = P.TryType();

	if !ok {
		// we must have a list of types
	}
	
	P.Ecart();
	return vars;
}


// Returns a list of *AST.VarDeclList or Type
func (P *Parser) ParseParameterList() *AST.List {
	P.Trace("ParameterList");
	
	list := AST.NewList();
	list.Add(P.ParseVarDeclList());
	for P.tok == Scanner.COMMA {
		P.Next();
		list.Add(P.ParseVarDeclList());
	}
	
	P.Ecart();
	return list;
}


// Returns a list of AST.VarDeclList
func (P *Parser) ParseParameters() *AST.List {
	P.Trace("Parameters");
	
	var list *AST.List;
	P.Expect(Scanner.LPAREN);
	if P.tok != Scanner.RPAREN {
		list = P.ParseParameterList();
	}
	P.Expect(Scanner.RPAREN);
	
	P.Ecart();
	return list;
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


func (P *Parser) ParseResult() *AST.List {
	P.Trace("Result");
	
	var result *AST.List;
	if P.tok == Scanner.LPAREN {
		result = P.ParseParameters();
	} else {
		typ, ok := P.TryType();
		if ok {
			vars := new(AST.VarDeclList);
			vars.typ = typ;
			list := AST.NewList();
			list.Add(vars);
			result = list;
		}
	}

	P.Ecart();
	return result;
}


// Function types
//
// (params)
// (params) type
// (params) (results)

func (P *Parser) ParseFunctionType() *AST.FunctionType {
	P.Trace("FunctionType");
	
	typ := new(AST.FunctionType);
	typ.pos = P.pos;
	typ.params = P.ParseParameters();
	typ.result = P.ParseResult();
	
	P.Ecart();
	return typ;
}


func (P *Parser) ParseMethodDecl() *AST.MethodDecl {
	P.Trace("MethodDecl");
	
	decl := new(AST.MethodDecl);
	decl.ident = P.ParseIdent();
	decl.typ = P.ParseFunctionType();
	
	P.Ecart();
	return decl;
}


func (P *Parser) ParseInterfaceType() *AST.InterfaceType {
	P.Trace("InterfaceType");
	
	typ := new(AST.InterfaceType);
	typ.pos = P.pos;
	typ.methods = AST.NewList();
	
	P.Expect(Scanner.INTERFACE);
	
	if P.tok == Scanner.LBRACE {
		P.Next();
		for P.tok == Scanner.IDENT {
			typ.methods.Add(P.ParseMethodDecl());
			if P.tok != Scanner.RBRACE {
				P.Expect(Scanner.SEMICOLON);
			}
		}
		P.Expect(Scanner.RBRACE);
	}

	P.Ecart();
	return typ;
}


func (P *Parser) ParseMapType() *AST.MapType {
	P.Trace("MapType");
	
	typ := new(AST.MapType);
	typ.pos = P.pos;
	
	P.Expect(Scanner.MAP);
	P.Expect(Scanner.LBRACK);
	typ.key = P.ParseVarType();
	P.Expect(Scanner.RBRACK);
	typ.val = P.ParseVarType();
	
	P.Ecart();
	return typ;
}


func (P *Parser) ParseStructType() *AST.StructType {
	P.Trace("StructType");

	typ := new(AST.StructType);
	typ.pos = P.pos;
	typ.fields = AST.NewList();
	
	P.Expect(Scanner.STRUCT);
	
	if P.tok == Scanner.LBRACE {
		P.Next();
		for P.tok == Scanner.IDENT {
			typ.fields.Add(P.ParseVarDeclList());
			if P.tok != Scanner.RBRACE {
				P.Expect(Scanner.SEMICOLON);
			}
		}
		P.OptSemicolon();
		P.Expect(Scanner.RBRACE);
	}

	P.Ecart();
	return typ;
}


func (P *Parser) ParsePointerType() *AST.PointerType {
	P.Trace("PointerType");
	
	typ := new(AST.PointerType);
	typ.pos = P.pos;
	
	P.Expect(Scanner.MUL);
	typ.base = P.ParseType();
	
	P.Ecart();
	return typ;
}


// Returns false if no type was found.
func (P *Parser) TryType() (typ_ AST.Type, ok_ bool) {
	P.Trace("Type (try)");
	
	var typ AST.Type = AST.NIL;
	found := true;
	switch P.tok {
	case Scanner.IDENT: typ = P.ParseTypeName();
	case Scanner.LBRACK: typ = P.ParseArrayType();
	case Scanner.CHAN, Scanner.ARROW: typ = P.ParseChannelType();
	case Scanner.INTERFACE: typ = P.ParseInterfaceType();
	case Scanner.LPAREN: typ = P.ParseFunctionType();
	case Scanner.MAP: typ = P.ParseMapType();
	case Scanner.STRUCT: typ = P.ParseStructType();
	case Scanner.MUL: typ = P.ParsePointerType();
	default: found = false;
	}

	P.Ecart();
	return typ, found;
}


// ----------------------------------------------------------------------------
// Blocks

func (P *Parser) ParseStatementList() *AST.List {
	P.Trace("StatementList");
	
	stats := AST.NewList();
	for P.tok != Scanner.CASE && P.tok != Scanner.DEFAULT && P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
		stats.Add(P.ParseStatement());
		if P.tok == Scanner.SEMICOLON {
			P.Next();
		} else if P.semi {
			P.semi = false;  // consume inserted ";"
		} else {
			break;
		}
	}
	
	P.Ecart();
	return stats;
}


func (P *Parser) ParseBlock() *AST.Block {
	P.Trace("Block");
	
	block := new(AST.Block);
	block.pos = P.pos;
	
	P.Expect(Scanner.LBRACE);
	P.OpenScope();
	if P.tok != Scanner.RBRACE {
		block.stats = P.ParseStatementList();
	}
	P.OptSemicolon();
	P.CloseScope();
	P.Expect(Scanner.RBRACE);
	P.semi = true;  // allow optional semicolon
	
	P.Ecart();
	return block;
}


// ----------------------------------------------------------------------------
// Expressions

func (P *Parser) ParseExpressionList(list *AST.List) {
	P.Trace("ExpressionList");

	list.Add(P.ParseExpression());
	for P.tok == Scanner.COMMA {
		P.Next();
		list.Add(P.ParseExpression());
	}
	
	P.Ecart();
}


func (P *Parser) ParseNewExpressionList() *AST.List {
	list := AST.NewList();
	P.ParseExpressionList(list);
	return list;
}


func (P *Parser) ParseFunctionLit() *AST.FunctionLit {
	P.Trace("FunctionLit");
	
	fun := new(AST.FunctionLit);
	fun.pos = P.pos;
	
	P.Expect(Scanner.FUNC);
	fun.typ = P.ParseFunctionType();
	fun.body = P.ParseBlock();
	
	P.Ecart();
	return fun;
}


func (P *Parser) ParseExpressionPair() AST.Expr {
	P.Trace("ExpressionPair");

	p := new(AST.Pair);
	p.x = P.ParseExpression();
	p.pos = P.pos;
	P.Expect(Scanner.COLON);
	p.y = P.ParseExpression();
	
	P.Ecart();
	return p;
}


func (P *Parser) ParseExpressionPairList(list *AST.List) {
	P.Trace("ExpressionPairList");

	list.Add(P.ParseExpressionPair());
	for P.tok == Scanner.COMMA {
		list.Add(P.ParseExpressionPair());
	}
	
	P.Ecart();
}


func (P *Parser) ParseCompositeLit(typ AST.Type) AST.Expr {
	P.Trace("CompositeLit");
	
	lit := new(AST.CompositeLit);
	lit.pos = P.pos;
	lit.typ = typ;
	lit.vals = AST.NewList();
	
	P.Expect(Scanner.LBRACE);
	// TODO: should allow trailing ','
	if P.tok != Scanner.RBRACE {
		x := P.ParseExpression();
		if P.tok == Scanner.COMMA {
			P.Next();
			lit.vals.Add(x);
			if P.tok != Scanner.RBRACE {
				P.ParseExpressionList(lit.vals);
			}
		} else if P.tok == Scanner.COLON {
			p := new(AST.Pair);
			p.pos = P.pos;
			p.x = x;
			P.Next();
			p.y = P.ParseExpression();
			lit.vals.Add(p);
			if P.tok == Scanner.COMMA {
				P.Next();
				if P.tok != Scanner.RBRACE {
					P.ParseExpressionPairList(lit.vals);
				}
			}
		} else {
			lit.vals.Add(x);
		}
	}
	P.Expect(Scanner.RBRACE);

	P.Ecart();
	return lit;
}


func (P *Parser) ParseOperand() AST.Expr {
	P.Trace("Operand");

	var op AST.Expr;

	switch P.tok {
	case Scanner.IDENT:
		op = P.ParseIdent();
		
	case Scanner.LPAREN:
		P.Next();
		op = P.ParseExpression();
		P.Expect(Scanner.RPAREN);

	case Scanner.INT, Scanner.FLOAT, Scanner.STRING:
		lit := new(AST.Literal);
		lit.pos, lit.tok, lit.val = P.pos, P.tok, P.val;
		op = lit;
		P.Next();

	case Scanner.FUNC:
		op = P.ParseFunctionLit();
		
	case Scanner.HASH:
		P.Next();
		typ := P.ParseType();
		P.ParseCompositeLit(typ);
		op = AST.NIL;

	default:
		if P.tok != Scanner.IDENT {
			typ, ok := P.TryType();
			if ok {
				op = P.ParseCompositeLit(typ);
				break;
			}
		}

		P.Error(P.pos, "operand expected");
		P.Next();  // make progress
	}

	P.Ecart();
	return op;
}


func (P *Parser) ParseSelectorOrTypeGuard(x AST.Expr) AST.Expr {
	P.Trace("SelectorOrTypeGuard");

	pos := P.pos;
	P.Expect(Scanner.PERIOD);
	
	if P.tok == Scanner.IDENT {
		ident := P.ParseIdent();
		
		z := new(AST.Selector);
		z.pos, z.x, z.field = pos, x, ident.val;
		x = z;
		
	} else {
		P.Expect(Scanner.LPAREN);
		P.ParseType();
		P.Expect(Scanner.RPAREN);
	}
	
	P.Ecart();
	return x;
}


func (P *Parser) ParseIndexOrSlice(x AST.Expr) AST.Expr {
	P.Trace("IndexOrSlice");
	
	pos := P.pos;
	P.Expect(Scanner.LBRACK);
	i := P.ParseExpression();
	if P.tok == Scanner.COLON {
		P.Next();
		j := P.ParseExpression();
		// TODO: handle this case
	}
	P.Expect(Scanner.RBRACK);

	z := new(AST.Index);
	z.pos, z.x, z.index = pos, x, i;
	
	P.Ecart();
	return z;
}


func (P *Parser) ParseCall(x AST.Expr) *AST.Call {
	P.Trace("Call");

	call := new(AST.Call);
	call.pos = P.pos;
	call.fun = x;
	call.args = nil;
	
	P.Expect(Scanner.LPAREN);
	if P.tok != Scanner.RPAREN {
	   	// first arguments could be a type if the call is to "new"
		// - exclude type names because they could be expression starts
		// - exclude "("'s because function types are not allowed and they indicate an expression
		// - still a problem for "new(*T)" (the "*")
		// - possibility: make "new" a keyword again (or disallow "*" types in new)
		if P.tok != Scanner.IDENT && P.tok != Scanner.LPAREN {
			typ, ok := P.TryType();
			if ok {
				call.args = AST.NewList();
				call.args.Add(typ);
				if P.tok == Scanner.COMMA {
					P.Next();
					if P.tok != Scanner.RPAREN {
						P.ParseExpressionList(call.args);
					}
				}
			} else {
				call.args = P.ParseNewExpressionList();
			}
		} else {
			call.args = P.ParseNewExpressionList();
		}
	}
	P.Expect(Scanner.RPAREN);
	
	P.Ecart();
	return call;
}


func (P *Parser) ParsePrimaryExpr() AST.Expr {
	P.Trace("PrimaryExpr");
	
	x := P.ParseOperand();
	for {
		switch P.tok {
		case Scanner.PERIOD: x = P.ParseSelectorOrTypeGuard(x);
		case Scanner.LBRACK: x = P.ParseIndexOrSlice(x);
		case Scanner.LPAREN: x = P.ParseCall(x);
		default: goto exit;
		}
	}
exit:

	P.Ecart();
	return x;
}


func (P *Parser) ParseUnaryExpr() AST.Expr {
	P.Trace("UnaryExpr");
	
	var x AST.Expr = AST.NIL;
	switch P.tok {
	case
		Scanner.ADD, Scanner.SUB,
		Scanner.NOT, Scanner.XOR,
		Scanner.MUL, Scanner.ARROW,
		Scanner.AND:
			pos, tok := P.pos, P.tok;
			P.Next();
			y := P.ParseUnaryExpr();

			z := new(AST.Unary);
			z.pos, z.tok, z.x = pos, tok, y;
			x = z;
			
		default:
			x = P.ParsePrimaryExpr();
	}
	
	P.Ecart();
	return x;
}


func (P *Parser) ParseBinaryExpr(prec1 int) AST.Expr {
	P.Trace("BinaryExpr");
	
	x := P.ParseUnaryExpr();
	for prec := Scanner.Precedence(P.tok); prec >= prec1; prec-- {
		for Scanner.Precedence(P.tok) == prec {
			pos, tok := P.pos, P.tok;
			P.Next();
			y := P.ParseBinaryExpr(prec + 1);
			
			z := new(AST.Binary);
			z.pos, z.tok, z.x, z.y = pos, tok, x, y;
			x = z;
		}
	}
	
	P.Ecart();
	return x;
}


func (P *Parser) ParseExpression() AST.Expr {
	P.Trace("Expression");
	indent := P.indent;
	
	x := P.ParseBinaryExpr(1);
	
	if indent != P.indent {
		panic("imbalanced tracing code (Expression)");
	}

	P.Ecart();
	return x;
}


// ----------------------------------------------------------------------------
// Statements

func (P *Parser) ParseSimpleStat() AST.Stat {
	P.Trace("SimpleStat");
	
	var stat AST.Stat = AST.NIL;
	x := P.ParseNewExpressionList();
	
	switch P.tok {
	case Scanner.COLON:
		// label declaration
		l := new(AST.Label);
		l.pos = P.pos;
		if x.len() == 1 {
			l.ident = x.at(0);
		} else {
			P.Error(P.pos, "illegal label declaration");
			l.ident = AST.NIL;
		}
		P.Next();  // consume ":"
		P.semi = true;  // allow optional semicolon
		stat = l;
		
	case
		Scanner.DEFINE, Scanner.ASSIGN, Scanner.ADD_ASSIGN,
		Scanner.SUB_ASSIGN, Scanner.MUL_ASSIGN, Scanner.QUO_ASSIGN,
		Scanner.REM_ASSIGN, Scanner.AND_ASSIGN, Scanner.OR_ASSIGN,
		Scanner.XOR_ASSIGN, Scanner.SHL_ASSIGN, Scanner.SHR_ASSIGN:
		pos, tok := P.pos, P.tok;
		P.Next();
		y := P.ParseNewExpressionList();
		a := new(AST.Assignment);
		a.pos, a.tok, a.lhs, a.rhs = pos, tok, x, y;
		stat = a;
		
	default:
		if P.tok == Scanner.INC || P.tok == Scanner.DEC {
			s := new(AST.IncDecStat);
			s.pos, s.tok = P.pos, P.tok;
			if x.len() == 1 {
				s.expr = x.at(0);
			} else {
				P.Error(P.pos, "more then one operand");
			}
			P.Next();
			stat = s;
		} else {
			s := new(AST.ExprStat);
			if x != nil && x.len() > 0 {
				s.expr = x.at(0);
			} else {
				// this is a syntax error
				s.expr = AST.NIL;
			}
			stat = s;
		}
	}
	
	P.Ecart();
	return stat;
}


func (P *Parser) ParseGoStat() *AST.GoStat {
	P.Trace("GoStat");
	
	stat := new(AST.GoStat);
	stat.pos = P.pos;
	
	P.Expect(Scanner.GO);
	stat.expr = P.ParseExpression();
	
	P.Ecart();
	return stat;
}


func (P *Parser) ParseReturnStat() *AST.ReturnStat {
	P.Trace("ReturnStat");
	
	stat := new(AST.ReturnStat);
	stat.pos = P.pos;
	
	P.Expect(Scanner.RETURN);
	if P.tok != Scanner.SEMICOLON && P.tok != Scanner.RBRACE {
		stat.res = P.ParseNewExpressionList();
	}
	
	P.Ecart();
	return stat;
}


func (P *Parser) ParseControlFlowStat(tok int) *AST.ControlFlowStat {
	P.Trace("ControlFlowStat");
	
	stat := new(AST.ControlFlowStat);
	stat.pos, stat.tok = P.pos, P.tok;
	
	P.Expect(tok);
	if P.tok == Scanner.IDENT {
		stat.label = P.ParseIdent();
	}
	
	P.Ecart();
	return stat;
}


func (P *Parser) ParseControlClause(keyword int) *AST.ControlClause {
	P.Trace("StatHeader");
	
	ctrl := new(AST.ControlClause);
	ctrl.init, ctrl.expr, ctrl.post = AST.NIL, AST.NIL, AST.NIL;

	P.Expect(keyword);
	if P.tok != Scanner.LBRACE {
		if P.tok != Scanner.SEMICOLON {
			ctrl.init = P.ParseSimpleStat();
			ctrl.has_init = true;
		}
		if P.tok == Scanner.SEMICOLON {
			P.Next();
			if P.tok != Scanner.SEMICOLON && P.tok != Scanner.LBRACE {
				ctrl.expr = P.ParseExpression();
				ctrl.has_expr = true;
			}
			if keyword == Scanner.FOR {
				P.Expect(Scanner.SEMICOLON);
				if P.tok != Scanner.LBRACE {
					ctrl.post = P.ParseSimpleStat();
					ctrl.has_post = true;
				}
			}
		} else {
			ctrl.expr, ctrl.has_expr = ctrl.init, ctrl.has_init;
			ctrl.init, ctrl.has_init = AST.NIL, false;
		}
	}

	P.Ecart();
	return ctrl;
}


func (P *Parser) ParseIfStat() *AST.IfStat {
	P.Trace("IfStat");

	stat := new(AST.IfStat);
	stat.pos = P.pos;
	stat.ctrl = P.ParseControlClause(Scanner.IF);
	stat.then = P.ParseBlock();
	if P.tok == Scanner.ELSE {
		P.Next();
		if P.tok == Scanner.IF {
			stat.else_ = P.ParseIfStat();
		} else {
			// TODO: Should be P.ParseBlock().
			stat.else_ = P.ParseStatement();
		}
		stat.has_else = true;
	}
	
	P.Ecart();
	return stat;
}


func (P *Parser) ParseForStat() *AST.ForStat {
	P.Trace("ForStat");
	
	stat := new(AST.ForStat);
	stat.pos = P.pos;
	
	stat.ctrl = P.ParseControlClause(Scanner.FOR);
	stat.body = P.ParseBlock();
	
	P.Ecart();
	return stat;
}


func (P *Parser) ParseCase() *AST.CaseClause {
	P.Trace("Case");
	
	clause := new(AST.CaseClause);
	clause.pos = P.pos;
	
	if P.tok == Scanner.CASE {
		P.Next();
		clause.exprs = P.ParseNewExpressionList();
	} else {
		P.Expect(Scanner.DEFAULT);
	}
	P.Expect(Scanner.COLON);
	
	P.Ecart();
	return clause;
}


func (P *Parser) ParseCaseClause() *AST.CaseClause {
	P.Trace("CaseClause");

	clause := P.ParseCase();
	if P.tok != Scanner.CASE && P.tok != Scanner.DEFAULT && P.tok != Scanner.RBRACE {
		clause.stats = P.ParseStatementList();
	}
	
	P.Ecart();
	return clause;
}


func (P *Parser) ParseSwitchStat() *AST.SwitchStat {
	P.Trace("SwitchStat");
	
	stat := new(AST.SwitchStat);
	stat.pos = P.pos;
	stat.ctrl = P.ParseControlClause(Scanner.SWITCH);
	stat.cases = AST.NewList();
	
	P.Expect(Scanner.LBRACE);
	for P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
		stat.cases.Add(P.ParseCaseClause());
	}
	P.Expect(Scanner.RBRACE);
	P.semi = true;  // allow optional semicolon

	P.Ecart();
	return stat;
}


func (P *Parser) ParseCommCase() {
  P.Trace("CommCase");
  
  if P.tok == Scanner.CASE {
	P.Next();
	P.ParseExpression();
	if P.tok == Scanner.ASSIGN || P.tok == Scanner.DEFINE {
		P.Next();
		P.Expect(Scanner.ARROW);
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
	}
	
	P.Ecart();
}


func (P *Parser) ParseSelectStat() {
	P.Trace("SelectStat");
	
	P.Expect(Scanner.SELECT);
	P.Expect(Scanner.LBRACE);
	for P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
		P.ParseCommClause();
	}
	P.Expect(Scanner.RBRACE);
	P.semi = true;  // allow optional semicolon
	
	P.Ecart();
}


func (P *Parser) ParseFallthroughStat() {
	P.Trace("FallthroughStat");
	
	P.Expect(Scanner.FALLTHROUGH);

	P.Ecart();
}


func (P *Parser) ParseEmptyStat() {
	P.Trace("EmptyStat");
	P.Ecart();
}


func (P *Parser) ParseRangeStat() {
	P.Trace("RangeStat");
	
	P.Expect(Scanner.RANGE);
	P.ParseIdentList();
	P.Expect(Scanner.DEFINE);
	P.ParseExpression();
	P.ParseBlock();
	
	P.Ecart();;
}


func (P *Parser) ParseStatement() AST.Stat {
	P.Trace("Statement");
	indent := P.indent;

	var stat AST.Stat = AST.NIL;
	switch P.tok {
	case Scanner.CONST, Scanner.TYPE, Scanner.VAR:
		stat = P.ParseDeclaration();
	case Scanner.FUNC:
		// for now we do not allow local function declarations
		fallthrough;
	case Scanner.MUL, Scanner.ARROW, Scanner.IDENT, Scanner.LPAREN:
		stat = P.ParseSimpleStat();
	case Scanner.GO:
		stat = P.ParseGoStat();
	case Scanner.RETURN:
		stat = P.ParseReturnStat();
	case Scanner.BREAK, Scanner.CONTINUE, Scanner.GOTO:
		stat = P.ParseControlFlowStat(P.tok);
	case Scanner.LBRACE:
		stat = P.ParseBlock();
	case Scanner.IF:
		stat = P.ParseIfStat();
	case Scanner.FOR:
		stat = P.ParseForStat();
	case Scanner.SWITCH:
		stat = P.ParseSwitchStat();
	case Scanner.RANGE:
		P.ParseRangeStat();
	case Scanner.SELECT:
		P.ParseSelectStat();
	case Scanner.FALLTHROUGH:
		P.ParseFallthroughStat();
	default:
		P.ParseEmptyStat();  // for complete tracing output only
	}

	if indent != P.indent {
		panic("imbalanced tracing code (Statement)");
	}
	P.Ecart();
	return stat;
}


// ----------------------------------------------------------------------------
// Declarations

func (P *Parser) ParseImportSpec() *AST.ImportDecl {
	P.Trace("ImportSpec");
	
	decl := new(AST.ImportDecl);

	if P.tok == Scanner.PERIOD {
		P.Error(P.pos, `"import ." not yet handled properly`);
		P.Next();
	} else if P.tok == Scanner.IDENT {
		decl.ident = P.ParseIdent();
	}
	
	if P.tok == Scanner.STRING {
		// TODO eventually the scanner should strip the quotes
		decl.file = P.val;
		P.Next();
	} else {
		P.Expect(Scanner.STRING);  // use Expect() error handling
	}
	
	P.Ecart();
	return decl;
}


func (P *Parser) ParseConstSpec(exported bool) *AST.ConstDecl {
	P.Trace("ConstSpec");
	
	decl := new(AST.ConstDecl);
	decl.ident = P.ParseIdent();
	var ok bool;
	decl.typ, ok = P.TryType();
	decl.val = AST.NIL;
	
	if P.tok == Scanner.ASSIGN {
		P.Next();
		decl.val = P.ParseExpression();
	}
	
	P.Ecart();
	return decl;
}


func (P *Parser) ParseTypeSpec(exported bool) *AST.TypeDecl {
	P.Trace("TypeSpec");

	decl := new(AST.TypeDecl);
	decl.ident = P.ParseIdent();
	decl.typ = P.ParseType();
	P.semi = true;  // allow optional semicolon
	
	P.Ecart();
	return decl;
}


func (P *Parser) ParseVarSpec(exported bool) *AST.VarDecl {
	P.Trace("VarSpec");
	
	decl := new(AST.VarDecl);
	decl.idents = P.ParseIdentList();
	if P.tok == Scanner.ASSIGN {
		P.Next();
		decl.typ = AST.NIL;
		decl.vals = P.ParseNewExpressionList();
	} else {
		decl.typ = P.ParseVarType();
		if P.tok == Scanner.ASSIGN {
			P.Next();
			decl.vals = P.ParseNewExpressionList();
		}
	}
	
	P.Ecart();
	return decl;
}


// TODO Replace this by using function pointers derived from methods.
func (P *Parser) ParseSpec(exported bool, keyword int) AST.Decl {
	switch keyword {
	case Scanner.IMPORT: return P.ParseImportSpec();
	case Scanner.CONST: return P.ParseConstSpec(exported);
	case Scanner.TYPE: return P.ParseTypeSpec(exported);
	case Scanner.VAR: return P.ParseVarSpec(exported);
	}
	panic("UNREACHABLE");
	return AST.NIL;
}


func (P *Parser) ParseDecl(exported bool, keyword int) *AST.Declaration {
	P.Trace("Decl");
	
	decl := new(AST.Declaration);
	decl.decls = AST.NewList();
	decl.pos, decl.tok = P.pos, P.tok;
	
	P.Expect(keyword);
	if P.tok == Scanner.LPAREN {
		P.Next();
		for P.tok != Scanner.RPAREN && P.tok != Scanner.EOF {
			decl.decls.Add(P.ParseSpec(exported, keyword));
			if P.tok == Scanner.SEMICOLON {
				P.Next();
			} else {
				break;
			}
		}
		P.Expect(Scanner.RPAREN);
		P.semi = true;  // allow optional semicolon
		
	} else {
		decl.decls.Add(P.ParseSpec(exported, keyword));
	}
	
	P.Ecart();
	return decl;
}


// Function declarations
//
// func        ident (params)
// func        ident (params) type
// func        ident (params) (results)
// func (recv) ident (params)
// func (recv) ident (params) type
// func (recv) ident (params) (results)

func (P *Parser) ParseFunctionDecl(exported bool) *AST.FuncDecl {
	P.Trace("FunctionDecl");
	
	fun := new(AST.FuncDecl);
	fun.pos = P.pos;

	P.Expect(Scanner.FUNC);

	P.OpenScope();
	P.level--;

	var recv *AST.VarDeclList;
	if P.tok == Scanner.LPAREN {
		pos := P.pos;
		tmp := P.ParseParameters();
		if tmp.len() > 0 {
			recv = tmp.at(0);
		}
		if recv.idents.len() != 1 {
			P.Error(pos, "must have exactly one receiver");
		}
	}
	
	fun.ident = P.ParseIdent();
	fun.typ = P.ParseFunctionType();
	fun.typ.recv = recv;
	
	P.level++;
	P.CloseScope();

	if P.tok == Scanner.LBRACE {
		fun.body = P.ParseBlock();
	}
	
	P.Ecart();
	
	return fun;
}


func (P *Parser) ParseExportDecl() {
	P.Trace("ExportDecl");
	
	// TODO This is deprecated syntax and should go away eventually.
	// (Also at the moment the syntax is everything goes...)
	//P.Expect(Scanner.EXPORT);

	has_paren := false;
	if P.tok == Scanner.LPAREN {
		P.Next();
		has_paren = true;
	}
	for P.tok == Scanner.IDENT {
		ident := P.ParseIdent();
		if P.tok == Scanner.COMMA {
			P.Next();  // TODO this seems wrong
		}
	}
	if has_paren {
		P.Expect(Scanner.RPAREN)
	}
	
	P.Ecart();
}


func (P *Parser) ParseDeclaration() AST.Node {
	P.Trace("Declaration");
	indent := P.indent;
	
	var node AST.Node;

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
		node = P.ParseDecl(exported, P.tok);
	case Scanner.FUNC:
		node = P.ParseFunctionDecl(exported);
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
	return node;
}


// ----------------------------------------------------------------------------
// Program

func (P *Parser) ParseProgram() *AST.Program {
	P.Trace("Program");
	
	P.OpenScope();
	pos := P.pos;
	P.Expect(Scanner.PACKAGE);
	ident := P.ParseIdent();
	
	decls := AST.NewList();
	{	P.OpenScope();
		if P.level != 0 {
			panic("incorrect scope level");
		}
		
		for P.tok == Scanner.IMPORT {
			decls.Add(P.ParseDecl(false, Scanner.IMPORT));
			P.OptSemicolon();
		}
		
		for P.tok != Scanner.EOF {
			decls.Add(P.ParseDeclaration());
			P.OptSemicolon();
		}
		
		if P.level != 0 {
			panic("incorrect scope level");
		}
		P.CloseScope();
	}
	
	P.CloseScope();
	P.Ecart();
	
	x := new(AST.Program);
	x.pos, x.ident, x.decls = pos, ident, decls;
	return x;
}
