// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Parser

import Scanner "scanner"
import Node "node"


export type Parser struct {
	verbose bool;
	indent uint;
	scanner *Scanner.Scanner;
	tokchan *<-chan *Scanner.Token;
	
	// Scanner.Token
	pos int;  // token source position
	tok int;  // one token look-ahead
	val string;  // token value (for IDENT, NUMBER, STRING only)
	
	// Non-syntactic parser control
	opt_semi bool;  // true if semicolon is optional

	// Nesting levels
	expr_lev int;  // 0 = control clause level, 1 = expr inside ()'s
	scope_lev int;  // 0 = global scope, 1 = function scope of global functions, etc.
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
	P.indent++;  // always check proper identation
}


func (P *Parser) Ecart() {
	P.indent--;  // always check proper identation
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
	P.opt_semi = false;
	if P.verbose {
		P.PrintIndent();
		print("[", P.pos, "] ", Scanner.TokenString(P.tok), "\n");
	}
}


func (P *Parser) Open(verbose bool, scanner *Scanner.Scanner, tokchan *<-chan *Scanner.Token) {
	P.verbose = verbose;
	P.indent = 0;
	P.scanner = scanner;
	P.tokchan = tokchan;
	P.Next();
	P.expr_lev = 1;
	P.scope_lev = 0;
}


func (P *Parser) Error(pos int, msg string) {
	P.scanner.Error(pos, msg);
}


func (P *Parser) Expect(tok int) {
	if P.tok != tok {
		P.Error(P.pos, "expected '" + Scanner.TokenString(tok) + "', found '" + Scanner.TokenString(P.tok) + "'");
	}
	P.Next();  // make progress in any case
}


func (P *Parser) OptSemicolon() {
	if P.tok == Scanner.SEMICOLON {
		P.Next();
	}
}


// ----------------------------------------------------------------------------
// Common productions

func (P *Parser) TryType() *Node.Type;
func (P *Parser) ParseExpression() *Node.Expr;
func (P *Parser) ParseStatement() *Node.Stat;
func (P *Parser) ParseDeclaration() *Node.Decl;


func (P *Parser) ParseIdent() *Node.Expr {
	P.Trace("Ident");

	var x *Node.Expr;
	if P.tok == Scanner.IDENT {
		x = Node.NewLit(P.pos, Scanner.IDENT, P.val);
		if P.verbose {
			P.PrintIndent();
			print("Ident = \"", x.s, "\"\n");
		}
		P.Next();
	} else {
		P.Expect(Scanner.IDENT);  // use Expect() error handling
	}
	
	P.Ecart();
	return x;
}


func (P *Parser) ParseIdentList() *Node.List {
	P.Trace("IdentList");

	list := Node.NewList();
	list.Add(P.ParseIdent());
	for P.tok == Scanner.COMMA {
		P.Next();
		list.Add(P.ParseIdent());
	}

	P.Ecart();
	return list;
}


// ----------------------------------------------------------------------------
// Types

func (P *Parser) ParseType() *Node.Type {
	P.Trace("Type");
	
	typ := P.TryType();
	if typ == nil {
		P.Error(P.pos, "type expected");
	}
	
	P.Ecart();
	return typ;
}


func (P *Parser) ParseVarType() *Node.Type {
	P.Trace("VarType");
	
	typ := P.ParseType();
	
	P.Ecart();
	return typ;
}


func (P *Parser) ParseQualifiedIdent() *Node.Expr {
	P.Trace("QualifiedIdent");

	x := P.ParseIdent();
	for P.tok == Scanner.PERIOD {
		pos := P.pos;
		P.Next();
		y := P.ParseIdent();
		x = Node.NewExpr(pos, Scanner.PERIOD, x, y);
	}
	
	P.Ecart();
	return x;
}


func (P *Parser) ParseTypeName() *Node.Type {
	P.Trace("TypeName");
	
	t := Node.NewType(P.pos, P.tok);
	t.expr = P.ParseQualifiedIdent();

	P.Ecart();
	return t;
}


func (P *Parser) ParseArrayType() *Node.Type {
	P.Trace("ArrayType");
	
	t := Node.NewType(P.pos, Scanner.LBRACK);
	P.Expect(Scanner.LBRACK);
	if P.tok != Scanner.RBRACK {
		t.expr = P.ParseExpression();
	}
	P.Expect(Scanner.RBRACK);
	t.elt = P.ParseType();

	P.Ecart();
	return t;
}


func (P *Parser) ParseChannelType() *Node.Type {
	P.Trace("ChannelType");
	
	t := Node.NewType(P.pos, Scanner.CHAN);
	t.mode = Node.FULL;
	if P.tok == Scanner.CHAN {
		P.Next();
		if P.tok == Scanner.ARROW {
			P.Next();
			t.mode = Node.SEND;
		}
	} else {
		P.Expect(Scanner.ARROW);
		P.Expect(Scanner.CHAN);
		t.mode = Node.RECV;
	}
	t.elt = P.ParseVarType();

	P.Ecart();
	return t;
}


func (P *Parser) ParseVarDeclList(list *Node.List) {
	P.Trace("VarDeclList");

	i0 := list.len();
	list.Add(P.ParseType());
	for P.tok == Scanner.COMMA {
		P.Next();
		list.Add(P.ParseType());
	}

	typ := P.TryType();

	if typ != nil {
		// all list entries must be identifiers
		// convert the type entries into identifiers
		for i, n := i0, list.len(); i < n; i++ {
			t := list.at(i).(*Node.Type);
			if t.tok == Scanner.IDENT && t.expr.tok == Scanner.IDENT {
				list.set(i, t.expr);
			} else {
				list.set(i, Node.NewLit(t.pos, Scanner.IDENT, "bad"));
				P.Error(t.pos, "identifier expected");
			}
		}
		// add type
		list.Add(Node.NewTypeExpr(typ.pos, typ));

	} else {
		// all list entries are types
		// convert all type entries into type expressions
		for i, n := i0, list.len(); i < n; i++ {
			t := list.at(i).(*Node.Type);
			list.set(i, Node.NewTypeExpr(t.pos, t));
		}
		
		if P.tok == Scanner.COMMA {
			panic("internal parser error");
		}
	}
	
	P.Ecart();
}


func (P *Parser) ParseParameterList() *Node.List {
	P.Trace("ParameterList");
	
	list := Node.NewList();
	P.ParseVarDeclList(list);
	for P.tok == Scanner.COMMA {
		P.Next();
		P.ParseVarDeclList(list);
	}
	
	P.Ecart();
	return list;
}


func (P *Parser) ParseParameters() *Node.Type {
	P.Trace("Parameters");
	
	t := Node.NewType(P.pos, Scanner.STRUCT);
	P.Expect(Scanner.LPAREN);
	if P.tok != Scanner.RPAREN {
		t.list = P.ParseParameterList();
	}
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


func (P *Parser) ParseResult() *Node.Type {
	P.Trace("Result");
	
	var t *Node.Type;
	if P.tok == Scanner.LPAREN {
		t = P.ParseParameters();
	} else {
		typ := P.TryType();
		if typ != nil {
			t = Node.NewType(P.pos, Scanner.STRUCT);
			t.list = Node.NewList();
			t.list.Add(Node.NewTypeExpr(typ.pos, typ));
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

func (P *Parser) ParseFunctionType() *Node.Type {
	P.Trace("FunctionType");
	
	t := Node.NewType(P.pos, Scanner.LPAREN);
	t.list = P.ParseParameters().list;  // TODO find better solution
	t.elt = P.ParseResult();
	
	P.Ecart();
	return t;
}


func (P *Parser) ParseMethodDecl() *Node.Decl {
	P.Trace("MethodDecl");
	
	P.ParseIdent();
	P.ParseFunctionType();
	
	P.Ecart();
	return nil;
}


func (P *Parser) ParseInterfaceType() *Node.Type {
	P.Trace("InterfaceType");
	
	t := Node.NewType(P.pos, Scanner.INTERFACE);
	P.Expect(Scanner.INTERFACE);
	if P.tok == Scanner.LBRACE {
		P.Next();
		for P.tok == Scanner.IDENT {
			P.ParseMethodDecl();
			if P.tok != Scanner.RBRACE {
				P.Expect(Scanner.SEMICOLON);
			}
		}
		P.Expect(Scanner.RBRACE);
	}

	P.Ecart();
	return t;
}


func (P *Parser) ParseMapType() *Node.Type {
	P.Trace("MapType");
	
	t := Node.NewType(P.pos, Scanner.MAP);
	P.Expect(Scanner.MAP);
	P.Expect(Scanner.LBRACK);
	t.key = P.ParseVarType();
	P.Expect(Scanner.RBRACK);
	t.elt = P.ParseVarType();
	
	P.Ecart();
	return t;
}


func (P *Parser) ParseStructType() *Node.Type {
	P.Trace("StructType");

	t := Node.NewType(P.pos, Scanner.STRUCT);
	P.Expect(Scanner.STRUCT);
	if P.tok == Scanner.LBRACE {
		P.Next();
		t.list = Node.NewList();
		for P.tok == Scanner.IDENT {
			P.ParseVarDeclList(t.list);
			if P.tok != Scanner.RBRACE {
				P.Expect(Scanner.SEMICOLON);
			}
		}
		P.OptSemicolon();
		P.Expect(Scanner.RBRACE);
	}

	P.Ecart();
	return t;
}


func (P *Parser) ParsePointerType() *Node.Type {
	P.Trace("PointerType");
	
	typ := Node.NewType(P.pos, Scanner.MUL);
	P.Expect(Scanner.MUL);
	typ.elt = P.ParseType();
	
	P.Ecart();
	return typ;
}


// Returns nil if no type was found.
func (P *Parser) TryType() *Node.Type {
	P.Trace("Type (try)");
	
	var t *Node.Type;
	switch P.tok {
	case Scanner.IDENT: t = P.ParseTypeName();
	case Scanner.LBRACK: t = P.ParseArrayType();
	case Scanner.CHAN, Scanner.ARROW: t = P.ParseChannelType();
	case Scanner.INTERFACE: t = P.ParseInterfaceType();
	case Scanner.LPAREN: t = P.ParseFunctionType();
	case Scanner.MAP: t = P.ParseMapType();
	case Scanner.STRUCT: t = P.ParseStructType();
	case Scanner.MUL: t = P.ParsePointerType();
	}

	P.Ecart();
	return t;
}


// ----------------------------------------------------------------------------
// Blocks

func (P *Parser) ParseStatementList() *Node.List {
	P.Trace("StatementList");
	
	list := Node.NewList();
	for P.tok != Scanner.CASE && P.tok != Scanner.DEFAULT && P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
		list.Add(P.ParseStatement());
		if P.tok == Scanner.SEMICOLON {
			P.Next();
		} else if P.opt_semi {
			P.opt_semi = false;  // "consume" optional semicolon
		} else {
			break;
		}
	}
	
	P.Ecart();
	return list;
}


func (P *Parser) ParseBlock() *Node.List {
	P.Trace("Block");
	
	var s *Node.List;
	P.Expect(Scanner.LBRACE);
	if P.tok != Scanner.RBRACE {
		s = P.ParseStatementList();
	}
	P.Expect(Scanner.RBRACE);
	P.opt_semi = true;
	
	P.Ecart();
	return s;
}


// ----------------------------------------------------------------------------
// Expressions

// TODO: Make this non-recursive.
func (P *Parser) ParseExpressionList() *Node.Expr {
	P.Trace("ExpressionList");

	x := P.ParseExpression();
	if P.tok == Scanner.COMMA {
		pos := P.pos;
		P.Next();
		y := P.ParseExpressionList();
		x = Node.NewExpr(pos, Scanner.COMMA, x, y);
	}
	
	P.Ecart();
	return x;
}


func (P *Parser) ParseFunctionLit() *Node.Expr {
	P.Trace("FunctionLit");
	
	P.Expect(Scanner.FUNC);
	P.ParseFunctionType();
	P.scope_lev++;
	P.ParseBlock();
	P.scope_lev--;
	
	P.Ecart();
	return Node.NewLit(P.pos, Scanner.INT, "0");  // "null" expr
}


func (P *Parser) ParseOperand() *Node.Expr {
	P.Trace("Operand");

	var x *Node.Expr;
	switch P.tok {
	case Scanner.IDENT:
		x = P.ParseIdent();
		
	case Scanner.LPAREN:
		P.Next();
		P.expr_lev++;
		x = P.ParseExpression();
		P.expr_lev--;
		P.Expect(Scanner.RPAREN);

	case Scanner.INT, Scanner.FLOAT, Scanner.STRING:
		x = Node.NewLit(P.pos, P.tok, P.val);
		P.Next();
		if x.tok == Scanner.STRING {
			for ; P.tok == Scanner.STRING; P.Next() {
				x.s += P.val;
			}
		}

	case Scanner.FUNC:
		x = P.ParseFunctionLit();
		
	default:
		t := P.TryType();
		if t != nil {
			x = Node.NewTypeExpr(t.pos, t);
		} else {
			P.Error(P.pos, "operand expected");
			P.Next();  // make progress
			x = Node.NewLit(P.pos, Scanner.INT, "0");  // "null" expr
		}
	}

	P.Ecart();
	return x;
}


func (P *Parser) ParseSelectorOrTypeGuard(x *Node.Expr) *Node.Expr {
	P.Trace("SelectorOrTypeGuard");

	pos := P.pos;
	P.Expect(Scanner.PERIOD);
	
	if P.tok == Scanner.IDENT {
		y := P.ParseIdent();
		x = Node.NewExpr(pos, Scanner.PERIOD, x, y);
		
	} else {
		P.Expect(Scanner.LPAREN);
		P.ParseType();
		P.Expect(Scanner.RPAREN);
	}
	
	P.Ecart();
	return x;
}


// mode = 0: single or pair accepted
// mode = 1: single only accepted
// mode = 2: pair only accepted
func (P *Parser) ParseExpressionPair(mode int) *Node.Expr {
	P.Trace("ExpressionPair");

	x := P.ParseExpression();
	if mode == 0 && P.tok == Scanner.COLON || mode == 2 {
		pos := P.pos;
		P.Expect(Scanner.COLON);
		y := P.ParseExpression();
		x = Node.NewExpr(pos, Scanner.COLON, x, y);
	}

	P.Ecart();
	return x;
}


func (P *Parser) ParseIndex(x *Node.Expr) *Node.Expr {
	P.Trace("IndexOrSlice");
	
	pos := P.pos;
	P.Expect(Scanner.LBRACK);
	i := P.ParseExpressionPair(0);
	P.Expect(Scanner.RBRACK);
	
	P.Ecart();
	return Node.NewExpr(pos, Scanner.LBRACK, x, i);
}


func (P *Parser) ParseCall(x *Node.Expr) *Node.Expr {
	P.Trace("Call");

	x = Node.NewExpr(P.pos, Scanner.LPAREN, x, nil);
	P.Expect(Scanner.LPAREN);
	if P.tok != Scanner.RPAREN {
		x.y = P.ParseExpressionList();
	}
	P.Expect(Scanner.RPAREN);
	
	P.Ecart();
	return x;
}


func (P *Parser) ParseCompositeLit(t *Node.Type) *Node.Expr {
	P.Trace("CompositeLit");

	mode := 0;
	P.Expect(Scanner.LBRACE);
	for P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
		x := P.ParseExpressionPair(mode);
		if mode == 0 {
			// first expression determines mode
			if x.tok == Scanner.COLON {
				mode = 2;
			} else {
				mode = 1;
			}
		}
		if P.tok == Scanner.COMMA {
			P.Next();
		} else {
			break;
		}
	}
	P.Expect(Scanner.RBRACE);
	
	P.Ecart();
	return Node.NewLit(P.pos, Scanner.INT, "0");  // "null" expr
}


func (P *Parser) ParsePrimaryExpr() *Node.Expr {
	P.Trace("PrimaryExpr");
	
	x := P.ParseOperand();
	for {
		switch P.tok {
		case Scanner.PERIOD: x = P.ParseSelectorOrTypeGuard(x);
		case Scanner.LBRACK: x = P.ParseIndex(x);
		case Scanner.LPAREN: x = P.ParseCall(x);
		case Scanner.LBRACE:
			if P.expr_lev > 0 {
				var t *Node.Type;
				if x.tok == Scanner.TYPE {
					t = x.t;
				} else if x.tok == Scanner.IDENT {
					// assume a type name
					t = Node.NewType(x.pos, Scanner.IDENT);
					t.expr = x;
				} else {
					P.Error(x.pos, "type expected for composite literal");
				}
				x = P.ParseCompositeLit(t);
			} else {
				// composites inside control clauses must be parenthesized
				goto exit;
			}
		default: goto exit;
		}
	}
exit:

	P.Ecart();
	return x;
}


func (P *Parser) ParseUnaryExpr() *Node.Expr {
	P.Trace("UnaryExpr");
	
	var x *Node.Expr;
	switch P.tok {
	case
		Scanner.ADD, Scanner.SUB,
		Scanner.NOT, Scanner.XOR,
		Scanner.MUL, Scanner.ARROW,
		Scanner.AND:
			pos, tok := P.pos, P.tok;
			P.Next();
			y := P.ParseUnaryExpr();
			x = Node.NewExpr(pos, tok, nil, y);
			
		default:
			x = P.ParsePrimaryExpr();
	}
	
	P.Ecart();
	return x;
}


func (P *Parser) ParseBinaryExpr(prec1 int) *Node.Expr {
	P.Trace("BinaryExpr");
	
	x := P.ParseUnaryExpr();
	for prec := Scanner.Precedence(P.tok); prec >= prec1; prec-- {
		for Scanner.Precedence(P.tok) == prec {
			pos, tok := P.pos, P.tok;
			P.Next();
			y := P.ParseBinaryExpr(prec + 1);
			x = Node.NewExpr(pos, tok, x, y);
		}
	}
	
	P.Ecart();
	return x;
}


func (P *Parser) ParseExpression() *Node.Expr {
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

func (P *Parser) ParseSimpleStat() *Node.Stat {
	P.Trace("SimpleStat");
	
	var s *Node.Stat;
	x := P.ParseExpressionList();
	
	switch P.tok {
	case Scanner.COLON:
		// label declaration
		if x.len() == 1 {
			s = Node.NewStat(P.pos, Scanner.COLON);
			s.expr = x;
		} else {
			P.Error(P.pos, "illegal label declaration");
		}
		P.Next();  // consume ":"
		P.opt_semi = true;
		
	case
		Scanner.DEFINE, Scanner.ASSIGN, Scanner.ADD_ASSIGN,
		Scanner.SUB_ASSIGN, Scanner.MUL_ASSIGN, Scanner.QUO_ASSIGN,
		Scanner.REM_ASSIGN, Scanner.AND_ASSIGN, Scanner.OR_ASSIGN,
		Scanner.XOR_ASSIGN, Scanner.SHL_ASSIGN, Scanner.SHR_ASSIGN:
		s = Node.NewStat(P.pos, P.tok);
		P.Next();
		s.lhs = x;
		s.expr = P.ParseExpressionList();

	default:
		if P.tok == Scanner.INC || P.tok == Scanner.DEC {
			s = Node.NewStat(P.pos, P.tok);
			if x.len() == 1 {
				s.expr = x;
			} else {
				P.Error(P.pos, "more then one operand");
			}
			P.Next();  // consume "++" or "--"
		} else {
			s = Node.NewStat(P.pos, 0);  // TODO give this a token value
			if x.len() == 1 {
				s.expr = x;
			} else {
				P.Error(P.pos, "syntax error");
			}
		}
	}
	
	P.Ecart();
	return s;
}


func (P *Parser) ParseGoStat() *Node.Stat {
	P.Trace("GoStat");
	
	s := Node.NewStat(P.pos, Scanner.GO);
	P.Expect(Scanner.GO);
	s.expr = P.ParseExpression();
	
	P.Ecart();
	return s;
}


func (P *Parser) ParseReturnStat() *Node.Stat {
	P.Trace("ReturnStat");
	
	s := Node.NewStat(P.pos, Scanner.RETURN);
	P.Expect(Scanner.RETURN);
	if P.tok != Scanner.SEMICOLON && P.tok != Scanner.RBRACE {
		s.expr = P.ParseExpressionList();
	}
	
	P.Ecart();
	return s;
}


func (P *Parser) ParseControlFlowStat(tok int) *Node.Stat {
	P.Trace("ControlFlowStat");
	
	s := Node.NewStat(P.pos, tok);
	P.Expect(tok);
	if tok != Scanner.FALLTHROUGH && P.tok == Scanner.IDENT {
		s.expr = P.ParseIdent();
	}
	
	P.Ecart();
	return s;
}


func (P *Parser) ParseControlClause(keyword int) *Node.Stat {
	P.Trace("ControlClause");
	
	s := Node.NewStat(P.pos, keyword);
	P.Expect(keyword);
	if P.tok != Scanner.LBRACE {
		prev_lev := P.expr_lev;
		P.expr_lev = 0;
		if P.tok != Scanner.SEMICOLON {
			s.init = P.ParseSimpleStat();
		}
		if P.tok == Scanner.SEMICOLON {
			P.Next();
			if P.tok != Scanner.SEMICOLON && P.tok != Scanner.LBRACE {
				s.expr = P.ParseExpression();
			}
			if keyword == Scanner.FOR {
				P.Expect(Scanner.SEMICOLON);
				if P.tok != Scanner.LBRACE {
					s.post = P.ParseSimpleStat();
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


func (P *Parser) ParseIfStat() *Node.Stat {
	P.Trace("IfStat");

	s := P.ParseControlClause(Scanner.IF);
	s.block = P.ParseBlock();
	if P.tok == Scanner.ELSE {
		P.Next();
		if P.tok == Scanner.IF {
			s.post = P.ParseIfStat();
		} else {
			// For 6g compliance - should really be P.ParseBlock()
			t := P.ParseStatement();
			if t.tok != Scanner.LBRACE {
				// wrap in a block if we don't have one
				t1 := Node.NewStat(P.pos, Scanner.LBRACE);
				t1.block = Node.NewList();
				t1.block.Add(t);
				t = t1;
			}
			s.post = t;
		}
	}
	
	P.Ecart();
	return s;
}


func (P *Parser) ParseForStat() *Node.Stat {
	P.Trace("ForStat");
	
	s := P.ParseControlClause(Scanner.FOR);
	s.block = P.ParseBlock();
	
	P.Ecart();
	return s;
}


func (P *Parser) ParseCase() *Node.Stat {
	P.Trace("Case");
	
	s := Node.NewStat(P.pos, P.tok);
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


func (P *Parser) ParseCaseClause() *Node.Stat {
	P.Trace("CaseClause");

	s := P.ParseCase();
	if P.tok != Scanner.CASE && P.tok != Scanner.DEFAULT && P.tok != Scanner.RBRACE {
		s.block = P.ParseStatementList();
	}
	
	P.Ecart();
	return s;
}


func (P *Parser) ParseSwitchStat() *Node.Stat {
	P.Trace("SwitchStat");
	
	s := P.ParseControlClause(Scanner.SWITCH);
	s.block = Node.NewList();
	P.Expect(Scanner.LBRACE);
	for P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
		s.block.Add(P.ParseCaseClause());
	}
	P.Expect(Scanner.RBRACE);
	P.opt_semi = true;

	P.Ecart();
	return s;
}


func (P *Parser) ParseCommCase() *Node.Stat {
	P.Trace("CommCase");

	s := Node.NewStat(P.pos, Scanner.CASE);
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
	return s;
}


func (P *Parser) ParseCommClause() *Node.Stat {
	P.Trace("CommClause");
	
	s := P.ParseCommCase();
	if P.tok != Scanner.CASE && P.tok != Scanner.DEFAULT && P.tok != Scanner.RBRACE {
		s.block = P.ParseStatementList();
	}
	
	P.Ecart();
	return s;
}


func (P *Parser) ParseSelectStat() *Node.Stat {
	P.Trace("SelectStat");
	
	s := Node.NewStat(P.pos, Scanner.SELECT);
	s.block = Node.NewList();
	P.Expect(Scanner.SELECT);
	P.Expect(Scanner.LBRACE);
	for P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
		s.block.Add(P.ParseCommClause());
	}
	P.Expect(Scanner.RBRACE);
	P.opt_semi = true;
	
	P.Ecart();
	return s;
}


func (P *Parser) ParseRangeStat() *Node.Stat {
	P.Trace("RangeStat");
	
	s := Node.NewStat(P.pos, Scanner.RANGE);
	P.Expect(Scanner.RANGE);
	P.ParseIdentList();
	P.Expect(Scanner.DEFINE);
	s.expr = P.ParseExpression();
	s.block = P.ParseBlock();
	
	P.Ecart();
	return s;
}


func (P *Parser) ParseEmptyStat() {
	P.Trace("EmptyStat");
	P.Ecart();
}


func (P *Parser) ParseStatement() *Node.Stat {
	P.Trace("Statement");
	indent := P.indent;

	var s *Node.Stat;
	switch P.tok {
	case Scanner.CONST, Scanner.TYPE, Scanner.VAR:
		s = Node.NewStat(P.pos, P.tok);
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
		s = P.ParseSimpleStat();
	case Scanner.GO:
		s = P.ParseGoStat();
	case Scanner.RETURN:
		s = P.ParseReturnStat();
	case Scanner.BREAK, Scanner.CONTINUE, Scanner.GOTO, Scanner.FALLTHROUGH:
		s = P.ParseControlFlowStat(P.tok);
	case Scanner.LBRACE:
		s = Node.NewStat(P.pos, Scanner.LBRACE);
		s.block = P.ParseBlock();
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
		P.ParseEmptyStat();  // for complete tracing output only
	}

	if indent != P.indent {
		panic("imbalanced tracing code (Statement)");
	}
	P.Ecart();
	return s;
}


// ----------------------------------------------------------------------------
// Declarations

func (P *Parser) ParseImportSpec() *Node.Decl {
	P.Trace("ImportSpec");
	
	d := Node.NewDecl(P.pos, Scanner.IMPORT, false);
	if P.tok == Scanner.PERIOD {
		P.Error(P.pos, `"import ." not yet handled properly`);
		P.Next();
	} else if P.tok == Scanner.IDENT {
		d.ident = P.ParseIdent();
	}
	
	if P.tok == Scanner.STRING {
		// TODO eventually the scanner should strip the quotes
		d.val = Node.NewLit(P.pos, Scanner.STRING, P.val);
		P.Next();
	} else {
		P.Expect(Scanner.STRING);  // use Expect() error handling
	}
	
	P.Ecart();
	return d;
}


func (P *Parser) ParseConstSpec(exported bool) *Node.Decl {
	P.Trace("ConstSpec");
	
	d := Node.NewDecl(P.pos, Scanner.CONST, exported);
	d.ident = P.ParseIdent();
	d.typ = P.TryType();
	if P.tok == Scanner.ASSIGN {
		P.Next();
		d.val = P.ParseExpression();
	}
	
	P.Ecart();
	return d;
}


func (P *Parser) ParseTypeSpec(exported bool) *Node.Decl {
	P.Trace("TypeSpec");

	d := Node.NewDecl(P.pos, Scanner.TYPE, exported);
	d.ident = P.ParseIdent();
	d.typ = P.ParseType();
	P.opt_semi = true;
	
	P.Ecart();
	return d;
}


func (P *Parser) ParseVarSpec(exported bool) *Node.Decl {
	P.Trace("VarSpec");
	
	d := Node.NewDecl(P.pos, Scanner.VAR, exported);
	P.ParseIdentList();
	if P.tok == Scanner.ASSIGN {
		P.Next();
		P.ParseExpressionList();
	} else {
		P.ParseVarType();
		if P.tok == Scanner.ASSIGN {
			P.Next();
			P.ParseExpressionList();
		}
	}
	
	P.Ecart();
	return d;
}


// TODO Replace this by using function pointers derived from methods.
func (P *Parser) ParseSpec(exported bool, keyword int) *Node.Decl {
	switch keyword {
	case Scanner.IMPORT: return P.ParseImportSpec();
	case Scanner.CONST: return P.ParseConstSpec(exported);
	case Scanner.TYPE: return P.ParseTypeSpec(exported);
	case Scanner.VAR: return P.ParseVarSpec(exported);
	}
	panic("UNREACHABLE");
	return nil;
}


func (P *Parser) ParseDecl(exported bool, keyword int) *Node.Decl {
	P.Trace("Decl");
	
	var d *Node.Decl;
	P.Expect(keyword);
	if P.tok == Scanner.LPAREN {
		P.Next();
		d = Node.NewDecl(P.pos, keyword, exported);
		d.list = Node.NewList();
		for P.tok != Scanner.RPAREN && P.tok != Scanner.EOF {
			d.list.Add(P.ParseSpec(exported, keyword));
			if P.tok == Scanner.SEMICOLON {
				P.Next();
			} else {
				break;
			}
		}
		P.Expect(Scanner.RPAREN);
		P.opt_semi = true;
		
	} else {
		d = P.ParseSpec(exported, keyword);
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

func (P *Parser) ParseFunctionDecl(exported bool) *Node.Decl {
	P.Trace("FunctionDecl");
	
	d := Node.NewDecl(P.pos, Scanner.FUNC, exported);
	P.Expect(Scanner.FUNC);
	if P.tok == Scanner.LPAREN {
		pos := P.pos;
		recv := P.ParseParameters();
		// TODO: fix this
		/*
		if recv.list.len() != 1 {
			P.Error(pos, "must have exactly one receiver");
		}
		*/
	}
	
	d.ident = P.ParseIdent();
	d.typ = P.ParseFunctionType();
	
	if P.tok == Scanner.LBRACE {
		P.scope_lev++;
		d.list = P.ParseBlock();
		P.scope_lev--;
	}
	
	P.Ecart();
	return d;
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
		P.ParseIdent();
		if P.tok == Scanner.COMMA {
			P.Next();  // TODO this seems wrong
		}
	}
	if has_paren {
		P.Expect(Scanner.RPAREN)
	}
	
	P.Ecart();
}


func (P *Parser) ParseDeclaration() *Node.Decl {
	P.Trace("Declaration");
	indent := P.indent;
	
	var d *Node.Decl;

	exported := false;
	if P.tok == Scanner.EXPORT {
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
	return d;
}


// ----------------------------------------------------------------------------
// Program

func (P *Parser) ParseProgram() *Node.Program {
	P.Trace("Program");
	
	p := Node.NewProgram(P.pos);
	P.Expect(Scanner.PACKAGE);
	p.ident = P.ParseIdent();
	
	p.decls = Node.NewList();
	for P.tok == Scanner.IMPORT {
		p.decls.Add(P.ParseDecl(false, Scanner.IMPORT));
		P.OptSemicolon();
	}
		
	for P.tok != Scanner.EOF {
		p.decls.Add(P.ParseDeclaration());
		P.OptSemicolon();
	}
	
	P.Ecart();
	return p;
}
