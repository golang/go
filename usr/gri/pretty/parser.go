// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Parser

import Scanner "scanner"
import Node "node"


export type Parser struct {
	// Tracing/debugging
	verbose, sixg bool;
	indent uint;
	
	// Scanner
	scanner *Scanner.Scanner;
	tokchan *<-chan *Scanner.Token;
	comments *Node.List;
	
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
		P.comments.Add(Node.NewComment(P.pos, P.val));
	}
}


func (P *Parser) Open(verbose, sixg bool, scanner *Scanner.Scanner, tokchan *<-chan *Scanner.Token) {
	P.verbose = verbose;
	P.sixg = sixg;
	P.indent = 0;
	
	P.scanner = scanner;
	P.tokchan = tokchan;
	P.comments = Node.NewList();
	
	P.Next();
	P.expr_lev = 1;
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
// AST support

func ExprType(x *Node.Expr) *Node.Type {
	var t *Node.Type;
	if x.tok == Scanner.TYPE {
		t = x.t;
	} else if x.tok == Scanner.IDENT {
		// assume a type name
		t = Node.NewType(x.pos, Scanner.IDENT);
		t.expr = x;
	} else if x.tok == Scanner.PERIOD && x.y != nil && ExprType(x.x) != nil {
		// possibly a qualified (type) identifier
		t = Node.NewType(x.pos, Scanner.IDENT);
		t.expr = x;
	}
	return t;
}


func (P *Parser) NoType(x *Node.Expr) *Node.Expr {
	if x != nil && x.tok == Scanner.TYPE {
		P.Error(x.pos, "expected expression, found type");
		x = Node.NewLit(x.pos, Scanner.INT, "");
	}
	return x;
}


func (P *Parser) NewExpr(pos, tok int, x, y *Node.Expr) *Node.Expr {
	return Node.NewExpr(pos, tok, P.NoType(x), P.NoType(y));
}


// ----------------------------------------------------------------------------
// Common productions

func (P *Parser) TryType() *Node.Type;
func (P *Parser) ParseExpression(prec int) *Node.Expr;
func (P *Parser) ParseStatement() *Node.Stat;
func (P *Parser) ParseDeclaration() *Node.Decl;


func (P *Parser) ParseIdent() *Node.Expr {
	P.Trace("Ident");

	x := Node.BadExpr;
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


func (P *Parser) ParseIdentList() *Node.Expr {
	P.Trace("IdentList");

	x := P.ParseIdent();
	if P.tok == Scanner.COMMA {
		pos := P.pos;
		P.Next();
		y := P.ParseIdentList();
		x = P.NewExpr(pos, Scanner.COMMA, x, y);
	}

	P.Ecart();
	return x;
}


// ----------------------------------------------------------------------------
// Types

func (P *Parser) ParseType() *Node.Type {
	P.Trace("Type");
	
	t := P.TryType();
	if t == nil {
		P.Error(P.pos, "type expected");
		t = Node.BadType;
	}
	
	P.Ecart();
	return t;
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
		x = P.NewExpr(pos, Scanner.PERIOD, x, y);
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
		t.expr = P.ParseExpression(1);
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


// TODO: The code below (ParseVarDecl, ParseVarDeclList) is all too
// complicated. There must be a better way to do this.

func (P *Parser) ParseVarDecl(expect_ident bool) *Node.Type {
	t := Node.BadType;
	if expect_ident {
		x := P.ParseIdent();
		t = Node.NewType(x.pos, Scanner.IDENT);
		t.expr = x;
	} else {
		t = P.ParseType();
	}
	return t;
}


func (P *Parser) ParseVarDeclList(list *Node.List) {
	P.Trace("VarDeclList");

	// parse a list of types
	i0 := list.len();
	list.Add(P.ParseVarDecl(i0 > 0));
	for P.tok == Scanner.COMMA {
		P.Next();
		list.Add(P.ParseVarDecl(i0 > 0));
	}

	var typ *Node.Type;
	if i0 > 0 {
		// not the first parameter section; we must have a type
		typ = P.ParseType();
	} else {
		// first parameter section; we may have a type
		typ = P.TryType();
	}

	// convert the list into a list of (type) expressions
	if typ != nil {
		// all list entries must be identifiers
		// convert the type entries into identifiers
		for i, n := i0, list.len(); i < n; i++ {
			t := list.at(i).(*Node.Type);
			if t.tok == Scanner.IDENT && t.expr.tok == Scanner.IDENT {
				list.set(i, t.expr);
			} else {
				list.set(i, Node.BadExpr);
				P.Error(t.pos, "identifier expected");
			}
		}
		// add type
		list.Add(Node.NewTypeExpr(typ));

	} else {
		// all list entries are types
		// convert all type entries into type expressions
		for i, n := i0, list.len(); i < n; i++ {
			t := list.at(i).(*Node.Type);
			list.set(i, Node.NewTypeExpr(t));
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
			t.list.Add(Node.NewTypeExpr(typ));
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


func (P *Parser) ParseMethodSpec(list *Node.List) {
	P.Trace("MethodDecl");
	
	list.Add(P.ParseIdent());
	list.Add(Node.NewTypeExpr(P.ParseFunctionType()));
	
	P.Ecart();
}


func (P *Parser) ParseInterfaceType() *Node.Type {
	P.Trace("InterfaceType");
	
	t := Node.NewType(P.pos, Scanner.INTERFACE);
	P.Expect(Scanner.INTERFACE);
	if P.tok == Scanner.LBRACE {
		P.Next();
		t.list = Node.NewList();
		for P.tok == Scanner.IDENT {
			P.ParseMethodSpec(t.list);
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
	
	t := Node.NewType(P.pos, Scanner.MUL);
	P.Expect(Scanner.MUL);
	t.elt = P.ParseType();
	
	P.Ecart();
	return t;
}


func (P *Parser) TryType() *Node.Type {
	P.Trace("Type (try)");
	
	t := Node.BadType;
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

func (P *Parser) ParseStatementList() *Node.List {
	P.Trace("StatementList");
	
	list := Node.NewList();
	for P.tok != Scanner.CASE && P.tok != Scanner.DEFAULT && P.tok != Scanner.RBRACE && P.tok != Scanner.EOF {
		s := P.ParseStatement();
		if s != nil {
			// not the empty statement
			list.Add(s);
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


func (P *Parser) ParseBlock() *Node.List {
	P.Trace("Block");
	
	P.Expect(Scanner.LBRACE);
	s := P.ParseStatementList();
	P.Expect(Scanner.RBRACE);
	P.opt_semi = true;
	
	P.Ecart();
	return s;
}


// ----------------------------------------------------------------------------
// Expressions

// TODO make this non-recursive
func (P *Parser) ParseExpressionList() *Node.Expr {
	P.Trace("ExpressionList");

	x := P.ParseExpression(1);
	if P.tok == Scanner.COMMA {
		pos := P.pos;
		P.Next();
		y := P.ParseExpressionList();
		x = P.NewExpr(pos, Scanner.COMMA, x, y);
	}
	
	P.Ecart();
	return x;
}


func (P *Parser) ParseFunctionLit() *Node.Expr {
	P.Trace("FunctionLit");
	
	x := Node.NewLit(P.pos, Scanner.FUNC, "");
	P.Expect(Scanner.FUNC);
	x.t = P.ParseFunctionType();
	P.scope_lev++;
	x.block = P.ParseBlock();
	P.scope_lev--;
	
	P.Ecart();
	return x;
}


func (P *Parser) ParseOperand() *Node.Expr {
	P.Trace("Operand");

	x := Node.BadExpr;
	switch P.tok {
	case Scanner.IDENT:
		x = P.ParseIdent();
		
	case Scanner.LPAREN:
		// TODO we could have a function type here as in: new(*())
		// (currently not working)
		P.Next();
		P.expr_lev++;
		x = P.ParseExpression(1);
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
			x = Node.NewTypeExpr(t);
		} else {
			P.Error(P.pos, "operand expected");
			P.Next();  // make progress
		}
	}

	P.Ecart();
	return x;
}


func (P *Parser) ParseSelectorOrTypeGuard(x *Node.Expr) *Node.Expr {
	P.Trace("SelectorOrTypeGuard");

	x = P.NewExpr(P.pos, Scanner.PERIOD, x, nil);
	P.Expect(Scanner.PERIOD);
	
	if P.tok == Scanner.IDENT {
		x.y = P.ParseIdent();
		
	} else {
		P.Expect(Scanner.LPAREN);
		x.t = P.ParseType();
		P.Expect(Scanner.RPAREN);
	}
	
	P.Ecart();
	return x;
}


func (P *Parser) ParseIndex(x *Node.Expr) *Node.Expr {
	P.Trace("IndexOrSlice");
	
	pos := P.pos;
	P.Expect(Scanner.LBRACK);
	i := P.ParseExpression(0);
	P.Expect(Scanner.RBRACK);
	
	P.Ecart();
	return P.NewExpr(pos, Scanner.LBRACK, x, i);
}


func (P *Parser) ParseBinaryExpr(prec1 int) *Node.Expr

func (P *Parser) ParseCall(x *Node.Expr) *Node.Expr {
	P.Trace("Call");

	x = P.NewExpr(P.pos, Scanner.LPAREN, x, nil);
	P.Expect(Scanner.LPAREN);
	if P.tok != Scanner.RPAREN {
		// the very first argument may be a type if the function called is new()
		// call ParseBinaryExpr() which allows type expressions (instead of ParseExpression)
		y := P.ParseBinaryExpr(1);
		if P.tok == Scanner.COMMA {
			pos := P.pos;
			P.Next();
			z := P.ParseExpressionList();
			// create list manually because NewExpr checks for type expressions
			z = P.NewExpr(pos, Scanner.COMMA, nil, z);
			z.x = y;
			y = z;
		}
		x.y = y;
	}
	P.Expect(Scanner.RPAREN);
	
	P.Ecart();
	return x;
}


func (P *Parser) ParseCompositeList(mode int) *Node.Expr {
	x := P.ParseExpression(0);
	
	switch mode {
	case 0:  // first element determines mode
		mode = 1;
		if x.tok == Scanner.COLON {
			mode = 2;
		}
	case 1:
		if x.tok == Scanner.COLON {
			P.Error(x.x.pos, "single value expected; found pair");
		}
	case 2:
		if x.tok != Scanner.COLON {
			P.Error(x.pos, "key:value pair expected; found single value");
		}
	}

	if P.tok == Scanner.COMMA {
		pos := P.pos;
		P.Next();
		if P.tok != Scanner.RBRACE {
			y := P.ParseCompositeList(mode);
			x = P.NewExpr(pos, Scanner.COMMA, x, y);
		}
	}

	return x;
}


func (P *Parser) ParseCompositeLit(t *Node.Type) *Node.Expr {
	P.Trace("CompositeLit");
	
	x := P.NewExpr(P.pos, Scanner.LBRACE, nil, nil);
	x.t = t;
	P.Expect(Scanner.LBRACE);
	if P.tok != Scanner.RBRACE {
		x.y = P.ParseCompositeList(0);
	}
	P.Expect(Scanner.RBRACE);
	
	P.Ecart();
	return x;
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
			// assume a composite literal only if x could be a type
			// and if we are not inside control clause (expr_lev > 0)
			// (composites inside control clauses must be parenthesized)
			var t *Node.Type;
			if P.expr_lev > 0 {
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


func (P *Parser) ParseUnaryExpr() *Node.Expr {
	P.Trace("UnaryExpr");
	
	x := Node.BadExpr;
	switch P.tok {
	case Scanner.ADD, Scanner.SUB, Scanner.MUL, Scanner.NOT, Scanner.XOR, Scanner.ARROW, Scanner.AND:
		pos, tok := P.pos, P.tok;
		P.Next();
		y := P.ParseUnaryExpr();
		if tok == Scanner.MUL && y.tok == Scanner.TYPE {
			// pointer type
			t := Node.NewType(pos, Scanner.MUL);
			t.elt = y.t;
			x = Node.NewTypeExpr(t);
		} else {
			x = P.NewExpr(pos, tok, nil, y);
		}
	
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
			x = P.NewExpr(pos, tok, x, y);
		}
	}
	
	P.Ecart();
	return x;
}


func (P *Parser) ParseExpression(prec int) *Node.Expr {
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

func (P *Parser) ParseSimpleStat() *Node.Stat {
	P.Trace("SimpleStat");
	
	s := Node.BadStat;
	x := P.ParseExpressionList();
	
	switch P.tok {
	case Scanner.COLON:
		// label declaration
		s = Node.NewStat(P.pos, Scanner.COLON);
		s.expr = x;
		if x.len() != 1 {
			P.Error(x.pos, "illegal label declaration");
		}
		P.Next();  // consume ":"
		P.opt_semi = true;
		
	case
		Scanner.DEFINE, Scanner.ASSIGN, Scanner.ADD_ASSIGN,
		Scanner.SUB_ASSIGN, Scanner.MUL_ASSIGN, Scanner.QUO_ASSIGN,
		Scanner.REM_ASSIGN, Scanner.AND_ASSIGN, Scanner.OR_ASSIGN,
		Scanner.XOR_ASSIGN, Scanner.SHL_ASSIGN, Scanner.SHR_ASSIGN:
		// assignment
		pos, tok := P.pos, P.tok;
		P.Next();
		y := P.ParseExpressionList();
		if xl, yl := x.len(), y.len(); xl > 1 && yl > 1 && xl != yl {
			P.Error(x.pos, "arity of lhs doesn't match rhs");
		}
		s = Node.NewStat(x.pos, Scanner.EXPRSTAT);
		s.expr = Node.NewExpr(pos, tok, x, y);

	default:
		var pos, tok int;
		if P.tok == Scanner.INC || P.tok == Scanner.DEC {
			pos, tok = P.pos, P.tok;
			P.Next();
		} else {
			pos, tok = x.pos, Scanner.EXPRSTAT;
		}
		s = Node.NewStat(pos, tok);
		s.expr = x;
		if x.len() != 1 {
			P.Error(x.pos, "only one expression allowed");
		}
	}
	
	P.Ecart();
	return s;
}


func (P *Parser) ParseGoStat() *Node.Stat {
	P.Trace("GoStat");
	
	s := Node.NewStat(P.pos, Scanner.GO);
	P.Expect(Scanner.GO);
	s.expr = P.ParseExpression(1);
	
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
				s.expr = P.ParseExpression(1);
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
		s1 := Node.BadStat;
		if P.sixg {
			s1 = P.ParseStatement();
			if s1 != nil {
				// not the empty statement
				if s1.tok != Scanner.LBRACE {
					// wrap in a block if we don't have one
					b := Node.NewStat(P.pos, Scanner.LBRACE);
					b.block = Node.NewList();
					b.block.Add(s1);
					s1 = b;
				}
				s.post = s1;
			}
		} else if P.tok == Scanner.IF {
			s1 = P.ParseIfStat();
		} else {
			s1 = Node.NewStat(P.pos, Scanner.LBRACE);
			s1.block = P.ParseBlock();
		}
		s.post = s1;
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
		P.ParseExpression(1);
		if P.tok == Scanner.ASSIGN || P.tok == Scanner.DEFINE {
			P.Next();
			P.Expect(Scanner.ARROW);
			P.ParseExpression(1);
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
	s.expr = P.ParseExpression(1);
	s.block = P.ParseBlock();
	
	P.Ecart();
	return s;
}


func (P *Parser) ParseStatement() *Node.Stat {
	P.Trace("Statement");
	indent := P.indent;

	s := Node.BadStat;
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
		d.val = P.ParseExpression(1);
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
	
	P.Ecart();
	return d;
}


// TODO replace this by using function pointers derived from methods
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
	
	d := Node.BadDecl;
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
	
	var recv *Node.Type;
	if P.tok == Scanner.LPAREN {
		pos := P.pos;
		recv = P.ParseParameters();
		if recv.nfields() != 1 {
			P.Error(pos, "must have exactly one receiver");
		}
	}
	
	d.ident = P.ParseIdent();
	d.typ = P.ParseFunctionType();
	d.typ.key = recv;

	if P.tok == Scanner.LBRACE {
		P.scope_lev++;
		d.list = P.ParseBlock();
		P.scope_lev--;
	}
	
	P.Ecart();
	return d;
}


func (P *Parser) ParseExportDecl() *Node.Decl {
	P.Trace("ExportDecl");
	
	d := Node.NewDecl(P.pos, Scanner.EXPORT, false);
	d.ident = P.ParseIdentList();

	P.Ecart();
	return d;
}


func (P *Parser) ParseDeclaration() *Node.Decl {
	P.Trace("Declaration");
	indent := P.indent;
	
	d := Node.BadDecl;
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
	
	p.comments = P.comments;
	
	P.Ecart();
	return p;
}
