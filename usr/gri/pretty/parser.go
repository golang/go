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
	
	// Token
	tok int;  // one token look-ahead
	pos int;  // token source position
	val string;  // token value (for IDENT, NUMBER, STRING only)

	// Nesting level
	level int;  // 0 = global scope, -1 = function/struct scope of global functions/structs, etc.
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
		P.tok, P.pos, P.val = P.scanner.Scan();
	} else {
		t := <-P.tokchan;
		P.tok, P.pos, P.val = t.tok, t.pos, t.val;
	}
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


func (P *Parser) Optional(tok int) {
	if P.tok == tok {
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

func (P *Parser) TryType() bool;
func (P *Parser) ParseExpression() AST.Expr;
func (P *Parser) TryStatement() bool;
func (P *Parser) ParseDeclaration();


func (P *Parser) ParseIdent() *AST.Ident {
	P.Trace("Ident");

	ident := new(AST.Ident);
	ident.pos_, ident.val_ = P.pos, "";
	if P.tok == Scanner.IDENT {
		ident.val_ = P.val;
		if P.verbose {
			P.PrintIndent();
			print("Ident = \"", ident.val_, "\"\n");
		}
		P.Next();
	} else {
		P.Expect(Scanner.IDENT);  // use Expect() error handling
	}
	
	P.Ecart();
	return ident;
}


func (P *Parser) ParseIdentList() int {
	P.Trace("IdentList");

	P.ParseIdent();
	n := 1;
	for P.tok == Scanner.COMMA {
		P.Next();
		P.ParseIdent();
		n++;
	}

	P.Ecart();
	return n;
}


func (P *Parser) ParseQualifiedIdent(ident *AST.Ident) AST.Expr {
	P.Trace("QualifiedIdent");

	if ident == nil {
		ident = P.ParseIdent();
	}
	if P.tok == Scanner.PERIOD {
	   	 P.Next();
		 ident = P.ParseIdent();
	}
	
	P.Ecart();
	return ident;
}


// ----------------------------------------------------------------------------
// Types

func (P *Parser) ParseType() {
	P.Trace("Type");
	
	typ := P.TryType();
	if !typ {
		P.Error(P.pos, "type expected");
	}
	
	P.Ecart();
}


func (P *Parser) ParseVarType() {
	P.Trace("VarType");
	
	P.ParseType();
	
	P.Ecart();
}


func (P *Parser) ParseTypeName() AST.Expr {
	P.Trace("TypeName");
	
	x := P.ParseQualifiedIdent(nil);

	P.Ecart();
	return x;
}


func (P *Parser) ParseArrayType() {
	P.Trace("ArrayType");
	
	P.Expect(Scanner.LBRACK);
	if P.tok != Scanner.RBRACK {
		// TODO set typ.len
		P.ParseExpression();
	}
	P.Expect(Scanner.RBRACK);
	P.ParseType();

	P.Ecart();	
}


func (P *Parser) ParseChannelType() {
	P.Trace("ChannelType");
	
	if P.tok == Scanner.CHAN {
		P.Next();
		if P.tok == Scanner.ARROW {
			P.Next();
		}
	} else {
		P.Expect(Scanner.ARROW);
		P.Expect(Scanner.CHAN);
	}
	P.ParseVarType();

	P.Ecart();	
}


func (P *Parser) ParseVarDeclList() int {
	P.Trace("VarDeclList");
	
	n := P.ParseIdentList();
	P.ParseVarType();
	
	P.Ecart();
	return n;
}


func (P *Parser) ParseParameterList() int {
	P.Trace("ParameterList");
	
	n := P.ParseVarDeclList();
	for P.tok == Scanner.COMMA {
		P.Next();
		n += P.ParseVarDeclList();
	}
	
	P.Ecart();
	return n;
}


func (P *Parser) ParseParameters() int {
	P.Trace("Parameters");
	
	n := 0;
	P.Expect(Scanner.LPAREN);
	if P.tok != Scanner.RPAREN {
		n = P.ParseParameterList();
	}
	P.Expect(Scanner.RPAREN);
	
	P.Ecart();
	return n;
}


func (P *Parser) ParseResult() {
	P.Trace("Result");
	
	if P.tok == Scanner.LPAREN {
		// one or more named results
		// TODO: here we allow empty returns - should probably fix this
		P.ParseParameters();

	} else {
		// anonymous result
		P.TryType();
	}

	P.Ecart();
}


// Signatures
//
// (params)
// (params) type
// (params) (results)

func (P *Parser) ParseSignature() {
	P.Trace("Signature");
	
	P.OpenScope();
	P.level--;

	P.ParseParameters();
	P.ParseResult();

	P.level++;
	P.CloseScope();
	
	P.Ecart();
}


// Named signatures
//
//        ident (params)
//        ident (params) type
//        ident (params) (results)
// (recv) ident (params)
// (recv) ident (params) type
// (recv) ident (params) (results)

func (P *Parser) ParseNamedSignature() *AST.Ident {
	P.Trace("NamedSignature");
	
	P.OpenScope();
	P.level--;

	if P.tok == Scanner.LPAREN {
		recv_pos := P.pos;
		n := P.ParseParameters();
		if n != 1 {
			P.Error(recv_pos, "must have exactly one receiver");
			panic("UNIMPLEMENTED (ParseNamedSignature)");
			// TODO do something useful here
		}
	}
	
	ident := P.ParseIdent();

	P.ParseParameters();
	
	P.ParseResult();
	P.level++;
	P.CloseScope();
	
	P.Ecart();
	return ident;
}


func (P *Parser) ParseFunctionType() {
	P.Trace("FunctionType");
	
	typ := P.ParseSignature();

	P.Ecart();
}


func (P *Parser) ParseMethodDecl() {
	P.Trace("MethodDecl");
	
	ident := P.ParseIdent();
	P.OpenScope();
	P.level--;
	
	P.ParseParameters();
	
	//r0 := sig.entries.len;
	P.ParseResult();
	P.level++;
	P.CloseScope();
	P.Optional(Scanner.SEMICOLON);
	
	P.Ecart();
}


func (P *Parser) ParseInterfaceType()  {
	P.Trace("InterfaceType");
	
	P.Expect(Scanner.INTERFACE);
	P.Expect(Scanner.LBRACE);
	P.OpenScope();
	P.level--;
	for P.tok >= Scanner.IDENT {
		P.ParseMethodDecl();
	}
	P.level++;
	P.CloseScope();
	P.Expect(Scanner.RBRACE);
	
	P.Ecart();
}


func (P *Parser) ParseMapType() {
	P.Trace("MapType");
	
	P.Expect(Scanner.MAP);
	P.Expect(Scanner.LBRACK);
	P.ParseVarType();
	P.Expect(Scanner.RBRACK);
	P.ParseVarType();
	
	P.Ecart();
}


func (P *Parser) ParseStructType() {
	P.Trace("StructType");
	
	P.Expect(Scanner.STRUCT);
	P.Expect(Scanner.LBRACE);
	P.OpenScope();
	P.level--;
	for P.tok >= Scanner.IDENT {
		P.ParseVarDeclList();
		if P.tok != Scanner.RBRACE {
			P.Expect(Scanner.SEMICOLON);
		}
	}
	P.Optional(Scanner.SEMICOLON);
	P.level++;
	P.CloseScope();
	P.Expect(Scanner.RBRACE);
	
	P.Ecart();
}


func (P *Parser) ParsePointerType() {
	P.Trace("PointerType");
	
	P.Expect(Scanner.MUL);
	P.ParseType();
	
	P.Ecart();
}


// Returns false if no type was found.
func (P *Parser) TryType() bool {
	P.Trace("Type (try)");
	
	found := true;
	switch P.tok {
	case Scanner.IDENT: P.ParseTypeName();
	case Scanner.LBRACK: P.ParseArrayType();
	case Scanner.CHAN, Scanner.ARROW: P.ParseChannelType();
	case Scanner.INTERFACE: P.ParseInterfaceType();
	case Scanner.LPAREN: P.ParseSignature();
	case Scanner.MAP: P.ParseMapType();
	case Scanner.STRUCT: P.ParseStructType();
	case Scanner.MUL: P.ParsePointerType();
	default: found = false;
	}

	P.Ecart();
	return found;
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


func (P *Parser) ParseFunctionLit() AST.Expr {
	P.Trace("FunctionLit");
	
	P.Expect(Scanner.FUNC);
	P.ParseSignature();  // replace this with ParseFunctionType() and it won't work - 6g bug?
	P.ParseBlock();
	
	P.Ecart();
	var x AST.Expr;
	return x;
}


func (P *Parser) ParseExpressionPair() {
	P.Trace("ExpressionPair");

	P.ParseExpression();
	P.Expect(Scanner.COLON);
	P.ParseExpression();
	
	P.Ecart();
}


func (P *Parser) ParseExpressionPairList() {
	P.Trace("ExpressionPairList");

	P.ParseExpressionPair();
	for P.tok == Scanner.COMMA {
		P.ParseExpressionPair();
	}
	
	P.Ecart();
}


func (P *Parser) ParseCompositeLit() AST.Expr {
	P.Trace("CompositeLit");
	
	P.Expect(Scanner.HASH);
	P.ParseType();
	P.Expect(Scanner.LBRACE);
	// TODO: should allow trailing ','
	if P.tok != Scanner.RBRACE {
		P.ParseExpression();
		if P.tok == Scanner.COMMA {
			P.Next();
			if P.tok != Scanner.RBRACE {
				P.ParseExpressionList();
			}
		} else if P.tok == Scanner.COLON {
			P.Next();
			P.ParseExpression();
			if P.tok == Scanner.COMMA {
				P.Next();
				if P.tok != Scanner.RBRACE {
					P.ParseExpressionPairList();
				}
			}
		}
	}
	P.Expect(Scanner.RBRACE);

	P.Ecart();
	var x AST.Expr;
	return x;
}


func (P *Parser) ParseOperand(ident *AST.Ident) AST.Expr {
	P.Trace("Operand");

	if ident == nil && P.tok == Scanner.IDENT {
		// no look-ahead yet
		ident = P.ParseIdent();
	}

	var x AST.Expr;

	if ident != nil {
		// we have an identifier

	} else {
	
		switch P.tok {
		case Scanner.IDENT:
			panic("UNREACHABLE");
			
		case Scanner.LPAREN:
			P.Next();
			x = P.ParseExpression();
			P.Expect(Scanner.RPAREN);
			
		case Scanner.INT:
			P.Next();

		case Scanner.FLOAT:
			P.Next();

		case Scanner.STRING:
			P.Next();

		case Scanner.FUNC:
			P.ParseFunctionLit();
			
		case Scanner.HASH:
			P.ParseCompositeLit();

		default:
			P.Error(P.pos, "operand expected");
			P.Next();  // make progress
		}
	
	}

	P.Ecart();
	return x;
}


func (P *Parser) ParseSelectorOrTypeAssertion(x AST.Expr) AST.Expr {
	P.Trace("SelectorOrTypeAssertion");

	P.Expect(Scanner.PERIOD);
	pos := P.pos;
	
	if P.tok >= Scanner.IDENT {
		P.ParseIdent();
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
	
	P.Expect(Scanner.LBRACK);
	i := P.ParseExpression();
	if P.tok == Scanner.COLON {
		P.Next();
		j := P.ParseExpression();
	}
	P.Expect(Scanner.RBRACK);
		
	P.Ecart();
	return x;
}


func (P *Parser) ParseCall(x AST.Expr) AST.Expr {
	P.Trace("Call");

	P.Expect(Scanner.LPAREN);
	if P.tok != Scanner.RPAREN {
	   	// first arguments could be a type if the call is to "new"
		// - exclude type names because they could be expression starts
		// - exclude "("'s because function types are not allowed and they indicate an expression
		if P.tok != Scanner.IDENT && P.tok != Scanner.LPAREN && P.TryType() {
		   	if P.tok == Scanner.COMMA {
			   	 P.Next();
				 if P.tok != Scanner.RPAREN {
				    	  P.ParseExpressionList();
				 }
			}
		} else {
			P.ParseExpressionList();
		}
	}
	P.Expect(Scanner.RPAREN);
	
	P.Ecart();
	return x;
}


func (P *Parser) ParsePrimaryExpr(ident *AST.Ident) AST.Expr {
	P.Trace("PrimaryExpr");
	
	x := P.ParseOperand(ident);
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


func (P *Parser) ParseUnaryExpr() AST.Expr {
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
	
	x := P.ParsePrimaryExpr(nil);
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


func (P *Parser) ParseBinaryExpr(ident *AST.Ident, prec1 int) AST.Expr {
	P.Trace("BinaryExpr");
	
	var x AST.Expr;
	if ident != nil {
		x = P.ParsePrimaryExpr(ident);
	} else {
		x = P.ParseUnaryExpr();
	}

	for prec := Precedence(P.tok); prec >= prec1; prec-- {
		for Precedence(P.tok) == prec {
			P.Next();
			y := P.ParseBinaryExpr(nil, prec + 1);
		}
	}
	
	P.Ecart();
	return x;
}


// Expressions where the first token may be an identifier which has already been consumed.
func (P *Parser) ParseIdentExpression(ident *AST.Ident) AST.Expr {
	P.Trace("IdentExpression");
	indent := P.indent;
	
	x := P.ParseBinaryExpr(ident, 1);
	
	if indent != P.indent {
		panic("imbalanced tracing code (Expression)");
	}

	P.Ecart();
	return x;
}


func (P *Parser) ParseExpression() AST.Expr {
	P.Trace("Expression");
	
	x := P.ParseIdentExpression(nil);

	P.Ecart();
	return x;
}


// ----------------------------------------------------------------------------
// Statements

func (P *Parser) ParseSimpleStat() {
	P.Trace("SimpleStat");
	
	P.ParseExpressionList();
	
	switch P.tok {
	case Scanner.COLON:
		// label declaration
		P.Next();  // consume ":"
		
	case Scanner.DEFINE:
		// variable declaration
		P.Next();  // consume ":="
		P.ParseExpressionList();
		
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
		P.Next();
		P.ParseExpressionList();
		
	default:
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


func (P *Parser) ParseRangeStat() {
	P.Trace("RangeStat");
	
	P.Expect(Scanner.RANGE);
	P.ParseIdentList();
	P.Expect(Scanner.DEFINE);
	P.ParseExpression();
	P.ParseBlock();
	
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
		panic("imbalanced tracing code (Statement)");
	}
	P.Ecart();
	return res;
}


// ----------------------------------------------------------------------------
// Declarations

func (P *Parser) ParseImportSpec() {
	P.Trace("ImportSpec");
	
	if P.tok == Scanner.PERIOD {
		P.Error(P.pos, `"import ." not yet handled properly`);
		P.Next();
	} else if P.tok == Scanner.IDENT {
		P.ParseIdent();
	}
	
	if P.tok == Scanner.STRING {
		// TODO eventually the scanner should strip the quotes
		P.Next();
	} else {
		P.Expect(Scanner.STRING);  // use Expect() error handling
	}
	
	P.Ecart();
}


func (P *Parser) ParseConstSpec(exported bool) {
	P.Trace("ConstSpec");
	
	list := P.ParseIdent();
	P.TryType();
	
	if P.tok == Scanner.ASSIGN {
		P.Next();
		P.ParseExpressionList();
	}
	
	P.Ecart();
}


func (P *Parser) ParseTypeSpec(exported bool) {
	P.Trace("TypeSpec");

	ident := P.ParseIdent();
	P.ParseType();
	
	P.Ecart();
}


func (P *Parser) ParseVarSpec(exported bool) {
	P.Trace("VarSpec");
	
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
	ident := P.ParseNamedSignature();
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

func (P *Parser) ParseProgram() {
	P.Trace("Program");
	
	P.OpenScope();
	P.Expect(Scanner.PACKAGE);
	obj := P.ParseIdent();
	P.Optional(Scanner.SEMICOLON);
	
	{	P.OpenScope();
		if P.level != 0 {
			panic("incorrect scope level");
		}
		
		for P.tok == Scanner.IMPORT {
			P.ParseDecl(false, Scanner.IMPORT);
			P.Optional(Scanner.SEMICOLON);
		}
		
		for P.tok != Scanner.EOF {
			P.ParseDeclaration();
			P.Optional(Scanner.SEMICOLON);
		}
		
		if P.level != 0 {
			panic("incorrect scope level");
		}
		P.CloseScope();
	}
	
	P.CloseScope();
	P.Ecart();
}
