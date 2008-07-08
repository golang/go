// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Parser

import Scanner "scanner"


export Parser
type Parser struct {
	verbose bool;
	S *Scanner.Scanner;
	tok int;  // one token look-ahead
	beg, end int;  // token position
};


func (P *Parser) Next() {
	P.tok, P.beg, P.end = P.S.Scan()
}


func (P *Parser) Open(S *Scanner.Scanner, verbose bool) {
	P.verbose = verbose;
	P.S = S;
	P.Next();
}


func (P *Parser) Error(msg string) {
	print "error: ", msg, "\n";
}


func (P *Parser) Trace(msg string) {
	if P.verbose {
		print msg, "\n";
	}
}


func (P *Parser) Expect(tok int) {
	if tok != P.tok {
		P.Error("expected `" + Scanner.TokenName(tok) + "`, found `" + Scanner.TokenName(P.tok) + "`");
	}
	P.Next();  // make progress in any case
}


func (P *Parser) ParseType();
func (P *Parser) ParseExpression();


func (P *Parser) ParseIdent() {
	P.Trace("Ident");
	P.Expect(Scanner.IDENT);
}


func (P *Parser) ParseIdentList() {
	P.Trace("IdentList");
	P.ParseIdent();
	for P.tok == Scanner.COMMA {
		P.Next();
		P.ParseIdent();
	}
}


func (P *Parser) ParseQualifiedIdent() {
	P.Trace("QualifiedIdent");
	P.ParseIdent();
	if P.tok == Scanner.PERIOD {
		P.Next();
		P.ParseIdent();
	}
}


func (P *Parser) ParseTypeName() {
	P.Trace("TypeName");
	P.ParseQualifiedIdent();
}


func (P *Parser) ParseArrayType() {
	P.Trace("ArrayType");
	P.Expect(Scanner.LBRACK);
	if P.tok != Scanner.RBRACK {
		P.ParseExpression();
	}
	P.Expect(Scanner.RBRACK);
	P.ParseType();
}


func (P *Parser) ParseChannelType() {
	P.Trace("ChannelType");
	panic "ChannelType"
}


func (P *Parser) ParseInterfaceType() {
	P.Trace("InterfaceType");
	panic "InterfaceType"
}


func (P *Parser) ParseFunctionType() {
	P.Trace("FunctionType");
	panic "FunctionType"
}


func (P *Parser) ParseMapType() {
	P.Trace("MapType");
	P.Expect(Scanner.MAP);
	P.Expect(Scanner.LBRACK);
	P.ParseType();
	P.Expect(Scanner.RBRACK);
	P.ParseType();
}


func (P *Parser) ParseFieldDecl() {
	P.Trace("FieldDecl");
	P.ParseIdentList();
	P.ParseType();
}


func (P *Parser) ParseStructType() {
	P.Trace("StructType");
	P.Expect(Scanner.STRUCT);
	P.Expect(Scanner.LBRACE);
	if P.tok != Scanner.RBRACE {
		P.ParseFieldDecl();
		for P.tok == Scanner.SEMICOLON {
			P.Next();
			P.ParseFieldDecl();
		}
		if P.tok == Scanner.SEMICOLON {
			P.Next();
		}
	}
	P.Expect(Scanner.RBRACE);
}


func (P *Parser) ParsePointerType() {
	P.Trace("PointerType");
	P.Expect(Scanner.MUL);
	P.ParseType();
}


func (P *Parser) ParseType() {
	P.Trace("Type");
	switch P.tok {
	case Scanner.IDENT:
		P.ParseTypeName();
	case Scanner.LBRACK:
		P.ParseArrayType();
	case Scanner.CHAN:
		P.ParseChannelType();
	case Scanner.INTERFACE:
		P.ParseInterfaceType();
	case Scanner.FUNC:
		P.ParseFunctionType();
	case Scanner.MAP:
		P.ParseMapType();
	case Scanner.STRUCT:
		P.ParseStructType();
	case Scanner.MUL:
		P.ParsePointerType();
	default:
		P.Error("type expected");
	}
}


func (P *Parser) ParseImportSpec() {
	P.Trace("ImportSpec");
	if P.tok == Scanner.PERIOD {
		P.Next();
	} else if P.tok == Scanner.IDENT {
		P.Next();
	}
	P.Expect(Scanner.STRING);
}


func (P *Parser) ParseImportDecl() {
	P.Trace("ImportDecl");
	P.Expect(Scanner.IMPORT);
	if P.tok == Scanner.LPAREN {
		P.ParseImportSpec();
		for P.tok == Scanner.SEMICOLON {
			P.Next();
			P.ParseImportSpec();
		}
		if P.tok == Scanner.SEMICOLON {
			P.Next();
		}
	} else {
		P.ParseImportSpec();
	}
}


func (P *Parser) ParseExpressionList() {
  P.Trace("ExpressionList");
	P.ParseExpression();
	for P.tok == Scanner.COMMA {
		P.Next();
		P.ParseExpression();
	}
}


func (P *Parser) ParseConstSpec() {
	P.Trace("ConstSpec");
	P.ParseIdent();
	// TODO factor this code
	switch P.tok {
	case Scanner.IDENT, Scanner.LBRACK, Scanner.CHAN, Scanner.INTERFACE,
		Scanner.FUNC, Scanner.MAP, Scanner.STRUCT, Scanner.MUL:
		P.ParseType();
	default:
		break;
	}
	if P.tok == Scanner.ASSIGN {
		P.Next();
		P.ParseExpression();
	}
}


func (P *Parser) ParseConstDecl() {
	P.Trace("ConstDecl");
	P.Expect(Scanner.CONST);
	if P.tok == Scanner.LPAREN {
		P.ParseConstSpec();
		for P.tok == Scanner.SEMICOLON {
			P.Next();
			P.ParseConstSpec();
		}
		if P.tok == Scanner.SEMICOLON {
			P.Next();
		}
	} else {
		P.ParseConstSpec();
	}
}


func (P *Parser) ParseTypeSpec() {
	P.Trace("TypeSpec");
	P.ParseIdent();
	// TODO factor this code
	switch P.tok {
	case Scanner.IDENT, Scanner.LBRACK, Scanner.CHAN, Scanner.INTERFACE,
		Scanner.FUNC, Scanner.MAP, Scanner.STRUCT, Scanner.MUL:
		P.ParseType();
	default:
		break;
	}
}


func (P *Parser) ParseTypeDecl() {
	P.Trace("TypeDecl");
	P.Expect(Scanner.TYPE);
	if P.tok == Scanner.LPAREN {
		P.ParseTypeSpec();
		for P.tok == Scanner.SEMICOLON {
			P.Next();
			P.ParseTypeSpec();
		}
		if P.tok == Scanner.SEMICOLON {
			P.Next();
		}
	} else {
		P.ParseTypeSpec();
	}
}


func (P *Parser) ParseVarSpec() {
	P.Trace("VarSpec");
	P.ParseIdentList();
	if P.tok == Scanner.ASSIGN {
		P.Next();
		P.ParseExpressionList();
	} else {
		P.ParseType();
		if P.tok == Scanner.ASSIGN {
			P.Next();
			P.ParseExpressionList();
		}
	}
}


func (P *Parser) ParseVarDecl() {
	P.Trace("VarDecl");
	P.Expect(Scanner.VAR);
	if P.tok == Scanner.LPAREN {
		P.ParseVarSpec();
		for P.tok == Scanner.SEMICOLON {
			P.Next();
			P.ParseVarSpec();
		}
		if P.tok == Scanner.SEMICOLON {
			P.Next();
		}
	} else {
		P.ParseVarSpec();
	}
}


func (P *Parser) ParseParameterSection() {
	P.Trace("ParameterSection");
	P.ParseIdentList();
	P.ParseType();
}


func (P *Parser) ParseParameterList() {
	P.Trace("ParameterList");
	P.ParseParameterSection();
	for P.tok == Scanner.COMMA {
		P.Next();
		P.ParseParameterSection();
	}
}


func (P *Parser) ParseParameters() {
	P.Trace("Parameters");
	P.Expect(Scanner.LPAREN);
	if P.tok != Scanner.RPAREN {
		P.ParseParameterList();
	}
	P.Expect(Scanner.RPAREN);
}


func (P *Parser) ParseResult() {
	P.Trace("Result");
	if P.tok == Scanner.LPAREN {
		// TODO: here we allow empty returns - should proably fix this
		P.ParseParameters();
	} else {
		P.ParseType();
	}
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
	if P.tok == Scanner.LPAREN {
		P.ParseParameters();
	}

	P.ParseIdent();  // function name

	P.ParseParameters();

	// TODO factor this code
	switch P.tok {
	case Scanner.IDENT, Scanner.LBRACK, Scanner.CHAN, Scanner.INTERFACE,
		Scanner.FUNC, Scanner.MAP, Scanner.STRUCT, Scanner.MUL, Scanner.LPAREN:
		P.ParseResult();
	default:
		break;
	}
}


func (P *Parser) ParseDeclaration();
func (P *Parser) ParseStatement();
func (P *Parser) ParseBlock();


func (P *Parser) ParsePrimaryExprList() {
	P.Trace("PrimaryExprList");
	panic "PrimaryExprList"
}


func (P *Parser) ParseSimpleStat() {
	P.Trace("SimpleStat");
	P.ParseExpression();
	switch P.tok {
	case Scanner.ASSIGN:
		P.Next();
		P.ParseExpression();
	case Scanner.COMMA:
		P.Next();
		P.ParsePrimaryExprList();
	case Scanner.INC:
		P.Next();
	case Scanner.DEC:
		P.Next();
	}
}


func (P *Parser) ParseIfStat() {
	P.Trace("IfStat");
	P.Expect(Scanner.IF);
	if P.tok != Scanner.LBRACE {
		P.ParseSimpleStat();
		if P.tok == Scanner.SEMICOLON {
			P.ParseExpression();
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
}


func (P *Parser) ParseForStat() {
	P.Trace("ForStat");
	panic "for stat";
}


func (P *Parser) ParseSwitchStat() {
	P.Trace("SwitchStat");
	panic "switch stat";
}


func (P *Parser) ParseStatement() {
	P.Trace("Statement");
	switch P.tok {
	case Scanner.CONST: fallthrough;
	case Scanner.TYPE: fallthrough;
	case Scanner.VAR: fallthrough;
	case Scanner.FUNC:
		P.ParseDeclaration();
	case Scanner.IDENT:
		P.ParseSimpleStat();
	case Scanner.GO:
		panic "go statement";
	case Scanner.RETURN:
		panic "return statement";
	case Scanner.BREAK:
		panic "break statement";
	case Scanner.CONTINUE:
		panic "continue statement";
	case Scanner.GOTO:
		panic "goto statement";
	case Scanner.LBRACE:
		P.ParseBlock();
	case Scanner.IF:
		P.ParseIfStat();
	case Scanner.FOR:
		P.ParseForStat();
	case Scanner.SWITCH:
		P.ParseSwitchStat();
	case Scanner.RANGE:
		panic "range statement";
	case Scanner.SELECT:
		panic "select statement";
	default:
		P.Error("statement expected");
	}
}


func (P *Parser) ParseStatementList() {
	P.Trace("StatementList");
	P.ParseStatement();
	for P.tok == Scanner.SEMICOLON {
		P.Next();
		P.ParseStatement();
	}
}


func (P *Parser) ParseBlock() {
	P.Trace("Block");
	P.Expect(Scanner.LBRACE);
	if P.tok != Scanner.RBRACE && P.tok != Scanner.SEMICOLON {
		P.ParseStatementList();
	}
	if P.tok == Scanner.SEMICOLON {
		P.Next();
	}
	P.Expect(Scanner.RBRACE);
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
}


func (P *Parser) ParseExportDecl() {
	P.Trace("ExportDecl");
	P.Next();
}


func (P *Parser) ParseDeclaration() {
	P.Trace("Declaration");
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
		P.Error("declaration expected");
		P.Next();  // make progress
	}
}


func (P *Parser) ParseOperand() {
	P.Trace("Operand");
	P.Next();
}


func (P *Parser) ParseSelectorOrTypeAssertion() {
	P.Trace("SelectorOrTypeAssertion");
}


func (P *Parser) ParseIndexOrSlice() {
	P.Trace("IndexOrSlice");
}


func (P *Parser) ParseInvocation() {
	P.Trace("Invocation");
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
			return;
		}
	}
}


func (P *Parser) ParseUnaryExpr() {
	P.Trace("UnaryExpr");
	switch P.tok {
	case Scanner.ADD: fallthrough;
	case Scanner.SUB: fallthrough;
	case Scanner.NOT: fallthrough;
	case Scanner.XOR: fallthrough;
	case Scanner.LSS: fallthrough;
	case Scanner.GTR: fallthrough;
	case Scanner.MUL: fallthrough;
	case Scanner.AND:
		P.ParseUnaryExpr();
		return;
	}
	P.ParsePrimaryExpr();
}


func (P *Parser) ParseMultiplicativeExpr() {
	P.Trace("MultiplicativeExpr");
	P.ParseUnaryExpr();
	for {
		switch P.tok {
		case Scanner.MUL: fallthrough;
		case Scanner.QUO: fallthrough;
		case Scanner.REM: fallthrough;
		case Scanner.SHL: fallthrough;
		case Scanner.SHR: fallthrough;
		case Scanner.AND:
			P.ParseUnaryExpr();
		default:
			return;
		}
	}
}


func (P *Parser) ParseAdditiveExpr() {
	P.Trace("AdditiveExpr");
	P.ParseMultiplicativeExpr();
	for {
		switch P.tok {
		case Scanner.ADD: fallthrough;
		case Scanner.SUB: fallthrough;
		case Scanner.OR: fallthrough;
		case Scanner.XOR:
			P.ParseMultiplicativeExpr();
		default:
			return;
		}
	}
}


func (P *Parser) ParseRelationalExpr() {
	P.Trace("RelationalExpr");
	P.ParseAdditiveExpr();
	switch P.tok {
	case Scanner.EQL: fallthrough;
	case Scanner.NEQ: fallthrough;
	case Scanner.LSS: fallthrough;
	case Scanner.LEQ: fallthrough;
	case Scanner.GTR: fallthrough;
	case Scanner.GEQ:
		P.ParseAdditiveExpr();
	}
}


func (P *Parser) ParseLANDExpr() {
	P.Trace("LANDExpr");
	P.ParseRelationalExpr();
	for P.tok == Scanner.CAND {
		P.Next();
		P.ParseRelationalExpr();
	}
}


func (P *Parser) ParseLORExpr() {
	P.Trace("LORExpr");
	P.ParseLANDExpr();
	for P.tok == Scanner.COR {
		P.Next();
		P.ParseLANDExpr();
	}
}


func (P *Parser) ParseExpression() {
	P.Trace("Expression");
	P.Next();
}


func (P *Parser) ParseProgram() {
	P.Trace("Program");
	P.Expect(Scanner.PACKAGE);
	P.ParseIdent();
	for P.tok == Scanner.IMPORT {
		P.ParseImportDecl();
		if P.tok == Scanner.SEMICOLON {
			P.Next();
		}
	}
	for P.tok != Scanner.EOF {
		P.ParseDeclaration();
		if P.tok == Scanner.SEMICOLON {
			P.Next();
		}
	}
}
