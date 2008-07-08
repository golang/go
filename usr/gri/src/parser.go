// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Parser

import Scanner "scanner"


export Parser
type Parser struct {
	verbose, indent int;
	S *Scanner.Scanner;
	tok int;  // one token look-ahead
	beg, end int;  // token position
	ident string;  // last ident seen
}


func (P *Parser) PrintIndent() {
	for i := P.indent; i > 0; i-- {
		print "  ";
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
		print Scanner.TokenName(P.tok), "\n";
	}
}


func (P *Parser) Open(S *Scanner.Scanner, verbose int) {
	P.verbose = verbose;
	P.indent = 0;
	P.S = S;
	P.Next();
}


func (P *Parser) Error(msg string) {
	panic "error: ", msg, "\n";
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
	if P.verbose > 0 {
		P.PrintIndent();
		print "Ident = \"", P.ident, "\"\n";
	}
	P.Expect(Scanner.IDENT);
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


func (P *Parser) ParseQualifiedIdent() {
	P.Trace("QualifiedIdent");
	P.ParseIdent();
	if P.tok == Scanner.PERIOD {
		P.Next();
		P.ParseIdent();
	}
	P.Ecart();
}


func (P *Parser) ParseTypeName() {
	P.Trace("TypeName");
	P.ParseQualifiedIdent();
	P.Ecart();
}


func (P *Parser) ParseArrayType() {
	P.Trace("ArrayType");
	P.Expect(Scanner.LBRACK);
	if P.tok != Scanner.RBRACK {
		P.ParseExpression();
	}
	P.Expect(Scanner.RBRACK);
	P.ParseType();
	P.Ecart();
}


func (P *Parser) ParseChannelType() {
	P.Trace("ChannelType");
	panic "ChannelType";
	P.Ecart();
}


func (P *Parser) ParseInterfaceType() {
	P.Trace("InterfaceType");
	panic "InterfaceType";
	P.Ecart();
}


func (P *Parser) ParseFunctionType() {
	P.Trace("FunctionType");
	panic "FunctionType";
	P.Ecart();
}


func (P *Parser) ParseMapType() {
	P.Trace("MapType");
	P.Expect(Scanner.MAP);
	P.Expect(Scanner.LBRACK);
	P.ParseType();
	P.Expect(Scanner.RBRACK);
	P.ParseType();
	P.Ecart();
}


func (P *Parser) ParseFieldDecl() {
	P.Trace("FieldDecl");
	P.ParseIdentList();
	P.ParseType();
	P.Ecart();
}


func (P *Parser) ParseStructType() {
	P.Trace("StructType");
	P.Expect(Scanner.STRUCT);
	P.Expect(Scanner.LBRACE);
	for P.tok != Scanner.RBRACE {
		P.ParseFieldDecl();
		if P.tok != Scanner.RBRACE {
			P.Expect(Scanner.SEMICOLON);
		}
	}
	if P.tok == Scanner.SEMICOLON {
		P.Next();
	}
	P.Expect(Scanner.RBRACE);
	P.Ecart();
}


func (P *Parser) ParsePointerType() {
	P.Trace("PointerType");
	P.Expect(Scanner.MUL);
	P.ParseType();
	P.Ecart();
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
	P.Ecart();
}


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
	P.Ecart();
}


func (P *Parser) ParseExpressionList() {
	P.Trace("ExpressionList");
	P.ParseExpression();
	for P.tok == Scanner.COMMA {
		P.Next();
		P.ParseExpression();
	}
	P.Ecart();
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
	P.Ecart();
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
	P.Ecart();
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
	P.Ecart();
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
	P.Ecart();
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
	P.Ecart();
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
	P.Ecart();
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


func (P *Parser) ParseResult() {
	P.Trace("Result");
	if P.tok == Scanner.LPAREN {
		// TODO: here we allow empty returns - should proably fix this
		P.ParseParameters();
	} else {
		P.ParseType();
	}
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
	P.Ecart();
}


func (P *Parser) ParseDeclaration();
func (P *Parser) ParseStatement() bool;
func (P *Parser) ParseStatementList();
func (P *Parser) ParseBlock();
func (P *Parser) ParsePrimaryExpr();


func (P *Parser) ParsePrimaryExprList() {
	P.Trace("PrimaryExprList");
	P.ParsePrimaryExpr();
	for P.tok == Scanner.COMMA {
		P.Next();
		P.ParsePrimaryExpr();
	}
	P.Ecart();
}


func (P *Parser) ParseBuiltinStat() {
	P.Trace("BuiltinStat");
	P.Expect(Scanner.IDENT);
	P.ParseExpressionList();  // TODO should be optional
	P.Ecart();
}


func (P *Parser) ParseSimpleStat() {
	P.Trace("SimpleStat");
	P.ParseExpression();
	switch P.tok {
	case Scanner.ASSIGN: fallthrough;
	case Scanner.DEFINE:
		P.Next();
		P.ParseExpression();
	case Scanner.COMMA:
		P.Next();
		P.ParsePrimaryExprList();
		switch P.tok {
		case Scanner.ASSIGN:
		case Scanner.ADD_ASSIGN:
		case Scanner.SUB_ASSIGN:
		case Scanner.MUL_ASSIGN:
		case Scanner.QUO_ASSIGN:
		case Scanner.REM_ASSIGN:
		case Scanner.AND_ASSIGN:
		case Scanner.OR_ASSIGN:
		case Scanner.XOR_ASSIGN:
		case Scanner.SHL_ASSIGN:
		case Scanner.SHR_ASSIGN:
			break;
		default:
			P.Error("expected assignment operand");
		}
		P.Next();
		P.ParseExpressionList();
	case Scanner.INC:
		P.Next();
	case Scanner.DEC:
		P.Next();
	}
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


func (P *Parser) ParseBreakStat() {
	P.Trace("BreakStat");
	P.Expect(Scanner.BREAK);
	if P.tok == Scanner.IDENT {
		P.ParseIdent();
	}
	P.Ecart();
}


func (P *Parser) ParseContinueStat() {
	P.Trace("ContinueStat");
	P.Expect(Scanner.CONTINUE);
	if P.tok == Scanner.IDENT {
		P.ParseIdent();
	}
	P.Ecart();
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
			if !P.ParseStatement() {
				P.Error("statement expected");
			}
		}
	}
	P.Ecart();
}


func (P *Parser) ParseForStat() {
	P.Trace("ForStat");
	P.Expect(Scanner.FOR);
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
		if P.tok == Scanner.SEMICOLON {
			P.Next();
		}
	}
	if P.tok == Scanner.FALLTHROUGH {
		P.Next();
		if P.tok == Scanner.SEMICOLON {
			P.Next();
		}
	}
	P.Ecart();
}


func (P *Parser) ParseSwitchStat() {
	P.Trace("SwitchStat");
	P.Expect(Scanner.SWITCH);
	if P.tok != Scanner.LBRACE {
		P.ParseSimpleStat();
		if P.tok == Scanner.SEMICOLON {
			P.ParseExpression();
		}
	}
	P.Expect(Scanner.LBRACE);
	for P.tok != Scanner.RBRACE {
		P.ParseCaseClause();
	}
	P.Expect(Scanner.RBRACE);
	P.Ecart();
}


func (P *Parser) ParseStatement() bool {
	P.Trace("Statement");
	switch P.tok {
	case Scanner.CONST: fallthrough;
	case Scanner.TYPE: fallthrough;
	case Scanner.VAR: fallthrough;
	case Scanner.FUNC:
		P.ParseDeclaration();
	case Scanner.IDENT:
		switch P.ident {
		case "print", "panic":
			P.ParseBuiltinStat();
		default:
			P.ParseSimpleStat();
		}
	case Scanner.GO:
		panic "go statement";
	case Scanner.RETURN:
		P.ParseReturnStat();
	case Scanner.BREAK:
		P.ParseBreakStat();
	case Scanner.CONTINUE:
		P.ParseContinueStat();
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
		// no statement found
		P.Ecart();
		return false;
	}
	P.Ecart();
	return true;
}


func (P *Parser) ParseStatementList() {
	P.Trace("StatementList");
	for P.ParseStatement() {
		if P.tok == Scanner.SEMICOLON {
			P.Next();
		}
	}
	P.Ecart();
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
	P.ParseIdent();
	for P.tok == Scanner.COMMA {
		P.Next();
		P.ParseIdent();
	}
	P.Ecart();
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


func (P *Parser) ParseOperand() {
	P.Trace("Operand");
	switch P.tok {
	case Scanner.IDENT:
		P.ParseQualifiedIdent();
	case Scanner.STRING:
		fallthrough;
	case Scanner.NUMBER:
		P.Next();
	case Scanner.LPAREN:
		P.Next();
		P.ParseExpression();
		P.Expect(Scanner.LPAREN);
	case Scanner.IOTA: fallthrough;
	case Scanner.TRUE: fallthrough;
	case Scanner.FALSE:
		P.Next();
	case Scanner.NEW:
		P.ParseNew();
	default:
		panic "unknown operand"
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
		P.Next();
		P.ParseUnaryExpr();
		P.Ecart();
		return;
	}
	P.ParsePrimaryExpr();
	P.Ecart();
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
			P.Ecart();
			return;
		}
	}
	P.Ecart();
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
			P.Ecart();
			return;
		}
	}
	P.Ecart();
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
		P.Next();
		P.ParseAdditiveExpr();
	}
	P.Ecart();
}


func (P *Parser) ParseLANDExpr() {
	P.Trace("LANDExpr");
	P.ParseRelationalExpr();
	for P.tok == Scanner.CAND {
		P.Next();
		P.ParseRelationalExpr();
	}
	P.Ecart();
}


func (P *Parser) ParseLORExpr() {
	P.Trace("LORExpr");
	P.ParseLANDExpr();
	for P.tok == Scanner.COR {
		P.Next();
		P.ParseLANDExpr();
	}
	P.Ecart();
}


func (P *Parser) ParseExpression() {
	P.Trace("Expression");
	P.ParseLORExpr();
	P.Ecart();
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
	P.Ecart();
}
