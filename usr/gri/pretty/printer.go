// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Printer

import Scanner "scanner"
import AST "ast"


// Printer implements AST.Visitor
type Printer struct {
	level int;  // true scope level
	indent int;  // indentation level
	semi bool;  // pending ";"
	newl bool;  // pending "\n"
	prec int;  // operator precedence
}


func (P *Printer) String(s string) {
	if P.semi && P.level > 0 {  // no semicolons at level 0
		print(";");
	}
	if P.newl {
		print("\n");
		for i := P.indent; i > 0; i-- {
			print("\t");
		}
	}
	print(s);
	P.newl, P.semi = false, false;
}


func (P *Printer) NewLine() {  // explicit "\n"
	print("\n");
	P.semi, P.newl = false, true;
}


func (P *Printer) OpenScope(paren string) {
	P.semi, P.newl = false, false;
	P.String(paren);
	P.level++;
	P.indent++;
	P.newl = true;
}


func (P *Printer) CloseScope(paren string) {
	P.level--;
	P.indent--;
	P.newl = true;
	P.String(paren);
	P.semi, P.newl = false, true;
}


func (P *Printer) Print(x AST.Node) {
	outer := P.prec;
	P.prec = 0;
	x.Visit(P);
	P.prec = outer;
}


func (P *Printer) PrintList(p *AST.List) {
	for i := 0; i < p.len(); i++ {
		if i > 0 {
			P.String(", ");
		}
		P.Print(p.at(i));
	}
}


// ----------------------------------------------------------------------------
// Basics

func (P *Printer) DoNil(x *AST.Nil) {
	P.String("<NIL>");
}


func (P *Printer) DoIdent(x *AST.Ident) {
	P.String(x.val);
}


// ----------------------------------------------------------------------------
// Types

func (P *Printer) DoFunctionType(x *AST.FunctionType) {
	P.String("(");
	P.PrintList(x.params);
	P.String(")");
	if x.result != nil {
		P.String(" (");
		P.PrintList(x.result);
		P.String(")");
	}
}


func (P *Printer) DoArrayType(x *AST.ArrayType) {
	P.String("[");
	P.Print(x.len_);
	P.String("] ");
	P.Print(x.elt);
}


func (P *Printer) DoStructType(x *AST.StructType) {
	P.String("struct ");
	P.OpenScope("{");
	for i := 0; i < x.fields.len(); i++ {
		P.Print(x.fields.at(i));
		P.newl, P.semi = true, true;
	}
	P.CloseScope("}");
}


func (P *Printer) DoMapType(x *AST.MapType) {
	P.String("[");
	P.Print(x.key);
	P.String("] ");
	P.Print(x.val);
}


func (P *Printer) DoChannelType(x *AST.ChannelType) {
	switch x.mode {
	case AST.FULL: P.String("chan ");
	case AST.RECV: P.String("<-chan ");
	case AST.SEND: P.String("chan <- ");
	}
	P.Print(x.elt);
}


func (P *Printer) DoInterfaceType(x *AST.InterfaceType) {
	P.String("interface ");
	P.OpenScope("{");
	for i := 0; i < x.methods.len(); i++ {
		P.Print(x.methods.at(i));
		P.newl, P.semi = true, true;
	}
	P.CloseScope("}");
}


func (P *Printer) DoPointerType(x *AST.PointerType) {
	P.String("*");
	P.Print(x.base);
}


// ----------------------------------------------------------------------------
// Declarations

func (P *Printer) DoBlock(x *AST.Block);


func (P *Printer) DoImportDecl(x *AST.ImportDecl) {
	if x.ident != nil {
		P.Print(x.ident);
		P.String(" ");
	}
	P.String(x.file);
}


func (P *Printer) DoConstDecl(x *AST.ConstDecl) {
	P.Print(x.ident);
	P.String(" ");
	P.Print(x.typ);
	P.String(" = ");
	P.Print(x.val);
	P.semi = true;
}


func (P *Printer) DoTypeDecl(x *AST.TypeDecl) {
	P.Print(x.ident);
	P.String(" ");
	P.Print(x.typ);
	P.semi = true;
}


func (P *Printer) DoVarDecl(x *AST.VarDecl) {
	P.PrintList(x.idents);
	P.String(" ");
	P.Print(x.typ);
	if x.vals != nil {
		P.String(" = ");
		P.PrintList(x.vals);
	}
	P.semi = true;
}


func (P *Printer) DoVarDeclList(x *AST.VarDeclList) {
	if x.idents != nil {
		P.PrintList(x.idents);	
		P.String(" ");
	}
	P.Print(x.typ);
}


func (P *Printer) DoFuncDecl(x *AST.FuncDecl) {
	P.String("func ");
	if x.typ.recv != nil {
		P.String("(");
		P.DoVarDeclList(x.typ.recv);
		P.String(") ");
	}
	P.DoIdent(x.ident);
	P.DoFunctionType(x.typ);
	if x.body != nil {
		P.DoBlock(x.body);
	} else {
		P.String(" ;");
	}
	P.NewLine();
	P.NewLine();
}


func (P *Printer) DoMethodDecl(x *AST.MethodDecl) {
	P.DoIdent(x.ident);
	P.DoFunctionType(x.typ);
}


func (P *Printer) DoDeclaration(x *AST.Declaration) {
	P.String(Scanner.TokenName(x.tok));
	P.String(" ");
	switch x.decls.len() {
	case 0:
		P.String("()");
	case 1:
		P.Print(x.decls.at(0));
	default:
		P.OpenScope(" (");
		for i := 0; i < x.decls.len(); i++ {
			P.Print(x.decls.at(i));
			P.newl, P.semi = true, true;
		}
		P.CloseScope(")");
	}
	if P.level == 0 {
		P.NewLine();
	}
	P.newl = true;
}


// ----------------------------------------------------------------------------
// Expressions

func (P *Printer) DoBinary(x *AST.Binary) {
	outer := P.prec;
	P.prec = Scanner.Precedence(x.tok);
	
	if P.prec < outer {
		print("(");
	}
	
	P.Print(x.x);
	P.String(" " + Scanner.TokenName(x.tok) + " ");
	P.Print(x.y);
	
	if P.prec < outer {
		print(")");
	}

	P.prec = outer; 
}


func (P *Printer) DoUnary(x *AST.Unary) {
	P.String(Scanner.TokenName(x.tok));
	P.Print(x.x);
}


func (P *Printer) DoLiteral(x *AST.Literal) {
	P.String(x.val);
}


func (P *Printer) DoPair(x *AST.Pair) {
	P.Print(x.x);
	P.String(" : ");
	P.Print(x.y);
}


func (P *Printer) DoIndex(x *AST.Index) {
	P.Print(x.x);
	P.String("[");
	P.Print(x.index);
	P.String("]");
}


func (P *Printer) DoCall(x *AST.Call) {
	P.Print(x.fun);
	P.String("(");
	P.PrintList(x.args);
	P.String(")");
}


func (P *Printer) DoSelector(x *AST.Selector) {
	P.Print(x.x);
	P.String(".");
	P.String(x.field);
}


func (P *Printer) DoCompositeLit(x *AST.CompositeLit) {
	P.Print(x.typ);
	P.String("{");
	P.PrintList(x.vals);
	P.String("}");
}


func (P *Printer) DoFunctionLit(x *AST.FunctionLit) {
	P.String("func ");
	P.Print(x.typ);
	P.String(" ");
	P.Print(x.body);
}


// ----------------------------------------------------------------------------
// Statements

func (P *Printer) DoBlock(x *AST.Block) {
	P.OpenScope("{");
	for i := 0; i < x.stats.len(); i++ {
		P.Print(x.stats.at(i));
		P.newl = true;
	}
	P.CloseScope("}");
}


func (P *Printer) DoLabel(x *AST.Label) {
	P.indent--;
	P.newl = true;
	P.Print(x.ident);
	P.String(":");
	P.indent++;
}


func (P *Printer) DoExprStat(x *AST.ExprStat) {
	P.Print(x.expr);
	P.semi = true;
}


func (P *Printer) DoAssignment(x *AST.Assignment) {
	P.PrintList(x.lhs);
	P.String(" " + Scanner.TokenName(x.tok) + " ");
	P.PrintList(x.rhs);
	P.semi = true;
}


func (P *Printer) PrintControlClause(x *AST.ControlClause) {
	if x.has_init {
		P.String(" ");
		P.Print(x.init);
		P.semi = true;
		P.String("");
	}
	if x.has_expr {
		P.String(" ");
		P.Print(x.expr);
		P.semi = false;
	}
	if x.has_post {
		P.semi = true;
		P.String(" ");
		P.Print(x.post);
		P.semi = false;
	}
	P.String(" ");
}


func (P *Printer) DoIfStat(x *AST.IfStat) {
	P.String("if");
	P.PrintControlClause(x.ctrl);
	P.DoBlock(x.then);
	if x.has_else {
		P.newl = false;
		P.String(" else ");
		P.Print(x.else_);
	}
}


func (P *Printer) DoForStat(x *AST.ForStat) {
	P.String("for");
	P.PrintControlClause(x.ctrl);
	P.DoBlock(x.body);
}


func (P *Printer) DoCaseClause(x *AST.CaseClause) {
	if x.exprs != nil {
		P.String("case ");
		P.PrintList(x.exprs);
		P.String(":");
	} else {
		P.String("default:");
	}
	
	P.OpenScope("");
	for i := 0; i < x.stats.len(); i++ {
		P.Print(x.stats.at(i));
		P.newl = true;
	}
	if x.falls {
		P.String("fallthrough");
	}
	P.CloseScope("");
}


func (P *Printer) DoSwitchStat(x *AST.SwitchStat) {
	P.String("switch ");
	P.PrintControlClause(x.ctrl);
	P.OpenScope("{");
	P.indent--;
	for i := 0; i < x.cases.len(); i++ {
		P.Print(x.cases.at(i));
	}
	P.indent++;
	P.CloseScope("}");
}


func (P *Printer) DoReturnStat(x *AST.ReturnStat) {
	P.String("return ");
	P.PrintList(x.res);
	P.semi = true;
}


func (P *Printer) DoIncDecStat(x *AST.IncDecStat) {
	P.Print(x.expr);
	P.String(Scanner.TokenName(x.tok));
	P.semi = true;
}


func (P *Printer) DoControlFlowStat(x *AST.ControlFlowStat) {
	P.String(Scanner.TokenName(x.tok));
	if x.label != nil {
		P.String(" ");
		P.Print(x.label);
	}
	P.semi = true;
}


func (P *Printer) DoGoStat(x *AST.GoStat) {
	P.String("go ");
	P.Print(x.expr);
	P.semi = true;
}


// ----------------------------------------------------------------------------
// Program

func (P *Printer) DoProgram(x *AST.Program) {
	P.String("package ");
	P.DoIdent(x.ident);
	P.NewLine();
	for i := 0; i < x.decls.len(); i++ {
		P.Print(x.decls.at(i));
	}
	P.newl = true;
	P.String("");
}


// ----------------------------------------------------------------------------
// Driver

export func Print(x AST.Node) {
	var P Printer;
	(&P).Print(x);
	print("\n");
}

