// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Printer

import Scanner "scanner"
import AST "ast"


type Printer /* implements AST.Visitor */ struct {
	indent int;
}


func (P *Printer) String(s string) {
	print(s);
}


func (P *Printer) Print(x AST.Node) {
	x.Visit(P);
}


func (P *Printer) PrintExprList(p *AST.List) {
	if p != nil {
		for i := 0; i < p.len(); i++ {
			if i > 0 {
				P.String(", ");
			}
			P.Print(p.at(i));
		}
	}
}


// ----------------------------------------------------------------------------
// Basics

func (P *Printer) DoNil(x *AST.Nil) {
	P.String("?\n");
}


func (P *Printer) DoIdent(x *AST.Ident) {
	P.String(x.val);
}


// ----------------------------------------------------------------------------
// Declarations

func (P *Printer) DoBlock(x *AST.Block);

func (P *Printer) DoFuncDecl(x *AST.FuncDecl) {
	P.String("func ");
	P.DoIdent(x.ident);
	P.String("(... something here ...) ");
	if x.body != nil {
		P.DoBlock(x.body);
	} else {
		P.String(";\n");
	}
}


// ----------------------------------------------------------------------------
// Expressions

func (P *Printer) DoBinary(x *AST.Binary) {
	print("(");
	P.Print(x.x);
	P.String(" " + Scanner.TokenName(x.tok) + " ");
	P.Print(x.y);
	print(")");
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
	P.PrintExprList(x.args);
	P.String(")");
}


func (P *Printer) DoSelector(x *AST.Selector) {
	P.Print(x.x);
	P.String(".");
	P.String(x.field);
}


// ----------------------------------------------------------------------------
// Statements

func (P *Printer) DoBlock(x *AST.Block) {
	if x == nil || x.stats == nil {
		P.String("\n");
		return;
	}

	P.String("{\n");
	P.indent++;
	for i := 0; i < x.stats.len(); i++ {
		P.Print(x.stats.at(i));
		P.String("\n");
	}
	P.indent--;
	P.String("}\n");
}


func (P *Printer) DoExprStat(x *AST.ExprStat) {
	P.Print(x.expr);
}


func (P *Printer) DoAssignment(x *AST.Assignment) {
	P.PrintExprList(x.lhs);
	P.String(" " + Scanner.TokenName(x.tok) + " ");
	P.PrintExprList(x.rhs);
}


func (P *Printer) DoIf(x *AST.If) {
	P.String("if ");
	P.Print(x.cond);
	P.DoBlock(x.then);
	if x.else_ != nil {
		P.String("else ");
		P.DoBlock(x.else_);
	}
}


func (P *Printer) DoFor(x *AST.For) {
}


func (P *Printer) DoSwitch(x *AST.Switch) {
}


func (P *Printer) DoReturn(x *AST.Return) {
	P.String("return ");
	P.PrintExprList(x.res);
}


// ----------------------------------------------------------------------------
// Program

func (P *Printer) DoProgram(x *AST.Program) {
	P.String("package ");
	P.DoIdent(x.ident);
	P.String("\n");
	for i := 0; i < x.decls.len(); i++ {
		P.Print(x.decls.at(i));
	}
}


// ----------------------------------------------------------------------------
// Driver

export func Print(x AST.Node) {
	var P Printer;
	(&P).Print(x);
	print("\n");
}

