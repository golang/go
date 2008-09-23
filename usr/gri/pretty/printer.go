// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Printer

import Scanner "scanner"
import AST "ast"


type Printer struct {
	
}


func (P *Printer) Print(s string) {
	print(s);
}


func (P *Printer) PrintExpr(x AST.Expr) {
/*
	if x == nil {
		P.Print("<nil>");
		return;
	}
	
	switch x.tok {
	case Scanner.IDENT:
		P.Print(x.val);
	
	case Scanner.INT, Scanner.FLOAT, Scanner.STRING:
		P.Print(x.val);
		
	case Scanner.PERIOD:
		P.PrintExpr(x.x);
		P.Print(Scanner.TokenName(x.tok));
		P.PrintExpr(x.y);

	case Scanner.LBRACK:
		P.PrintExpr(x.x);
		P.Print("[");
		P.PrintExpr(x.y);
		P.Print("]");

	default:
		// unary or binary expression
		print("(");
		if x.x != nil {
			P.PrintExpr(x.x);
		}
		P.Print(" " + Scanner.TokenName(x.tok) + " ");
		P.PrintExpr(x.y);
		print(")");
	}
*/
}


export func Print(x AST.Expr) {
	var P Printer;
	print("expr = ");
	(&P).PrintExpr(x);
	print("\n");
}
