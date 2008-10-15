// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Printer

import Scanner "scanner"
import Node "node"


export type Printer struct {
	level int;  // true scope level
	indent int;  // indentation level
	semi bool;  // pending ";"
	newl bool;  // pending "\n"
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


func (P *Printer) Token(tok int) {
	P.String(Scanner.TokenString(tok));
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
	P.indent--;
	P.semi = false;
	P.String(paren);
	P.level--;
	P.semi, P.newl = false, true;
}


// ----------------------------------------------------------------------------
// Types

func (P *Printer) Expr(x *Node.Expr)

func (P *Printer) Type(t *Node.Type) {
	switch t.tok {
	case Scanner.IDENT:
		P.Expr(t.expr);

	case Scanner.LBRACK:
		P.String("[");
		if t.expr != nil {
			P.Expr(t.expr);
		}
		P.String("] ");
		P.Type(t.elt);

	case Scanner.STRUCT:
		P.String("struct");
		if t.list != nil {
			P.OpenScope(" {");
			/*
			for i := 0; i < x.fields.len(); i++ {
				P.Print(x.fields.at(i));
				P.newl, P.semi = true, true;
			}
			*/
			P.CloseScope("}");
		}

	case Scanner.MAP:
		P.String("[");
		P.Type(t.key);
		P.String("] ");
		P.Type(t.elt);

	case Scanner.CHAN:
		switch t.mode {
		case Node.FULL: P.String("chan ");
		case Node.RECV: P.String("<-chan ");
		case Node.SEND: P.String("chan <- ");
		}
		P.Type(t.elt);

	case Scanner.INTERFACE:
		P.String("interface");
		if t.list != nil {
			P.OpenScope(" {");
			/*
			for i := 0; i < x.methods.len(); i++ {
				P.Print(x.methods.at(i));
				P.newl, P.semi = true, true;
			}
			*/
			P.CloseScope("}");
		}

	case Scanner.MUL:
		P.String("*");
		P.Type(t.elt);

	case Scanner.LPAREN:
		P.String("(");
		//P.PrintList(x.params);
		P.String(")");
		/*
		if x.result != nil {
			P.String(" (");
			P.PrintList(x.result);
			P.String(")");
		}
		*/

	default:
		panic("UNREACHABLE");
	}
}


// ----------------------------------------------------------------------------
// Expressions

func (P *Printer) Expr1(x *Node.Expr, prec1 int) {
	if x == nil {
		return;  // empty expression list
	}

	switch x.tok {
	case Scanner.IDENT, Scanner.INT, Scanner.STRING, Scanner.FLOAT:
		P.String(x.s);

	case Scanner.COMMA:
		P.Expr1(x.x, 0);
		P.String(", ");
		P.Expr1(x.y, 0);

	case Scanner.PERIOD:
		P.Expr1(x.x, 8);
		P.String(".");
		P.Expr1(x.y, 8);
		
	case Scanner.LBRACK:
		P.Expr1(x.x, 8);
		P.String("[");
		P.Expr1(x.y, 0);
		P.String("]");

	case Scanner.LPAREN:
		P.Expr1(x.x, 8);
		P.String("(");
		P.Expr1(x.y, 0);
		P.String(")");
		
	default:
		if x.x == nil {
			// unary expression
			P.Token(x.tok);
			P.Expr1(x.y, 7);
		} else {
			// binary expression: print ()'s if necessary
			prec := Scanner.Precedence(x.tok);
			if prec < prec1 {
				print("(");
			}
			P.Expr1(x.x, prec);
			P.String(" ");
			P.Token(x.tok);
			P.String(" ");
			P.Expr1(x.y, prec);
			if prec < prec1 {
				print(")");
			}
		}
	}
}


func (P *Printer) Expr(x *Node.Expr) {
	P.Expr1(x, 0);
}


// ----------------------------------------------------------------------------
// Statements

func (P *Printer) Stat(s *Node.Stat)

func (P *Printer) StatementList(list *Node.List) {
	for i, n := 0, list.len(); i < n; i++ {
		P.Stat(list.at(i).(*Node.Stat));
		P.newl = true;
	}
}


func (P *Printer) Block(list *Node.List, indent bool) {
	P.OpenScope("{");
	if !indent {
		P.indent--;
	}
	P.StatementList(list);
	if !indent {
		P.indent++;
	}
	P.CloseScope("}");
}


func (P *Printer) ControlClause(s *Node.Stat) {
	if s.init != nil {
		P.String(" ");
		P.Stat(s.init);
		P.semi = true;
		P.String("");
	}
	if s.expr != nil {
		P.String(" ");
		P.Expr(s.expr);
		P.semi = false;
	}
	if s.tok == Scanner.FOR && s.post != nil {
		P.semi = true;
		P.String(" ");
		P.Stat(s.post);
		P.semi = false;
	}
	P.String(" ");
}


func (P *Printer) Declaration(d *Node.Decl);

func (P *Printer) Stat(s *Node.Stat) {
	if s == nil {  // TODO remove this check
		P.String("<nil stat>");
		return;
	}

	switch s.tok {
	case 0: // TODO use a real token const
		P.Expr(s.expr);
		P.semi = true;

	case Scanner.CONST, Scanner.TYPE, Scanner.VAR:
		P.Declaration(s.decl);

	case Scanner.DEFINE, Scanner.ASSIGN, Scanner.ADD_ASSIGN,
		Scanner.SUB_ASSIGN, Scanner.MUL_ASSIGN, Scanner.QUO_ASSIGN,
		Scanner.REM_ASSIGN, Scanner.AND_ASSIGN, Scanner.OR_ASSIGN,
		Scanner.XOR_ASSIGN, Scanner.SHL_ASSIGN, Scanner.SHR_ASSIGN:
		P.Expr(s.lhs);
		P.String(" ");
		P.Token(s.tok);
		P.String(" ");
		P.Expr(s.expr);
		P.semi = true;

	case Scanner.INC, Scanner.DEC:
		P.Expr(s.expr);
		P.Token(s.tok);
		P.semi = true;

	case Scanner.LBRACE:
		P.Block(s.block, true);

	case Scanner.IF:
		P.String("if");
		P.ControlClause(s);
		P.Block(s.block, true);
		if s.post != nil {
			P.newl = false;
			P.String(" else ");
			P.Stat(s.post);
		}

	case Scanner.FOR:
		P.String("for");
		P.ControlClause(s);
		P.Block(s.block, true);

	case Scanner.SWITCH, Scanner.SELECT:
		P.Token(s.tok);
		P.ControlClause(s);
		P.Block(s.block, false);

	case Scanner.CASE, Scanner.DEFAULT:
		P.Token(s.tok);
		if s.expr != nil {
			P.String(" ");
			P.Expr(s.expr);
		}
		P.String(":");
		P.OpenScope("");
		P.StatementList(s.block);
		P.CloseScope("");

	case Scanner.GO, Scanner.RETURN, Scanner.BREAK, Scanner.CONTINUE, Scanner.GOTO:
		P.Token(s.tok);
		P.String(" ");
		P.Expr(s.expr);
		P.semi = true;

	default:
		P.String("<stat>");
		P.semi = true;
	}
}


// ----------------------------------------------------------------------------
// Declarations


/*
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
		P.String(" ");
		P.DoBlock(x.body);
	} else {
		P.String(" ;");
	}
	P.NewLine();
	P.NewLine();

}


func (P *Printer) DoMethodDecl(x *AST.MethodDecl) {
	//P.DoIdent(x.ident);
	//P.DoFunctionType(x.typ);
}
*/


func (P *Printer) Declaration(d *Node.Decl) {
	if d.tok == Scanner.FUNC || d.ident == nil {
		if d.exported {
			P.String("export ");
		}
		P.Token(d.tok);
		P.String(" ");
	}

	if d.ident == nil {
		switch d.list.len() {
		case 0:
			P.String("()");
		case 1:
			P.Declaration(d.list.at(0).(*Node.Decl));
		default:
			P.OpenScope("(");
			for i := 0; i < d.list.len(); i++ {
				P.Declaration(d.list.at(i).(*Node.Decl));
				P.newl, P.semi = true, true;
			}
			P.CloseScope(")");
		}

	} else {
		P.Expr(d.ident);
		if d.typ != nil {
			P.String(" ");
			P.Type(d.typ);
		}
		if d.val != nil {
			P.String(" = ");
			P.Expr(d.val);
		}
		if d.list != nil {
			if d.tok != Scanner.FUNC {
				panic("must be a func declaration");
			}
			P.String(" ");
			P.Block(d.list, true);
		}
	}

	// extra newline at the top level
	if P.level == 0 {
		P.NewLine();
	}

	P.newl = true;
}


// ----------------------------------------------------------------------------
// Program

func (P *Printer) Program(p *Node.Program) {
	P.String("package ");
	P.Expr(p.ident);
	P.NewLine();
	for i := 0; i < p.decls.len(); i++ {
		P.Declaration(p.decls.at(i));
	}
	P.newl = true;
	P.String("");
}
