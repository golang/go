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

func (P *Printer) Val(tok int, val *Node.Val) {
	P.String(val.s);  // for now
}


func (P *Printer) Expr(x *Node.Expr) {
	if x == nil {
		return;  // empty expression list
	}

	switch x.tok {
	case Scanner.IDENT:
		P.String(x.ident);

	case Scanner.INT, Scanner.STRING, Scanner.FLOAT:
		P.Val(x.tok, x.val);

	case Scanner.LPAREN:
		// calls
		P.Expr(x.x);
		P.String("(");
		P.Expr(x.y);
		P.String(")");
		
	case Scanner.LBRACK:
		P.Expr(x.x);
		P.String("[");
		P.Expr(x.y);
		P.String("]");
		
	default:
		if x.x == nil {
			// unary expression
			P.String(Scanner.TokenName(x.tok));
			P.Expr(x.y);
		} else {
			// binary expression: print ()'s if necessary
			// TODO: pass precedence as parameter instead
			outer := P.prec;
			P.prec = Scanner.Precedence(x.tok);
			if P.prec < outer {
				print("(");
			}
			P.Expr(x.x);
			if x.tok != Scanner.PERIOD && x.tok != Scanner.COMMA {
				P.String(" ");
			}
			P.String(Scanner.TokenName(x.tok));
			if x.tok != Scanner.PERIOD {
				P.String(" ");
			}
			P.Expr(x.y);
			if P.prec < outer {
				print(")");
			}
			P.prec = outer; 
		}
	}
}


// ----------------------------------------------------------------------------
// Statements

/*
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
*/


func (P *Printer) Stat(s *Node.Stat)

func (P *Printer) StatementList(list *Node.List) {
	for i, n := 0, list.len(); i < n; i++ {
		P.Stat(list.at(i).(*Node.Stat));
		P.newl = true;
	}
}


func (P *Printer) Block(list *Node.List) {
	P.OpenScope("{");
	P.StatementList(list);
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
	if s.post != nil {
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
		P.String(Scanner.TokenName(s.tok));
		P.String(" ");
		P.Expr(s.expr);
		P.semi = true;

	case Scanner.INC, Scanner.DEC:
		P.Expr(s.expr);
		P.String(Scanner.TokenName(s.tok));
		P.semi = true;

	case Scanner.IF, Scanner.FOR, Scanner.SWITCH, Scanner.SELECT:
		P.String(Scanner.TokenName(s.tok));
		P.ControlClause(s);
		P.Block(s.block);
		
	case Scanner.CASE, Scanner.DEFAULT:
		P.String(Scanner.TokenName(s.tok));
		if s.expr != nil {
			P.String(" ");
			P.Expr(s.expr);
		}
		P.String(":");
		P.OpenScope("");
		P.StatementList(s.block);
		P.CloseScope("");
		
	case Scanner.GO, Scanner.RETURN, Scanner.BREAK, Scanner.CONTINUE, Scanner.GOTO:
		P.String("go ");
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
func (P *Printer) DoImportDecl(x *AST.ImportDecl) {
	if x.ident != nil {
		P.Print(x.ident);
		P.String(" ");
	}
	P.String(x.file);
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
		P.String(Scanner.TokenName(d.tok));
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
			P.Block(d.list);
		}
	}

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
