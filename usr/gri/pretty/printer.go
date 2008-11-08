// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Printer

import Strings "strings"
import Scanner "scanner"
import AST "ast"
import Flag "flag"
import Fmt "fmt"

var tabwith = Flag.Int("tabwidth", 4, nil, "tab width");


// ----------------------------------------------------------------------------
// Support

func assert(p bool) {
	if !p {
		panic("assert failed");
	}
}


func PrintBlanks(n int) {
	// TODO make this faster
	for ; n > 0; n-- {
		print(" ");
	}
}


// ----------------------------------------------------------------------------
// Implemententation of flexible tab stops.
// (http://nickgravgaard.com/elastictabstops/index.html)

type Buffer struct {
	lines AST.List;  // a list of lines; and each line is a list of strings
	widths AST.List;
}


func (b *Buffer) Newline() {
	b.lines.Add(AST.NewList());
}


func (b *Buffer) Init() {
	b.lines.Init();
	b.widths.Init();
	b.Newline();
}


func (b *Buffer) ComputeWidths() {
	// iterate through all columns j
	for j := 0; ; j++ {
		width := -1;  // initial column width
		
		// iterate through all lines i
		for i := 0; i < b.lines.len(); i++ {
			line := b.lines.at(i).(*AST.List);
			if j < line.len() {
				// the j.th column exists in this line
				w := len(line.at(j).(string));
				if w > width {
					width = w;
				}
			}
		}
	
		if width >= 0 {
			assert(b.widths.len() == j);
			b.widths.Add(width);
		} else {
			// no column j - we are done
			return;
		}
	}
}


func (b *Buffer) Flush() {
	b.ComputeWidths();
	
	// print the lines
	for i := 0; i < b.lines.len(); i++ {
		line := b.lines.at(i).(*AST.List);
		for j := 0; j < line.len(); j++ {
			s := line.at(j).(string);
			d := b.widths.at(j).(int) - len(s);
			assert(d >= 0);
			if d < int(tabwith.IVal()) {
				d = int(tabwith.IVal());
			}
			PrintBlanks(d);  // +1 padding
			print(s);
		}
		println();
	}
	
	b.lines.Clear();
	b.widths.Clear();
	b.Newline();
}


func (b *Buffer) Indent(n int) {
	line := b.lines.at(b.lines.len() - 1).(*AST.List);
	for ; n > 0; n-- {
		line.Add("");
	}
}


func (b *Buffer) Print(s string) {
	i := b.lines.len() - 1;
	line := b.lines.at(i).(*AST.List);
	j := line.len() - 1;
	if j < 0 {
		line.Add(s);
	} else {
		line.set(j, line.at(j).(string) + s);
	}
}


export type Printer struct {
	buf Buffer;
	
	// formatting control
	level int;  // true scope level
	indent int;  // indentation level
	semi bool;  // pending ";"
	newl int;  // pending "\n"'s

	// comments
	clist *AST.List;
	cindex int;
	cpos int;
}


const NEW_CODE = false;

func (P *Printer) String(pos int, s string) {
	if P.semi && P.level > 0 {  // no semicolons at level 0
		if NEW_CODE {
			P.buf.Print(";");
		} else {
			print(";");
		}
	}

	/*
	for pos > P.cpos {
		// we have a comment
		c := P.clist.at(P.cindex).(*AST.Comment);
		if len(c.text) > 1 && c.text[1] == '/' {
			print("  " + c.text);
			if P.newl <= 0 {
				P.newl = 1;  // line comments must have a newline
			}
		} else {
			print(c.text);
		}
		P.cindex++;
		if P.cindex < P.clist.len() {
			P.cpos = P.clist.at(P.cindex).(*AST.Comment).pos;
		} else {
			P.cpos = 1000000000;  // infinite
		}
	}
	*/

	if P.newl > 0 {
		if NEW_CODE {
			P.buf.Flush();
		}
		for i := P.newl; i > 0; i-- {
			if NEW_CODE {
				P.buf.Newline();
			} else {
				print("\n");
			}
		}
		if NEW_CODE {
			P.buf.Indent(P.indent);
		} else {
			for i := P.indent; i > 0; i-- {
				print("\t");
			}
		}
	}

	if NEW_CODE {
		P.buf.Print(s);
	} else {
		print(s);
	}

	P.semi, P.newl = false, 0;
}


func (P *Printer) Blank() {
	P.String(0, " ");
}


func (P *Printer) Token(pos int, tok int) {
	P.String(pos, Scanner.TokenString(tok));
}


func (P *Printer) OpenScope(paren string) {
	//P.semi, P.newl = false, 0;
	P.String(0, paren);
	P.level++;
	P.indent++;
	P.newl = 1;
}


func (P *Printer) CloseScope(paren string) {
	P.indent--;
	P.semi = false;
	P.String(0, paren);
	P.level--;
	P.semi, P.newl = false, 1;
}

func (P *Printer) Error(pos int, tok int, msg string) {
	P.String(0, "<");
	P.Token(pos, tok);
	P.String(0, " " + msg + ">");
}


// ----------------------------------------------------------------------------
// Types

func (P *Printer) Type(t *AST.Type)
func (P *Printer) Expr(x *AST.Expr)

func (P *Printer) Parameters(pos int, list *AST.List) {
	P.String(pos, "(");
	var prev int;
	for i, n := 0, list.len(); i < n; i++ {
		x := list.at(i).(*AST.Expr);
		if i > 0 {
			if prev == x.tok || prev == Scanner.TYPE {
				P.String(0, ", ");
			} else {
				P.Blank();
			}
		}
		P.Expr(x);
		prev = x.tok;
	}
	P.String(0, ")");
}


func (P *Printer) Fields(list *AST.List) {
	P.OpenScope("{");
	var prev int;
	for i, n := 0, list.len(); i < n; i++ {
		x := list.at(i).(*AST.Expr);
		if i > 0 {
			if prev == Scanner.TYPE && x.tok != Scanner.STRING || prev == Scanner.STRING {
				P.semi, P.newl = true, 1;
			} else if prev == x.tok {
				P.String(0, ", ");
			} else {
				P.Blank();
			}
		}
		P.Expr(x);
		prev = x.tok;
	}
	P.newl = 1;
	P.CloseScope("}");
}


func (P *Printer) Type(t *AST.Type) {
	switch t.tok {
	case Scanner.IDENT:
		P.Expr(t.expr);

	case Scanner.LBRACK:
		P.String(t.pos, "[");
		if t.expr != nil {
			P.Expr(t.expr);
		}
		P.String(0, "]");
		P.Type(t.elt);

	case Scanner.STRUCT, Scanner.INTERFACE:
		P.Token(t.pos, t.tok);
		if t.list != nil {
			P.Blank();
			P.Fields(t.list);
		}

	case Scanner.MAP:
		P.String(t.pos, "map [");
		P.Type(t.key);
		P.String(0, "]");
		P.Type(t.elt);

	case Scanner.CHAN:
		var m string;
		switch t.mode {
		case AST.FULL: m = "chan ";
		case AST.RECV: m = "<-chan ";
		case AST.SEND: m = "chan <- ";
		}
		P.String(t.pos, m);
		P.Type(t.elt);

	case Scanner.MUL:
		P.String(t.pos, "*");
		P.Type(t.elt);

	case Scanner.LPAREN:
		P.Parameters(t.pos, t.list);
		if t.elt != nil {
			P.Blank();
			P.Parameters(0, t.elt.list);
		}

	case Scanner.ELLIPSIS:
		P.String(t.pos, "...");

	default:
		P.Error(t.pos, t.tok, "type");
	}
}


// ----------------------------------------------------------------------------
// Expressions

func (P *Printer) Block(list *AST.List, indent bool);

func (P *Printer) Expr1(x *AST.Expr, prec1 int) {
	if x == nil {
		return;  // empty expression list
	}

	switch x.tok {
	case Scanner.TYPE:
		// type expr
		P.Type(x.t);

	case Scanner.IDENT, Scanner.INT, Scanner.STRING, Scanner.FLOAT:
		// literal
		P.String(x.pos, x.s);

	case Scanner.FUNC:
		// function literal
		P.String(x.pos, "func");
		P.Type(x.t);
		P.Block(x.block, true);
		P.newl = 0;

	case Scanner.COMMA:
		// list
		// (don't use binary expression printing because of different spacing)
		P.Expr(x.x);
		P.String(x.pos, ", ");
		P.Expr(x.y);

	case Scanner.PERIOD:
		// selector or type guard
		P.Expr1(x.x, Scanner.HighestPrec);
		P.String(x.pos, ".");
		if x.y != nil {
			P.Expr1(x.y, Scanner.HighestPrec);
		} else {
			P.String(0, "(");
			P.Type(x.t);
			P.String(0, ")");
		}
		
	case Scanner.LBRACK:
		// index
		P.Expr1(x.x, Scanner.HighestPrec);
		P.String(x.pos, "[");
		P.Expr1(x.y, 0);
		P.String(0, "]");

	case Scanner.LPAREN:
		// call
		P.Expr1(x.x, Scanner.HighestPrec);
		P.String(x.pos, "(");
		P.Expr(x.y);
		P.String(0, ")");

	case Scanner.LBRACE:
		// composite
		P.Type(x.t);
		P.String(x.pos, "{");
		P.Expr(x.y);
		P.String(0, "}");
		
	default:
		// unary and binary expressions including ":" for pairs
		prec := Scanner.UnaryPrec;
		if x.x != nil {
			prec = Scanner.Precedence(x.tok);
		}
		if prec < prec1 {
			P.String(0, "(");
		}
		if x.x == nil {
			// unary expression
			P.Token(x.pos, x.tok);
		} else {
			// binary expression
			P.Expr1(x.x, prec);
			P.Blank();
			P.Token(x.pos, x.tok);
			P.Blank();
		}
		P.Expr1(x.y, prec);
		if prec < prec1 {
			P.String(0, ")");
		}
	}
}


func (P *Printer) Expr(x *AST.Expr) {
	P.Expr1(x, Scanner.LowestPrec);
}


// ----------------------------------------------------------------------------
// Statements

func (P *Printer) Stat(s *AST.Stat)

func (P *Printer) StatementList(list *AST.List) {
	for i, n := 0, list.len(); i < n; i++ {
		P.Stat(list.at(i).(*AST.Stat));
		P.newl = 1;
	}
}


func (P *Printer) Block(list *AST.List, indent bool) {
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


func (P *Printer) ControlClause(s *AST.Stat) {
	has_post := s.tok == Scanner.FOR && s.post != nil;  // post also used by "if"
	if s.init == nil && !has_post {
		// no semicolons required
		if s.expr != nil {
			P.Blank();
			P.Expr(s.expr);
		}
	} else {
		// all semicolons required
		P.Blank();
		if s.init != nil {
			P.Stat(s.init);
		}
		P.semi = true;
		P.Blank();
		if s.expr != nil {
			P.Expr(s.expr);
		}
		if s.tok == Scanner.FOR {
			P.semi = true;
			if has_post {
				P.Blank();
				P.Stat(s.post);
				P.semi = false
			}
		}
	}
	P.Blank();
}


func (P *Printer) Declaration(d *AST.Decl, parenthesized bool);

func (P *Printer) Stat(s *AST.Stat) {
	switch s.tok {
	case Scanner.EXPRSTAT:
		// expression statement
		P.Expr(s.expr);
		P.semi = true;

	case Scanner.COLON:
		// label declaration
		P.indent--;
		P.Expr(s.expr);
		P.Token(s.pos, s.tok);
		P.indent++;
		P.semi = false;
		
	case Scanner.CONST, Scanner.TYPE, Scanner.VAR:
		// declaration
		P.Declaration(s.decl, false);

	case Scanner.INC, Scanner.DEC:
		P.Expr(s.expr);
		P.Token(s.pos, s.tok);
		P.semi = true;

	case Scanner.LBRACE:
		// block
		P.Block(s.block, true);

	case Scanner.IF:
		P.String(s.pos, "if");
		P.ControlClause(s);
		P.Block(s.block, true);
		if s.post != nil {
			P.newl = 0;
			P.String(0, " else ");
			P.Stat(s.post);
		}

	case Scanner.FOR:
		P.String(s.pos, "for");
		P.ControlClause(s);
		P.Block(s.block, true);

	case Scanner.SWITCH, Scanner.SELECT:
		P.Token(s.pos, s.tok);
		P.ControlClause(s);
		P.Block(s.block, false);

	case Scanner.CASE, Scanner.DEFAULT:
		P.Token(s.pos, s.tok);
		if s.expr != nil {
			P.Blank();
			P.Expr(s.expr);
		}
		P.String(0, ":");
		P.indent++;
		P.newl = 1;
		P.StatementList(s.block);
		P.indent--;
		P.newl = 1;

	case Scanner.GO, Scanner.RETURN, Scanner.FALLTHROUGH, Scanner.BREAK, Scanner.CONTINUE, Scanner.GOTO:
		P.Token(s.pos, s.tok);
		if s.expr != nil {
			P.Blank();
			P.Expr(s.expr);
		}
		P.semi = true;

	default:
		P.Error(s.pos, s.tok, "stat");
	}
}


// ----------------------------------------------------------------------------
// Declarations


func (P *Printer) Declaration(d *AST.Decl, parenthesized bool) {
	if !parenthesized {
		if d.exported {
			P.String(0, "export ");
		}
		P.Token(d.pos, d.tok);
		P.Blank();
	}

	if d.tok != Scanner.FUNC && d.list != nil {
		P.OpenScope("(");
		for i := 0; i < d.list.len(); i++ {
			P.Declaration(d.list.at(i).(*AST.Decl), true);
			P.semi, P.newl = true, 1;
		}
		P.CloseScope(")");

	} else {
		if d.tok == Scanner.FUNC && d.typ.key != nil {
			P.Parameters(0, d.typ.key.list);
			P.Blank();
		}

		P.Expr(d.ident);
		
		if d.typ != nil {
			P.Blank();
			P.Type(d.typ);
		}

		if d.val != nil {
			if d.tok == Scanner.IMPORT {
				P.Blank();
			} else {
				P.String(0, " = ");
			}
			P.Expr(d.val);
		}

		if d.list != nil {
			if d.tok != Scanner.FUNC {
				panic("must be a func declaration");
			}
			P.Blank();
			P.Block(d.list, true);
		}
		
		if d.tok != Scanner.TYPE {
			P.semi = true;
		}
	}
	
	P.newl = 1;

	// extra newline after a function declaration
	if d.tok == Scanner.FUNC {
		P.newl++;
	}
	
	// extra newline at the top level
	if P.level == 0 {
		P.newl++;
	}
}


// ----------------------------------------------------------------------------
// Program

func (P *Printer) Program(p *AST.Program) {
	// TODO should initialize all fields?
	P.buf.Init();
	
	P.clist = p.comments;
	P.cindex = 0;
	if p.comments.len() > 0 {
		P.cpos = p.comments.at(0).(*AST.Comment).pos;
	} else {
		P.cpos = 1000000000;  // infinite
	}

	// Print package
	P.String(p.pos, "package ");
	P.Expr(p.ident);
	P.newl = 2;
	for i := 0; i < p.decls.len(); i++ {
		P.Declaration(p.decls.at(i), false);
	}
	P.newl = 1;

	P.String(0, "");  // flush
}
