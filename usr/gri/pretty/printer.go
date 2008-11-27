// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Printer

import (
	"os";
	"array";
	"tabwriter";
	"flag";
	"fmt";
	Scanner "scanner";
	AST "ast";
)

var (
	debug = flag.Bool("debug", false, nil, "print debugging information");
	tabwidth = flag.Int("tabwidth", 4, nil, "tab width");
	usetabs = flag.Bool("usetabs", true, nil, "align with tabs instead of blanks");
	comments = flag.Bool("comments", false, nil, "enable printing of comments");
)


// ----------------------------------------------------------------------------
// Printer

// A variety of separators which are printed in a delayed fashion;
// depending on the next token.
const (
	none = iota;
	blank;
	tab;
	comma;
	semicolon;
)


// Additional printing state to control the output. Fine-tuning
// can be achieved by adding more specific state.
const (
	inline = iota;
	lineend;
	funcend;
)


type Printer struct {
	// output
	writer *tabwriter.Writer;
	
	// comments
	comments *array.Array;
	cindex int;
	cpos int;

	// formatting control
	lastpos int;  // pos after last string
	level int;  // true scope level
	indent int;  // indentation level
	separator int;  // pending separator
	state int;  // state info
}


func (P *Printer) NextComment() {
	P.cindex++;
	if P.comments != nil && P.cindex < P.comments.Len() {
		P.cpos = P.comments.At(P.cindex).(*AST.Comment).pos;
	} else {
		P.cpos = 1<<30;  // infinite
	}
}


func (P *Printer) Init(writer *tabwriter.Writer, comments *array.Array) {
	// writer
	padchar := byte(' ');
	if usetabs.BVal() {
		padchar = '\t';
	}
	P.writer = tabwriter.New(os.Stdout, int(tabwidth.IVal()), 1, padchar, true);

	// comments
	P.comments = comments;
	P.cindex = -1;
	P.NextComment();
	
	// formatting control initialized correctly by default
}


// ----------------------------------------------------------------------------
// Printing support

func (P *Printer) Printf(format string, s ...) {
	n, err := fmt.fprintf(P.writer, format, s);
	if err != nil {
		panic("print error - exiting");
	}
}


func (P *Printer) Newline() {
	P.Printf("\n");
	for i := P.indent; i > 0; i-- {
		P.Printf("\t");
	}
}


func (P *Printer) String(pos int, s string) {
	// correct pos if necessary
	if pos == 0 {
		pos = P.lastpos;  // estimate
	}

	// --------------------------------
	// print pending separator, if any
	// - keep track of white space printed for better comment formatting
	trailing_blank := false;
	trailing_tab := false;
	switch P.separator {
	case none:	// nothing to do
	case blank:
		P.Printf(" ");
		trailing_blank = true;
	case tab:
		P.Printf("\t");
		trailing_tab = true;
	case comma:
		P.Printf(",");
		if P.state == inline {
			P.Printf(" ");
			trailing_blank = true;
		}
	case semicolon:
		if P.level > 0 {	// no semicolons at level 0
			P.Printf(";");
			if P.state == inline {
				P.Printf(" ");
				trailing_blank = true;
			}
		}
	default:	panic("UNREACHABLE");
	}
	P.separator = none;

	// --------------------------------
	// interleave comments, if any
	nlcount := 0;
	for comments.BVal() && P.cpos < pos {
		// we have a comment/newline that comes before the string
		comment := P.comments.At(P.cindex).(*AST.Comment);
		ctext := comment.text;
		
		if ctext == "\n" {
			// found a newline in src - count them
			nlcount++;

		} else {
			// classify comment (len(ctext) >= 2)
			//-style comment
			if nlcount > 0 || P.cpos == 0 {
				// only white space before comment on this line
				// or file starts with comment
				// - indent
				P.Newline();
			} else {
				// black space before comment on this line
				if ctext[1] == '/' {
					//-style comment
					// - put in next cell
					if !trailing_tab {
						P.Printf("\t");
					}
				} else {
					/*-style comment */
					// - print surrounded by blanks
					if !trailing_blank && !trailing_tab {
						P.Printf(" ");
					}
					ctext += " ";
				}
			}
			
			if debug.BVal() {
				P.Printf("[%d]", P.cpos);
			}
			P.Printf("%s", ctext);

			if ctext[1] == '/' {
				//-style comments must end in newline
				if P.state == inline {  // don't override non-inline states
					P.state = lineend;
				}
			}
			
			nlcount = 0;
		}

		P.NextComment();
	}

	// --------------------------------
	// adjust formatting depending on state
	switch P.state {
	case inline:	// nothing to do
	case funcend:
		P.Printf("\n\n");
		fallthrough;
	case lineend:
		P.Newline();
	default:	panic("UNREACHABLE");
	}
	P.state = inline;

	// --------------------------------
	// print string
	if debug.BVal() {
		P.Printf("[%d]", pos);
	}
	P.Printf("%s", s);

	// --------------------------------
	// done
	P.lastpos = pos + len(s);  // rough estimate
}


func (P *Printer) Separator(separator int) {
	P.separator = separator;
	P.String(0, "");
}


func (P *Printer) Token(pos int, tok int) {
	P.String(pos, Scanner.TokenString(tok));
}


func (P *Printer) OpenScope(paren string) {
	P.String(0, paren);
	P.level++;
	P.indent++;
	P.state = lineend;
}


func (P *Printer) CloseScope(pos int, paren string) {
	P.indent--;
	P.String(pos, paren);
	P.level--;
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

func (P *Printer) Parameters(pos int, list *array.Array) {
	P.String(pos, "(");
	if list != nil {
		var prev int;
		for i, n := 0, list.Len(); i < n; i++ {
			x := list.At(i).(*AST.Expr);
			if i > 0 {
				if prev == x.tok || prev == Scanner.TYPE {
					P.Separator(comma);
				} else {
					P.Separator(blank);
				}
			}
			P.Expr(x);
			prev = x.tok;
		}
	}
	P.String(0, ")");
}


func (P *Printer) Fields(list *array.Array, end int) {
	P.OpenScope("{");
	if list != nil {
		var prev int;
		for i, n := 0, list.Len(); i < n; i++ {
			x := list.At(i).(*AST.Expr);
			if i > 0 {
				if prev == Scanner.TYPE && x.tok != Scanner.STRING || prev == Scanner.STRING {
					P.separator = semicolon;
					P.state = lineend;
				} else if prev == x.tok {
					P.separator = comma;
				} else {
					P.separator = tab;
				}
			}
			P.Expr(x);
			prev = x.tok;
		}
		P.state = lineend;
	}
	P.CloseScope(end, "}");
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
			P.separator = blank;
			P.Fields(t.list, t.end);
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
			P.separator = blank;
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

func (P *Printer) Block(list *array.Array, end int, indent bool);

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
		P.Block(x.block, x.end, true);
		P.state = inline;

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
			P.separator = blank;
			P.Token(x.pos, x.tok);
			P.separator = blank;
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

func (P *Printer) StatementList(list *array.Array) {
	if list != nil {
		for i, n := 0, list.Len(); i < n; i++ {
			P.Stat(list.At(i).(*AST.Stat));
			P.state = lineend;
		}
	}
}


func (P *Printer) Block(list *array.Array, end int, indent bool) {
	P.OpenScope("{");
	if !indent {
		P.indent--;
	}
	P.StatementList(list);
	if !indent {
		P.indent++;
	}
	P.separator = none;
	P.CloseScope(end, "}");
}


func (P *Printer) ControlClause(s *AST.Stat) {
	has_post := s.tok == Scanner.FOR && s.post != nil;  // post also used by "if"

	P.separator = blank;
	if s.init == nil && !has_post {
		// no semicolons required
		if s.expr != nil {
			P.Expr(s.expr);
		}
	} else {
		// all semicolons required
		// (they are not separators, print them explicitly)
		if s.init != nil {
			P.Stat(s.init);
			P.separator = none;
		}
		P.Printf(";");
		P.separator = blank;
		if s.expr != nil {
			P.Expr(s.expr);
			P.separator = none;
		}
		if s.tok == Scanner.FOR {
			P.Printf(";");
			P.separator = blank;
			if has_post {
				P.Stat(s.post);
			}
		}
	}
	P.separator = blank;
}


func (P *Printer) Declaration(d *AST.Decl, parenthesized bool);

func (P *Printer) Stat(s *AST.Stat) {
	switch s.tok {
	case Scanner.EXPRSTAT:
		// expression statement
		P.Expr(s.expr);
		P.separator = semicolon;

	case Scanner.COLON:
		// label declaration
		P.indent--;
		P.Expr(s.expr);
		P.Token(s.pos, s.tok);
		P.indent++;
		P.separator = none;
		
	case Scanner.CONST, Scanner.TYPE, Scanner.VAR:
		// declaration
		P.Declaration(s.decl, false);

	case Scanner.INC, Scanner.DEC:
		P.Expr(s.expr);
		P.Token(s.pos, s.tok);
		P.separator = semicolon;

	case Scanner.LBRACE:
		// block
		P.Block(s.block, s.end, true);

	case Scanner.IF:
		P.String(s.pos, "if");
		P.ControlClause(s);
		P.Block(s.block, s.end, true);
		if s.post != nil {
			P.separator = blank;
			P.String(0, "else");
			P.separator = blank;
			P.Stat(s.post);
		}

	case Scanner.FOR:
		P.String(s.pos, "for");
		P.ControlClause(s);
		P.Block(s.block, s.end, true);

	case Scanner.SWITCH, Scanner.SELECT:
		P.Token(s.pos, s.tok);
		P.ControlClause(s);
		P.Block(s.block, s.end, false);

	case Scanner.CASE, Scanner.DEFAULT:
		P.Token(s.pos, s.tok);
		if s.expr != nil {
			P.separator = blank;
			P.Expr(s.expr);
		}
		P.String(0, ":");
		P.indent++;
		P.state = lineend;
		P.StatementList(s.block);
		P.indent--;
		P.state = lineend;

	case Scanner.GO, Scanner.RETURN, Scanner.FALLTHROUGH, Scanner.BREAK, Scanner.CONTINUE, Scanner.GOTO:
		P.Token(s.pos, s.tok);
		if s.expr != nil {
			P.separator = blank;
			P.Expr(s.expr);
		}
		P.separator = semicolon;

	default:
		P.Error(s.pos, s.tok, "stat");
	}
}


// ----------------------------------------------------------------------------
// Declarations


func (P *Printer) Declaration(d *AST.Decl, parenthesized bool) {
	if !parenthesized {
		if d.exported {
			P.String(d.pos, "export ");
		}
		P.Token(d.pos, d.tok);
		P.separator = blank;
	}

	if d.tok != Scanner.FUNC && d.list != nil {
		P.OpenScope("(");
		for i := 0; i < d.list.Len(); i++ {
			P.Declaration(d.list.At(i).(*AST.Decl), true);
			P.separator = semicolon;
			P.state = lineend;
		}
		P.CloseScope(d.end, ")");

	} else {
		if d.tok == Scanner.FUNC && d.typ.key != nil {
			P.Parameters(0, d.typ.key.list);
			P.separator = blank;
		}

		P.Expr(d.ident);
		
		if d.typ != nil {
			if d.tok != Scanner.FUNC {
				P.separator = blank;
			}
			P.Type(d.typ);
		}

		if d.val != nil {
			P.String(0, "\t");
			if d.tok != Scanner.IMPORT {
				P.String(0, "= ");
			}
			P.Expr(d.val);
		}

		if d.list != nil {
			if d.tok != Scanner.FUNC {
				panic("must be a func declaration");
			}
			P.separator = blank;
			P.Block(d.list, d.end, true);
		}
		
		if d.tok != Scanner.TYPE {
			P.separator = semicolon;
		}
	}
	
	if d.tok == Scanner.FUNC {
		P.state = funcend;
	} else {
		P.state = lineend;
	}
}


// ----------------------------------------------------------------------------
// Program

func (P *Printer) Program(p *AST.Program) {
	P.String(p.pos, "package ");
	P.Expr(p.ident);
	P.state = lineend;
	for i := 0; i < p.decls.Len(); i++ {
		P.Declaration(p.decls.At(i), false);
	}
	P.state = lineend;
}


// ----------------------------------------------------------------------------
// External interface

export func Print(prog *AST.Program) {
	// setup
	padchar := byte(' ');
	if usetabs.BVal() {
		padchar = '\t';
	}
	writer := tabwriter.New(os.Stdout, int(tabwidth.IVal()), 1, padchar, true);
	var P Printer;
	P.Init(writer, prog.comments);

	P.Program(prog);
	
	// flush
	P.String(0, "");
	err := P.writer.Flush();
	if err != nil {
		panic("print error - exiting");
	}
}
