// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Printer

import (
	"os";
	"io";
	"array";
	"tabwriter";
	"flag";
	"fmt";
	Utils "utils";
	Scanner "scanner";
	AST "ast";
)

var (
	debug = flag.Bool("debug", false, "print debugging information");

	// layout control
	tabwidth = flag.Int("tabwidth", 8, "tab width");
	usetabs = flag.Bool("usetabs", true, "align with tabs instead of blanks");
	newlines = flag.Bool("newlines", true, "respect newlines in source");
	maxnewlines = flag.Int("maxnewlines", 3, "max. number of consecutive newlines");

	// formatting control
	html = flag.Bool("html", false, "generate html");
	comments = flag.Bool("comments", true, "print comments");
	optsemicolons = flag.Bool("optsemicolons", false, "print optional semicolons");
)


// ----------------------------------------------------------------------------
// Printer

// Separators - printed in a delayed fashion, depending on context.
const (
	none = iota;
	blank;
	tab;
	comma;
	semicolon;
)


// Semantic states - control formatting.
const (
	normal = iota;
	opening_scope;  // controls indentation, scope level
	closing_scope;  // controls indentation, scope level
	inside_list;  // controls extra line breaks
)


type Printer struct {
	// output
	text io.Write;

	// comments
	comments *array.Array;  // the list of all comments
	cindex int;  // the current comments index
	cpos int;  // the position of the next comment

	// current state
	lastpos int;  // pos after last string
	level int;  // scope level
	indentation int;  // indentation level (may be different from scope level)

	// formatting parameters
	separator int;  // pending separator
	newlines int;  // pending newlines

	// semantic state
	state int;  // current semantic state
	laststate int;  // state for last string
}


func (P *Printer) HasComment(pos int) bool {
	return *comments && P.cpos < pos;
}


func (P *Printer) NextComment() {
	P.cindex++;
	if P.comments != nil && P.cindex < P.comments.Len() {
		P.cpos = P.comments.At(P.cindex).(*AST.Comment).Pos;
	} else {
		P.cpos = 1<<30;  // infinite
	}
}


func (P *Printer) Init(text io.Write, comments *array.Array) {
	// writers
	P.text = text;

	// comments
	P.comments = comments;
	P.cindex = -1;
	P.NextComment();

	// formatting parameters & semantic state initialized correctly by default
}


// ----------------------------------------------------------------------------
// Printing support

func htmlEscape(s string) string {
	if *html {
		var esc string;
		for i := 0; i < len(s); i++ {
			switch s[i] {
			case '<': esc = "&lt;";
			case '&': esc = "&amp;";
			default: continue;
			}
			return s[0 : i] + esc + htmlEscape(s[i+1 : len(s)]);
		}
	}
	return s;
}


// Reduce contiguous sequences of '\t' in a string to a single '\t'.
func untabify(s string) string {
	for i := 0; i < len(s); i++ {
		if s[i] == '\t' {
			j := i;
			for j < len(s) && s[j] == '\t' {
				j++;
			}
			if j-i > 1 {  // more then one tab
				return s[0 : i+1] + untabify(s[j : len(s)]);
			}
		}
	}
	return s;
}


func (P *Printer) Printf(format string, s ...) {
	n, err := fmt.Fprintf(P.text, format, s);
	if err != nil {
		panic("print error - exiting");
	}
}


func (P *Printer) Newline(n int) {
	if n > 0 {
		m := int(*maxnewlines);
		if n > m {
			n = m;
		}
		for ; n > 0; n-- {
			P.Printf("\n");
		}
		for i := P.indentation; i > 0; i-- {
			P.Printf("\t");
		}
	}
}


func (P *Printer) TaggedString(pos int, tag, s, endtag string) {
	// use estimate for pos if we don't have one
	if pos == 0 {
		pos = P.lastpos;
	}

	// --------------------------------
	// print pending separator, if any
	// - keep track of white space printed for better comment formatting
	// TODO print white space separators after potential comments and newlines
	// (currently, we may get trailing white space before a newline)
	trailing_char := 0;
	switch P.separator {
	case none:	// nothing to do
	case blank:
		P.Printf(" ");
		trailing_char = ' ';
	case tab:
		P.Printf("\t");
		trailing_char = '\t';
	case comma:
		P.Printf(",");
		if P.newlines == 0 {
			P.Printf(" ");
			trailing_char = ' ';
		}
	case semicolon:
		if P.level > 0 {	// no semicolons at level 0
			P.Printf(";");
			if P.newlines == 0 {
				P.Printf(" ");
				trailing_char = ' ';
			}
		}
	default:	panic("UNREACHABLE");
	}
	P.separator = none;

	// --------------------------------
	// interleave comments, if any
	nlcount := 0;
	for ; P.HasComment(pos); P.NextComment() {
		// we have a comment/newline that comes before the string
		comment := P.comments.At(P.cindex).(*AST.Comment);
		ctext := comment.Text;

		if ctext == "\n" {
			// found a newline in src - count it
			nlcount++;

		} else {
			// classify comment (len(ctext) >= 2)
			//-style comment
			if nlcount > 0 || P.cpos == 0 {
				// only white space before comment on this line
				// or file starts with comment
				// - indent
				if !*newlines && P.cpos != 0 {
					nlcount = 1;
				}
				P.Newline(nlcount);
				nlcount = 0;

			} else {
				// black space before comment on this line
				if ctext[1] == '/' {
					//-style comment
					// - put in next cell unless a scope was just opened
					//   in which case we print 2 blanks (otherwise the
					//   entire scope gets indented like the next cell)
					if P.laststate == opening_scope {
						switch trailing_char {
						case ' ': P.Printf(" ");  // one space already printed
						case '\t': // do nothing
						default: P.Printf("  ");
						}
					} else {
						if trailing_char != '\t' {
							P.Printf("\t");
						}
					}
				} else {
					/*-style comment */
					// - print surrounded by blanks
					if trailing_char == 0 {
						P.Printf(" ");
					}
					ctext += " ";
				}
			}

			// print comment
			if *debug {
				P.Printf("[%d]", P.cpos);
			}
			// calling untabify increases the change for idempotent output
			// since tabs in comments are also interpreted by tabwriter
			P.Printf("%s", htmlEscape(untabify(ctext)));

			if ctext[1] == '/' {
				//-style comments must end in newline
				if P.newlines == 0 {  // don't add newlines if not needed
					P.newlines = 1;
				}
			}
		}
	}
	// At this point we may have nlcount > 0: In this case we found newlines
	// that were not followed by a comment. They are recognized (or not) when
	// printing newlines below.

	// --------------------------------
	// interpret state
	// (any pending separator or comment must be printed in previous state)
	switch P.state {
	case normal:
	case opening_scope:
	case closing_scope:
		P.indentation--;
	case inside_list:
	default:
		panic("UNREACHABLE");
	}

	// --------------------------------
	// print pending newlines
	if *newlines && (P.newlines > 0 || P.state == inside_list) && nlcount > P.newlines {
		// Respect additional newlines in the source, but only if we
		// enabled this feature (newlines.BVal()) and we are expecting
		// newlines (P.newlines > 0 || P.state == inside_list).
		// Otherwise - because we don't have all token positions - we
		// get funny formatting.
		P.newlines = nlcount;
	}
	nlcount = 0;
	P.Newline(P.newlines);
	P.newlines = 0;

	// --------------------------------
	// print string
	if *debug {
		P.Printf("[%d]", pos);
	}
	P.Printf("%s%s%s", tag, htmlEscape(s), endtag);

	// --------------------------------
	// interpret state
	switch P.state {
	case normal:
	case opening_scope:
		P.level++;
		P.indentation++;
	case closing_scope:
		P.level--;
	case inside_list:
	default:
		panic("UNREACHABLE");
	}
	P.laststate = P.state;
	P.state = none;

	// --------------------------------
	// done
	P.lastpos = pos + len(s);  // rough estimate
}


func (P *Printer) String(pos int, s string) {
	P.TaggedString(pos, "", s, "");
}


func (P *Printer) Token(pos int, tok int) {
	P.String(pos, Scanner.TokenString(tok));
}


func (P *Printer) Error(pos int, tok int, msg string) {
	P.String(0, "<");
	P.Token(pos, tok);
	P.String(0, " " + msg + ">");
}


// ----------------------------------------------------------------------------
// HTML support

func (P *Printer) HtmlPrologue(title string) {
	if *html {
		P.TaggedString(0,
			"<html>\n"
			"<head>\n"
			"	<META HTTP-EQUIV=\"Content-Type\" CONTENT=\"text/html; charset=UTF-8\">\n"
			"	<title>" + htmlEscape(title) + "</title>\n"
			"	<style type=\"text/css\">\n"
			"	</style>\n"
			"</head>\n"
			"<body>\n"
			"<pre>\n",
			"", ""
		)
	}
}


func (P *Printer) HtmlEpilogue() {
	if *html {
		P.TaggedString(0,
			"</pre>\n"
			"</body>\n"
			"<html>\n",
			"", ""
		)
	}
}


func (P *Printer) HtmlIdentifier(x *AST.Expr) {
	if x.Tok != Scanner.IDENT {
		panic();
	}
	obj := x.Obj;
	if *html && obj.Kind != AST.NONE {
		// depending on whether we have a declaration or use, generate different html
		// - no need to htmlEscape ident
		id := Utils.IntToString(obj.Id, 10);
		if x.Pos == obj.Pos {
			// probably the declaration of x
			P.TaggedString(x.Pos, `<a name="id` + id + `">`, obj.Ident, `</a>`);
		} else {
			// probably not the declaration of x
			P.TaggedString(x.Pos, `<a href="#id` + id + `">`, obj.Ident, `</a>`);
		}
	} else {
		P.String(x.Pos, obj.Ident);
	}
}


// ----------------------------------------------------------------------------
// Types

func (P *Printer) Type(t *AST.Type) int
func (P *Printer) Expr(x *AST.Expr)

func (P *Printer) Parameters(pos int, list *array.Array) {
	P.String(pos, "(");
	if list != nil {
		var prev int;
		for i, n := 0, list.Len(); i < n; i++ {
			x := list.At(i).(*AST.Expr);
			if i > 0 {
				if prev == x.Tok || prev == Scanner.TYPE {
					P.separator = comma;
				} else {
					P.separator = blank;
				}
			}
			P.Expr(x);
			prev = x.Tok;
		}
	}
	P.String(0, ")");
}


func (P *Printer) Fields(list *array.Array, end int) {
	P.state = opening_scope;
	P.String(0, "{");

	if list.Len() > 0 {
		P.newlines = 1;
		var prev int;
		for i, n := 0, list.Len(); i < n; i++ {
			x := list.At(i).(*AST.Expr);
			if i > 0 {
				if prev == Scanner.TYPE && x.Tok != Scanner.STRING || prev == Scanner.STRING {
					P.separator = semicolon;
					P.newlines = 1;
				} else if prev == x.Tok {
					P.separator = comma;
				} else {
					P.separator = tab;
				}
			}
			P.Expr(x);
			prev = x.Tok;
		}
		P.newlines = 1;
	}

	P.state = closing_scope;
	P.String(end, "}");
}


// Returns the separator (semicolon or none) required if
// the type is terminating a declaration or statement.
func (P *Printer) Type(t *AST.Type) int {
	separator := semicolon;

	switch t.Form {
	case AST.TYPENAME:
		P.Expr(t.Expr);

	case AST.ARRAY:
		P.String(t.Pos, "[");
		if t.Expr != nil {
			P.Expr(t.Expr);
		}
		P.String(0, "]");
		separator = P.Type(t.Elt);

	case AST.STRUCT, AST.INTERFACE:
		switch t.Form {
		case AST.STRUCT: P.String(t.Pos, "struct");
		case AST.INTERFACE: P.String(t.Pos, "interface");
		}
		if t.List != nil {
			P.separator = blank;
			P.Fields(t.List, t.End);
		}
		separator = none;

	case AST.MAP:
		P.String(t.Pos, "map [");
		P.Type(t.Key);
		P.String(0, "]");
		separator = P.Type(t.Elt);

	case AST.CHANNEL:
		var m string;
		switch t.Mode {
		case AST.FULL: m = "chan ";
		case AST.RECV: m = "<-chan ";
		case AST.SEND: m = "chan <- ";
		}
		P.String(t.Pos, m);
		separator = P.Type(t.Elt);

	case AST.POINTER:
		P.String(t.Pos, "*");
		separator = P.Type(t.Elt);

	case AST.FUNCTION:
		P.Parameters(t.Pos, t.List);
		if t.Elt != nil {
			P.separator = blank;
			list := t.Elt.List;
			if list.Len() > 1 {
				P.Parameters(0, list);
			} else {
				// single, anonymous result type
				P.Expr(list.At(0).(*AST.Expr));
			}
		}

	case AST.ELLIPSIS:
		P.String(t.Pos, "...");

	default:
		P.Error(t.Pos, t.Form, "type");
	}

	return separator;
}


// ----------------------------------------------------------------------------
// Expressions

func (P *Printer) Block(pos int, list *array.Array, end int, indent bool);

func (P *Printer) Expr1(x *AST.Expr, prec1 int) {
	if x == nil {
		return;  // empty expression list
	}

	switch x.Tok {
	case Scanner.TYPE:
		// type expr
		P.Type(x.Obj.Typ);

	case Scanner.IDENT:
		P.HtmlIdentifier(x);

	case Scanner.INT, Scanner.STRING, Scanner.FLOAT:
		// literal
		P.String(x.Pos, x.Obj.Ident);

	case Scanner.FUNC:
		// function literal
		P.String(x.Pos, "func");
		P.Type(x.Obj.Typ);
		P.Block(0, x.Obj.Block, x.Obj.End, true);
		P.newlines = 0;

	case Scanner.COMMA:
		// list
		// (don't use binary expression printing because of different spacing)
		P.Expr(x.X);
		P.String(x.Pos, ",");
		P.separator = blank;
		P.state = inside_list;
		P.Expr(x.Y);

	case Scanner.PERIOD:
		// selector or type guard
		P.Expr1(x.X, Scanner.HighestPrec);
		P.String(x.Pos, ".");
		if x.Y.Tok == Scanner.TYPE {
			P.String(0, "(");
			P.Expr(x.Y);
			P.String(0, ")");
		} else {
			P.Expr1(x.Y, Scanner.HighestPrec);
		}

	case Scanner.LBRACK:
		// index
		P.Expr1(x.X, Scanner.HighestPrec);
		P.String(x.Pos, "[");
		P.Expr1(x.Y, 0);
		P.String(0, "]");

	case Scanner.LPAREN:
		// call
		P.Expr1(x.X, Scanner.HighestPrec);
		P.String(x.Pos, "(");
		P.Expr(x.Y);
		P.String(0, ")");

	case Scanner.LBRACE:
		// composite literal
		P.Type(x.Obj.Typ);
		P.String(x.Pos, "{");
		P.Expr(x.Y);
		P.String(0, "}");

	default:
		// unary and binary expressions including ":" for pairs
		prec := Scanner.UnaryPrec;
		if x.X != nil {
			prec = Scanner.Precedence(x.Tok);
		}
		if prec < prec1 {
			P.String(0, "(");
		}
		if x.X == nil {
			// unary expression
			P.Token(x.Pos, x.Tok);
			if x.Tok == Scanner.RANGE {
				P.separator = blank;
			}
		} else {
			// binary expression
			P.Expr1(x.X, prec);
			P.separator = blank;
			P.Token(x.Pos, x.Tok);
			P.separator = blank;
		}
		P.Expr1(x.Y, prec);
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
		P.newlines = 1;
		for i, n := 0, list.Len(); i < n; i++ {
			P.Stat(list.At(i).(*AST.Stat));
			P.newlines = 1;
			P.state = inside_list;
		}
	}
}


func (P *Printer) Block(pos int, list *array.Array, end int, indent bool) {
	P.state = opening_scope;
	P.String(pos, "{");
	if !indent {
		P.indentation--;
	}
	P.StatementList(list);
	if !indent {
		P.indentation++;
	}
	if !*optsemicolons {
		P.separator = none;
	}
	P.state = closing_scope;
	P.String(end, "}");
}


func (P *Printer) ControlClause(s *AST.Stat) {
	has_post := s.Tok == Scanner.FOR && s.Post != nil;  // post also used by "if"

	P.separator = blank;
	if s.Init == nil && !has_post {
		// no semicolons required
		if s.Expr != nil {
			P.Expr(s.Expr);
		}
	} else {
		// all semicolons required
		// (they are not separators, print them explicitly)
		if s.Init != nil {
			P.Stat(s.Init);
			P.separator = none;
		}
		P.String(0, ";");
		P.separator = blank;
		if s.Expr != nil {
			P.Expr(s.Expr);
			P.separator = none;
		}
		if s.Tok == Scanner.FOR {
			P.String(0, ";");
			P.separator = blank;
			if has_post {
				P.Stat(s.Post);
			}
		}
	}
	P.separator = blank;
}


func (P *Printer) Declaration(d *AST.Decl, parenthesized bool);

func (P *Printer) Stat(s *AST.Stat) {
	switch s.Tok {
	case Scanner.EXPRSTAT:
		// expression statement
		P.Expr(s.Expr);
		P.separator = semicolon;

	case Scanner.COLON:
		// label declaration
		P.indentation--;
		P.Expr(s.Expr);
		P.Token(s.Pos, s.Tok);
		P.indentation++;
		P.separator = none;

	case Scanner.CONST, Scanner.TYPE, Scanner.VAR:
		// declaration
		P.Declaration(s.Decl, false);

	case Scanner.INC, Scanner.DEC:
		P.Expr(s.Expr);
		P.Token(s.Pos, s.Tok);
		P.separator = semicolon;

	case Scanner.LBRACE:
		// block
		P.Block(s.Pos, s.Block, s.End, true);

	case Scanner.IF:
		P.String(s.Pos, "if");
		P.ControlClause(s);
		P.Block(0, s.Block, s.End, true);
		if s.Post != nil {
			P.separator = blank;
			P.String(0, "else");
			P.separator = blank;
			P.Stat(s.Post);
		}

	case Scanner.FOR:
		P.String(s.Pos, "for");
		P.ControlClause(s);
		P.Block(0, s.Block, s.End, true);

	case Scanner.SWITCH, Scanner.SELECT:
		P.Token(s.Pos, s.Tok);
		P.ControlClause(s);
		P.Block(0, s.Block, s.End, false);

	case Scanner.CASE, Scanner.DEFAULT:
		P.Token(s.Pos, s.Tok);
		if s.Expr != nil {
			P.separator = blank;
			P.Expr(s.Expr);
		}
		P.String(0, ":");
		P.indentation++;
		P.StatementList(s.Block);
		P.indentation--;
		P.newlines = 1;

	case Scanner.GO, Scanner.RETURN, Scanner.FALLTHROUGH, Scanner.BREAK, Scanner.CONTINUE, Scanner.GOTO:
		P.Token(s.Pos, s.Tok);
		if s.Expr != nil {
			P.separator = blank;
			P.Expr(s.Expr);
		}
		P.separator = semicolon;

	default:
		P.Error(s.Pos, s.Tok, "stat");
	}
}


// ----------------------------------------------------------------------------
// Declarations

func (P *Printer) Declaration(d *AST.Decl, parenthesized bool) {
	if !parenthesized {
		if d.Exported {
			P.String(d.Pos, "export");
			P.separator = blank;
		}
		P.Token(d.Pos, d.Tok);
		P.separator = blank;
	}

	if d.Tok != Scanner.FUNC && d.List != nil {
		// group of parenthesized declarations
		P.state = opening_scope;
		P.String(0, "(");
		if d.List.Len() > 0 {
			P.newlines = 1;
			for i := 0; i < d.List.Len(); i++ {
				P.Declaration(d.List.At(i).(*AST.Decl), true);
				P.separator = semicolon;
				P.newlines = 1;
			}
		}
		P.state = closing_scope;
		P.String(d.End, ")");

	} else {
		// single declaration
		switch d.Tok {
		case Scanner.IMPORT:
			if d.Ident != nil {
				P.Expr(d.Ident);
			} else {
				P.String(d.Val.Pos, "");  // flush pending ';' separator/newlines
			}
			P.separator = tab;
			P.Expr(d.Val);
			P.separator = semicolon;

		case Scanner.EXPORT:
			P.Expr(d.Ident);
			P.separator = semicolon;

		case Scanner.TYPE:
			P.Expr(d.Ident);
			P.separator = blank;  // TODO switch to tab? (but indentation problem with structs)
			P.separator = P.Type(d.Typ);

		case Scanner.CONST, Scanner.VAR:
			P.Expr(d.Ident);
			if d.Typ != nil {
				P.separator = blank;  // TODO switch to tab? (indentation problem with structs)
				P.separator = P.Type(d.Typ);
			}
			if d.Val != nil {
				P.separator = tab;
				P.String(0, "=");
				P.separator = blank;
				P.Expr(d.Val);
			}
			P.separator = semicolon;

		case Scanner.FUNC:
			if d.Typ.Key != nil {
				// method: print receiver
				P.Parameters(0, d.Typ.Key.List);
				P.separator = blank;
			}
			P.Expr(d.Ident);
			P.separator = P.Type(d.Typ);
			if d.List != nil {
				P.separator = blank;
				P.Block(0, d.List, d.End, true);
			}

		default:
			P.Error(d.Pos, d.Tok, "decl");
		}
	}

	P.newlines = 2;
}


// ----------------------------------------------------------------------------
// Program

func (P *Printer) Program(p *AST.Program) {
	P.String(p.Pos, "package");
	P.separator = blank;
	P.Expr(p.Ident);
	P.newlines = 1;
	for i := 0; i < p.Decls.Len(); i++ {
		P.Declaration(p.Decls.At(i).(*AST.Decl), false);
	}
	P.newlines = 1;
}


// ----------------------------------------------------------------------------
// External interface

func Print(prog *AST.Program) {
	// setup
	var P Printer;
	padchar := byte(' ');
	if *usetabs {
		padchar = '\t';
	}
	text := tabwriter.New(os.Stdout, *tabwidth, 1, padchar, true, *html);
	P.Init(text, prog.Comments);

	// TODO would be better to make the name of the src file be the title
	P.HtmlPrologue("package " + prog.Ident.Obj.Ident);
	P.Program(prog);
	P.HtmlEpilogue();

	P.String(0, "");  // flush pending separator/newlines
	err := text.Flush();
	if err != nil {
		panic("print error - exiting");
	}
}
