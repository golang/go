// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Printer

import (
	"os";
	"io";
	"vector";
	"tabwriter";
	"flag";
	"fmt";
	"strings";
	"utils";
	"token";
	"scanner";
	"ast";
	"template";
	"utf8";
	"unicode";
	"symboltable";
)

var (
	debug = flag.Bool("debug", false, "print debugging information");

	// layout control
	tabwidth = flag.Int("tabwidth", 8, "tab width");
	usetabs = flag.Bool("usetabs", true, "align with tabs instead of blanks");
	newlines = flag.Bool("newlines", false, "respect newlines in source");
	maxnewlines = flag.Int("maxnewlines", 3, "max. number of consecutive newlines");

	// formatting control
	comments = flag.Bool("comments", true, "print comments");
	optsemicolons = flag.Bool("optsemicolons", false, "print optional semicolons");
)


// When we don't have a location use noloc.
// TODO make sure we always have a location.
var noloc scanner.Location;


// ----------------------------------------------------------------------------
// Elementary support

func unimplemented() {
	panic("unimplemented");
}


func unreachable() {
	panic("unreachable");
}


func assert(pred bool) {
	if !pred {
		panic("assertion failed");
	}
}


// TODO this should be an AST method
func isExported(name *ast.Ident) bool {
	ch, len := utf8.DecodeRuneInString(name.Str, 0);
	return unicode.IsUpper(ch);
}


func hasExportedNames(names []*ast.Ident) bool {
	for i, name := range names {
		if isExported(name) {
			return true;
		}
	}
	return false;
}


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
	
	// formatting control
	html bool;
	full bool;  // if false, print interface only; print all otherwise

	// comments
	comments []ast.CommentGroup;  // the list of all comments groups
	cindex int;  // the current comment group index
	cloc scanner.Location;  // the position of the next comment group

	// current state
	lastloc scanner.Location;  // location after last string
	level int;  // scope level
	indentation int;  // indentation level (may be different from scope level)

	// formatting parameters
	opt_semi bool;  // // true if semicolon separator is optional in statement list
	separator int;  // pending separator
	newlines int;  // pending newlines

	// semantic state
	state int;  // current semantic state
	laststate int;  // state for last string
	
	// expression precedence
	prec int;
}


func (P *Printer) hasComment(loc scanner.Location) bool {
	return *comments && P.cloc.Pos < loc.Pos;
}


func (P *Printer) nextCommentGroup() {
	P.cindex++;
	if P.comments != nil && P.cindex < len(P.comments) {
		P.cloc = P.comments[P.cindex][0].Loc;
	} else {
		P.cloc = scanner.Location{1<<30, 1<<30, 1};  // infinite
	}
}


func (P *Printer) Init(text io.Write, comments []ast.CommentGroup, html bool) {
	// writers
	P.text = text;
	
	// formatting control
	P.html = html;

	// comments
	P.comments = comments;
	P.cindex = -1;
	P.nextCommentGroup();

	// formatting parameters & semantic state initialized correctly by default
	
	// expression precedence
	P.prec = token.LowestPrec;
}


// ----------------------------------------------------------------------------
// Printing support

func (P *Printer) htmlEscape(s string) string {
	if P.html {
		var esc string;
		for i := 0; i < len(s); i++ {
			switch s[i] {
			case '<': esc = "&lt;";
			case '&': esc = "&amp;";
			default: continue;
			}
			return s[0 : i] + esc + P.htmlEscape(s[i+1 : len(s)]);
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


func (P *Printer) newline(n int) {
	if n > 0 {
		m := int(*maxnewlines);
		if n > m {
			n = m;
		}
		for n > 0 {
			P.Printf("\n");
			n--;
		}
		for i := P.indentation; i > 0; i-- {
			P.Printf("\t");
		}
	}
}


func (P *Printer) TaggedString(loc scanner.Location, tag, s, endtag string) {
	// use estimate for pos if we don't have one
	pos := loc.Pos;
	if pos == 0 {
		pos = P.lastloc.Pos;
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
	if P.full {
		for ; P.hasComment(loc); P.nextCommentGroup() {
			// we have a comment group that comes before the string
			comment := P.comments[P.cindex][0];  // TODO broken
			ctext := string(comment.Text);  // TODO get rid of string conversion here

			// classify comment (len(ctext) >= 2)
			//-style comment
			if nlcount > 0 || P.cloc.Pos == 0 {
				// only white space before comment on this line
				// or file starts with comment
				// - indent
				if !*newlines && P.cloc.Pos != 0 {
					nlcount = 1;
				}
				P.newline(nlcount);
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
				P.Printf("[%d]", P.cloc.Pos);
			}
			// calling untabify increases the change for idempotent output
			// since tabs in comments are also interpreted by tabwriter
			P.Printf("%s", P.htmlEscape(untabify(ctext)));
		}
		// At this point we may have nlcount > 0: In this case we found newlines
		// that were not followed by a comment. They are recognized (or not) when
		// printing newlines below.
	}

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
	P.newline(P.newlines);
	P.newlines = 0;

	// --------------------------------
	// print string
	if *debug {
		P.Printf("[%d]", pos);
	}
	P.Printf("%s%s%s", tag, P.htmlEscape(s), endtag);

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
	P.opt_semi = false;
	loc.Pos += len(s);  // rough estimate
	loc.Col += len(s);  // rough estimate
	P.lastloc = loc;
}


func (P *Printer) String(loc scanner.Location, s string) {
	P.TaggedString(loc, "", s, "");
}


func (P *Printer) Token(loc scanner.Location, tok int) {
	P.String(loc, token.TokenString(tok));
	//P.TaggedString(pos, "<b>", token.TokenString(tok), "</b>");
}


func (P *Printer) Error(loc scanner.Location, tok int, msg string) {
	fmt.Printf("\ninternal printing error: pos = %d, tok = %s, %s\n", loc.Pos, token.TokenString(tok), msg);
	panic();
}


// ----------------------------------------------------------------------------
// HTML support

func (P *Printer) HtmlIdentifier(x *ast.Ident) {
	P.String(x.Pos_, x.Str);
	/*
	obj := x.Obj;
	if P.html && obj.Kind != symbolTable.NONE {
		// depending on whether we have a declaration or use, generate different html
		// - no need to htmlEscape ident
		id := utils.IntToString(obj.Id, 10);
		if x.Loc_ == obj.Pos {
			// probably the declaration of x
			P.TaggedString(x.Loc_, `<a name="id` + id + `">`, obj.Ident, `</a>`);
		} else {
			// probably not the declaration of x
			P.TaggedString(x.Loc_, `<a href="#id` + id + `">`, obj.Ident, `</a>`);
		}
	} else {
		P.String(x.Loc_, obj.Ident);
	}
	*/
}


func (P *Printer) HtmlPackageName(loc scanner.Location, name string) {
	if P.html {
		sname := name[1 : len(name)-1];  // strip quotes  TODO do this elsewhere eventually
		// TODO CAPITAL HACK BELOW FIX THIS
		P.TaggedString(loc, `"<a href="/src/lib/` + sname + `.go">`, sname, `</a>"`);
	} else {
		P.String(loc, name);
	}
}


// ----------------------------------------------------------------------------
// Support

func (P *Printer) Expr(x ast.Expr)

func (P *Printer) Idents(list []*ast.Ident, full bool) int {
	n := 0;
	for i, x := range list {
		if n > 0 {
			P.Token(noloc, token.COMMA);
			P.separator = blank;
			P.state = inside_list;
		}
		if full || isExported(x) {
			P.Expr(x);
			n++;
		}
	}
	return n;
}


func (P *Printer) Exprs(list []ast.Expr) {
	for i, x := range list {
		if i > 0 {
			P.Token(noloc, token.COMMA);
			P.separator = blank;
			P.state = inside_list;
		}
		P.Expr(x);
	}
}


func (P *Printer) Parameters(list []*ast.Field) {
	P.Token(noloc, token.LPAREN);
	if len(list) > 0 {
		for i, par := range list {
			if i > 0 {
				P.separator = comma;
			}
			n := P.Idents(par.Names, true);
			if n > 0 {
				P.separator = blank
			};
			P.Expr(par.Typ);
		}
	}
	P.Token(noloc, token.RPAREN);
}


// Returns the separator (semicolon or none) required if
// the type is terminating a declaration or statement.
func (P *Printer) Signature(sig *ast.Signature) {
	P.Parameters(sig.Params);
	if sig.Result != nil {
		P.separator = blank;

		if len(sig.Result) == 1 && sig.Result[0].Names == nil {
			// single anonymous result
			// => no parentheses needed unless it's a function type
			fld := sig.Result[0];
			if dummy, is_ftyp := fld.Typ.(*ast.FunctionType); !is_ftyp {
				P.Expr(fld.Typ);
				return;
			}
		}
		
		P.Parameters(sig.Result);
	}
}


func (P *Printer) Fields(list []*ast.Field, end scanner.Location, is_interface bool) {
	P.state = opening_scope;
	P.separator = blank;
	P.Token(noloc, token.LBRACE);

	if len(list) > 0 {
		P.newlines = 1;
		for i, fld := range list {
			if i > 0 {
				P.separator = semicolon;
				P.newlines = 1;
			}
			n := P.Idents(fld.Names, P.full);
			if n > 0 {
				// at least one identifier
				P.separator = tab
			};
			if n > 0 || len(fld.Names) == 0 {
				// at least one identifier or anonymous field
				if is_interface {
					if ftyp, is_ftyp := fld.Typ.(*ast.FunctionType); is_ftyp {
						P.Signature(ftyp.Sig);
					} else {
						P.Expr(fld.Typ);
					}
				} else {
					P.Expr(fld.Typ);
					if fld.Tag != nil {
						P.separator = tab;
						P.Expr(fld.Tag);
					}
				}
			}
		}
		P.newlines = 1;
	}

	P.state = closing_scope;
	P.Token(end, token.RBRACE);
	P.opt_semi = true;
}


// ----------------------------------------------------------------------------
// Expressions

func (P *Printer) Block(b *ast.Block, indent bool)
func (P *Printer) Expr1(x ast.Expr, prec1 int)


func (P *Printer) DoBadExpr(x *ast.BadExpr) {
	P.String(noloc, "BadExpr");
}


func (P *Printer) DoIdent(x *ast.Ident) {
	P.HtmlIdentifier(x);
}


func (P *Printer) DoBinaryExpr(x *ast.BinaryExpr) {
	prec := token.Precedence(x.Op);
	if prec < P.prec {
		P.Token(noloc, token.LPAREN);
	}
	P.Expr1(x.X, prec);
	P.separator = blank;
	P.Token(x.Pos_, x.Op);
	P.separator = blank;
	P.Expr1(x.Y, prec);
	if prec < P.prec {
		P.Token(noloc, token.RPAREN);
	}
}


func (P *Printer) DoUnaryExpr(x *ast.UnaryExpr) {
	prec := token.UnaryPrec;
	if prec < P.prec {
		P.Token(noloc, token.LPAREN);
	}
	P.Token(x.Pos_, x.Op);
	P.Expr1(x.X, prec);
	if prec < P.prec {
		P.Token(noloc, token.RPAREN);
	}
}


func (P *Printer) DoBasicLit(x *ast.BasicLit) {
	// TODO get rid of string conversion here
	P.String(x.Pos_, string(x.Lit));
}


func (P *Printer) DoStringLit(x *ast.StringLit) {
	for i, x := range x.Strings {
		if i > 0 {
			P.separator = blank;
		}
		P.DoBasicLit(x);
	}
}


func (P *Printer) DoFunctionLit(x *ast.FunctionLit) {
	P.Token(x.Func, token.FUNC);
	P.Signature(x.Typ);
	P.separator = blank;
	P.Block(x.Body, true);
	P.newlines = 0;
}


func (P *Printer) DoGroup(x *ast.Group) {
	P.Token(x.Lparen, token.LPAREN);
	P.Expr(x.X);
	P.Token(x.Rparen, token.RPAREN);
}


func (P *Printer) DoSelector(x *ast.Selector) {
	P.Expr1(x.X, token.HighestPrec);
	P.Token(x.Period, token.PERIOD);
	P.Expr1(x.Sel, token.HighestPrec);
}


func (P *Printer) DoTypeAssertion(x *ast.TypeAssertion) {
	P.Expr1(x.X, token.HighestPrec);
	P.Token(x.Period, token.PERIOD);
	P.Token(x.Lparen, token.LPAREN);
	P.Expr(x.Typ);
	P.Token(x.Rparen, token.RPAREN);
}


func (P *Printer) DoIndex(x *ast.Index) {
	P.Expr1(x.X, token.HighestPrec);
	P.Token(x.Lbrack, token.LBRACK);
	P.Expr(x.Index);
	P.Token(x.Rbrack, token.RBRACK);
}


func (P *Printer) DoSlice(x *ast.Slice) {
	P.Expr1(x.X, token.HighestPrec);
	P.Token(x.Lbrack, token.LBRACK);
	P.Expr(x.Beg);
	P.Token(x.Colon, token.COLON);
	P.Expr(x.End);
	P.Token(x.Rbrack, token.RBRACK);
}


func (P *Printer) DoCall(x *ast.Call) {
	P.Expr1(x.Fun, token.HighestPrec);
	P.Token(x.Lparen, token.LPAREN);
	P.Exprs(x.Args);
	P.Token(x.Rparen, token.RPAREN);
}


func (P *Printer) DoCompositeLit(x *ast.CompositeLit) {
	P.Expr1(x.Typ, token.HighestPrec);
	P.Token(x.Lbrace, token.LBRACE);
	P.Exprs(x.Elts);
	P.Token(x.Rbrace, token.RBRACE);
}


func (P *Printer) DoEllipsis(x *ast.Ellipsis) {
	P.Token(x.Loc_, token.ELLIPSIS);
}


func (P *Printer) DoArrayType(x *ast.ArrayType) {
	P.Token(x.Loc_, token.LBRACK);
	if x.Len != nil {
		P.Expr(x.Len);
	}
	P.Token(noloc, token.RBRACK);
	P.Expr(x.Elt);
}


func (P *Printer) DoTypeType(x *ast.TypeType) {
	P.Token(x.Loc_, token.TYPE);
}


func (P *Printer) DoStructType(x *ast.StructType) {
	P.Token(x.Loc_, token.STRUCT);
	if x.End.Pos > 0 {
		P.Fields(x.Fields, x.End, false);
	}
}


func (P *Printer) DoPointerType(x *ast.PointerType) {
	P.Token(x.Loc_, token.MUL);
	P.Expr(x.Base);
}


func (P *Printer) DoFunctionType(x *ast.FunctionType) {
	P.Token(x.Loc_, token.FUNC);
	P.Signature(x.Sig);
}


func (P *Printer) DoInterfaceType(x *ast.InterfaceType) {
	P.Token(x.Loc_, token.INTERFACE);
	if x.End.Pos > 0 {
		P.Fields(x.Methods, x.End, true);
	}
}


func (P *Printer) DoSliceType(x *ast.SliceType) {
	unimplemented();
}


func (P *Printer) DoMapType(x *ast.MapType) {
	P.Token(x.Loc_, token.MAP);
	P.separator = blank;
	P.Token(noloc, token.LBRACK);
	P.Expr(x.Key);
	P.Token(noloc, token.RBRACK);
	P.Expr(x.Val);
}


func (P *Printer) DoChannelType(x *ast.ChannelType) {
	switch x.Dir {
	case ast.FULL:
		P.Token(x.Loc_, token.CHAN);
	case ast.RECV:
		P.Token(x.Loc_, token.ARROW);
		P.Token(noloc, token.CHAN);
	case ast.SEND:
		P.Token(x.Loc_, token.CHAN);
		P.separator = blank;
		P.Token(noloc, token.ARROW);
	}
	P.separator = blank;
	P.Expr(x.Val);
}


func (P *Printer) Expr1(x ast.Expr, prec1 int) {
	if x == nil {
		return;  // empty expression list
	}

	saved_prec := P.prec;
	P.prec = prec1;
	x.Visit(P);
	P.prec = saved_prec;
}


func (P *Printer) Expr(x ast.Expr) {
	P.Expr1(x, token.LowestPrec);
}


// ----------------------------------------------------------------------------
// Statements

func (P *Printer) Stat(s ast.Stat) {
	s.Visit(P);
}


func (P *Printer) StatementList(list *vector.Vector) {
	for i := 0; i < list.Len(); i++ {
		if i == 0 {
			P.newlines = 1;
		} else {  // i > 0
			if !P.opt_semi || *optsemicolons {
				// semicolon is required
				P.separator = semicolon;
			}
		}
		P.Stat(list.At(i).(ast.Stat));
		P.newlines = 1;
		P.state = inside_list;
	}
}


func (P *Printer) Block(b *ast.Block, indent bool) {
	P.state = opening_scope;
	P.Token(b.Loc, b.Tok);
	if !indent {
		P.indentation--;
	}
	P.StatementList(b.List);
	if !indent {
		P.indentation++;
	}
	if !*optsemicolons {
		P.separator = none;
	}
	P.state = closing_scope;
	if b.Tok == token.LBRACE {
		P.Token(b.End, token.RBRACE);
		P.opt_semi = true;
	} else {
		P.String(noloc, "");  // process closing_scope state transition!
	}
}


func (P *Printer) Decl(d ast.Decl);

func (P *Printer) DoBadStat(s *ast.BadStat) {
	panic();
}


func (P *Printer) DoLabeledStat(s *ast.LabeledStat) {
	P.indentation--;
	P.Expr(s.Label);
	P.Token(s.Loc, token.COLON);
	P.indentation++;
	// TODO be more clever if s.Stat is a labeled stat as well
	P.separator = tab;
	P.Stat(s.Stat);
}


func (P *Printer) DoDeclarationStat(s *ast.DeclarationStat) {
	P.Decl(s.Decl);
}


func (P *Printer) DoExpressionStat(s *ast.ExpressionStat) {
	switch s.Tok {
	case token.ILLEGAL:
		P.Expr(s.Expr);
	case token.INC, token.DEC:
		P.Expr(s.Expr);
		P.Token(s.Loc, s.Tok);
	case token.RETURN, token.GO, token.DEFER:
		P.Token(s.Loc, s.Tok);
		if s.Expr != nil {
			P.separator = blank;
			P.Expr(s.Expr);
		}
	default:
		P.Error(s.Loc, s.Tok, "DoExpressionStat");
		unreachable();
	}
}


func (P *Printer) DoAssignmentStat(s *ast.AssignmentStat) {
	P.Expr(s.Lhs);
	P.separator = blank;
	P.Token(s.Loc, s.Tok);
	P.separator = blank;
	P.Expr(s.Rhs);
}


func (P *Printer) DoTupleAssignStat(s *ast.TupleAssignStat) {
	P.Exprs(s.Lhs);
	P.separator = blank;
	P.Token(s.Loc, s.Tok);
	P.separator = blank;
	P.Exprs(s.Rhs);
}


func (P *Printer) DoIncDecStat(s *ast.IncDecStat) {
	P.Expr(s.Expr);
	P.Token(s.Loc, s.Tok);
}


func (P *Printer) DoCompositeStat(s *ast.CompositeStat) {
	P.Block(s.Body, true);
}


func (P *Printer) ControlClause(isForStat bool, init ast.Stat, expr ast.Expr, post ast.Stat) {
	P.separator = blank;
	if init == nil && post == nil {
		// no semicolons required
		if expr != nil {
			P.Expr(expr);
		}
	} else if range_clause, ok := init.(*ast.RangeClause); ok {
		// range clause
		P.Stat(range_clause);
	} else if typeswitch_clause, ok := init.(*ast.TypeSwitchClause); ok {
		// type switch clause
		P.Stat(typeswitch_clause);
	} else {
		// all semicolons required
		// (they are not separators, print them explicitly)
		if init != nil {
			P.Stat(init);
			P.separator = none;
		}
		P.Token(noloc, token.SEMICOLON);
		P.separator = blank;
		if expr != nil {
			P.Expr(expr);
			P.separator = none;
		}
		if isForStat {
			P.Token(noloc, token.SEMICOLON);
			P.separator = blank;
			if post != nil {
				P.Stat(post);
			}
		}
	}
	P.separator = blank;
}


func (P *Printer) DoIfStat(s *ast.IfStat) {
	P.Token(s.Loc, token.IF);
	P.ControlClause(false, s.Init, s.Cond, nil);
	P.Block(s.Body, true);
	if s.Else != nil {
		P.separator = blank;
		P.Token(noloc, token.ELSE);
		P.separator = blank;
		P.Stat(s.Else);
	}
}


func (P *Printer) DoRangeClause(s *ast.RangeClause) {
	P.Exprs(s.Lhs);
	P.separator = blank;
	P.Token(s.Loc, s.Tok);
	P.separator = blank;
	P.Token(noloc, token.RANGE);
	P.separator = blank;
	P.Expr(s.Rhs);
}


func (P *Printer) DoForStat(s *ast.ForStat) {
	P.Token(s.Loc, token.FOR);
	P.ControlClause(true, s.Init, s.Cond, s.Post);
	P.Block(s.Body, true);
}


func (P *Printer) DoTypeSwitchClause(s *ast.TypeSwitchClause) {
	P.Expr(s.Lhs);
	P.separator = blank;
	P.Token(s.Loc, token.DEFINE);
	P.separator = blank;
	P.Expr(s.Rhs);
	P.Token(s.Loc, token.PERIOD);
	P.Token(s.Loc, token.LPAREN);
	P.Token(s.Loc, token.TYPE);
	P.Token(s.Loc, token.RPAREN);
}


func (P *Printer) DoCaseClause(s *ast.CaseClause) {
	if s.Values != nil {
		P.Token(s.Loc, token.CASE);
		P.separator = blank;
		P.Exprs(s.Values);
	} else {
		P.Token(s.Loc, token.DEFAULT);
	}
	// TODO: try to use P.Block instead
	// P.Block(s.Body, true);
	P.Token(s.Body.Loc, token.COLON);
	P.indentation++;
	P.StatementList(s.Body.List);
	P.indentation--;
	P.newlines = 1;
}


func (P *Printer) DoSwitchStat(s *ast.SwitchStat) {
	P.Token(s.Loc, token.SWITCH);
	P.ControlClause(false, s.Init, s.Tag, nil);
	P.Block(s.Body, false);
}


func (P *Printer) DoTypeSwitchStat(s *ast.SwitchStat) {
	P.Token(s.Loc, token.SWITCH);
	P.ControlClause(false, s.Init, s.Tag, nil);
	P.Block(s.Body, false);
}


func (P *Printer) DoCommClause(s *ast.CommClause) {
	if s.Rhs != nil {
		P.Token(s.Loc, token.CASE);
		P.separator = blank;
		if s.Lhs != nil {
			P.Expr(s.Lhs);
			P.separator = blank;
			P.Token(noloc, s.Tok);
			P.separator = blank;
		}
		P.Expr(s.Rhs);
	} else {
		P.Token(s.Loc, token.DEFAULT);
	}
	// TODO: try to use P.Block instead
	// P.Block(s.Body, true);
	P.Token(s.Body.Loc, token.COLON);
	P.indentation++;
	P.StatementList(s.Body.List);
	P.indentation--;
	P.newlines = 1;
}


func (P *Printer) DoSelectStat(s *ast.SelectStat) {
	P.Token(s.Loc, token.SELECT);
	P.separator = blank;
	P.Block(s.Body, false);
}


func (P *Printer) DoControlFlowStat(s *ast.ControlFlowStat) {
	P.Token(s.Loc, s.Tok);
	if s.Label != nil {
		P.separator = blank;
		P.Expr(s.Label);
	}
}


func (P *Printer) DoReturnStat(s *ast.ReturnStat) {
	P.Token(s.Loc, token.RETURN);
	P.separator = blank;
	P.Exprs(s.Results);
}


func (P *Printer) DoEmptyStat(s *ast.EmptyStat) {
	P.String(s.Loc, "");
}


// ----------------------------------------------------------------------------
// Declarations

func (P *Printer) DoBadDecl(d *ast.BadDecl) {
	P.String(d.Loc, "<BAD DECL>");
}


func (P *Printer) DoImportDecl(d *ast.ImportDecl) {
	if d.Loc.Pos > 0 {
		P.Token(d.Loc, token.IMPORT);
		P.separator = blank;
	}
	if d.Name != nil {
		P.Expr(d.Name);
	} else {
		P.String(d.Path.Pos(), "");  // flush pending ';' separator/newlines
	}
	P.separator = tab;
	if lit, is_lit := d.Path.(*ast.StringLit); is_lit {
		// TODO incorrect (string lit could be a list of strings)
		P.HtmlPackageName(lit.Pos(), string(lit.Strings[0].Lit));
	} else {
		// we should only reach here for strange imports
		// import "foo" "bar"
		P.Expr(d.Path);
	}
	P.newlines = 2;
}


func (P *Printer) DoConstDecl(d *ast.ConstDecl) {
	if d.Loc.Pos > 0 {
		P.Token(d.Loc, token.CONST);
		P.separator = blank;
	}
	P.Idents(d.Names, P.full);
	if d.Typ != nil {
		P.separator = blank;  // TODO switch to tab? (indentation problem with structs)
		P.Expr(d.Typ);
	}
	if d.Values != nil {
		P.separator = tab;
		P.Token(noloc, token.ASSIGN);
		P.separator = blank;
		P.Exprs(d.Values);
	}
	P.newlines = 2;
}


func (P *Printer) DoTypeDecl(d *ast.TypeDecl) {
	if d.Loc.Pos > 0 {
		P.Token(d.Loc, token.TYPE);
		P.separator = blank;
	}
	P.Expr(d.Name);
	P.separator = blank;  // TODO switch to tab? (but indentation problem with structs)
	P.Expr(d.Typ);
	P.newlines = 2;
}


func (P *Printer) DoVarDecl(d *ast.VarDecl) {
	if d.Loc.Pos > 0 {
		P.Token(d.Loc, token.VAR);
		P.separator = blank;
	}
	P.Idents(d.Names, P.full);
	if d.Typ != nil {
		P.separator = blank;  // TODO switch to tab? (indentation problem with structs)
		P.Expr(d.Typ);
		//P.separator = P.Type(d.Typ);
	}
	if d.Values != nil {
		P.separator = tab;
		P.Token(noloc, token.ASSIGN);
		P.separator = blank;
		P.Exprs(d.Values);
	}
	P.newlines = 2;
}


func (P *Printer) DoFuncDecl(d *ast.FuncDecl) {
	P.Token(d.Loc, token.FUNC);
	P.separator = blank;
	if recv := d.Recv; recv != nil {
		// method: print receiver
		P.Token(noloc, token.LPAREN);
		if len(recv.Names) > 0 {
			P.Expr(recv.Names[0]);
			P.separator = blank;
		}
		P.Expr(recv.Typ);
		P.Token(noloc, token.RPAREN);
		P.separator = blank;
	}
	P.Expr(d.Name);
	P.Signature(d.Sig);
	if P.full && d.Body != nil {
		P.separator = blank;
		P.Block(d.Body, true);
	}
	P.newlines = 2;
}


func (P *Printer) DoDeclList(d *ast.DeclList) {
	P.Token(d.Loc, d.Tok);
	P.separator = blank;

	// group of parenthesized declarations
	P.state = opening_scope;
	P.Token(noloc, token.LPAREN);
	if len(d.List) > 0 {
		P.newlines = 1;
		for i := 0; i < len(d.List); i++ {
			if i > 0 {
				P.separator = semicolon;
			}
			P.Decl(d.List[i]);
			P.newlines = 1;
		}
	}
	P.state = closing_scope;
	P.Token(d.End, token.RPAREN);
	P.opt_semi = true;
	P.newlines = 2;
}


func (P *Printer) Decl(d ast.Decl) {
	d.Visit(P);
}


// ----------------------------------------------------------------------------
// Package interface

func stripWhiteSpace(s []byte) []byte {
	i, j := 0, len(s);
	for i < len(s) && s[i] <= ' ' {
		i++;
	}
	for j > i && s[j-1] <= ' ' {
		j--
	}
	return s[i : j];
}


func cleanComment(s []byte) []byte {
	switch s[1] {
	case '/': s = s[2 : len(s)-1];
	case '*': s = s[2 : len(s)-2];
	default : panic("illegal comment");
	}
	return stripWhiteSpace(s);
}


func (P *Printer) printComment(comment ast.CommentGroup) {
	in_paragraph := false;
	for i, c := range comment {
		s := cleanComment(c.Text);
		if len(s) > 0 {
			if !in_paragraph {
				P.Printf("<p>\n");
				in_paragraph = true;
			}
			P.Printf("%s\n", P.htmlEscape(untabify(string(s))));
		} else {
			if in_paragraph {
				P.Printf("</p>\n");
				in_paragraph = false;
			}
		}
	}
	if in_paragraph {
		P.Printf("</p>\n");
	}
}


func (P *Printer) Interface(p *ast.Program) {
	P.full = false;
	for i := 0; i < len(p.Decls); i++ {
		switch d := p.Decls[i].(type) {
		case *ast.ConstDecl:
			if hasExportedNames(d.Names) {
				P.Printf("<h2>Constants</h2>\n");
				P.Printf("<p><pre>");
				P.DoConstDecl(d);
				P.String(noloc, "");
				P.Printf("</pre></p>\n");
				if d.Comment != nil {
					P.printComment(d.Comment);
				}
			}

		case *ast.TypeDecl:
			if isExported(d.Name) {
				P.Printf("<h2>type %s</h2>\n", d.Name.Str);
				P.Printf("<p><pre>");
				P.DoTypeDecl(d);
				P.String(noloc, "");
				P.Printf("</pre></p>\n");
				if d.Comment != nil {
					P.printComment(d.Comment);
				}
			}

		case *ast.VarDecl:
			if hasExportedNames(d.Names) {
				P.Printf("<h2>Variables</h2>\n");
				P.Printf("<p><pre>");
				P.DoVarDecl(d);
				P.String(noloc, "");
				P.Printf("</pre></p>\n");
				if d.Comment != nil {
					P.printComment(d.Comment);
				}
			}

		case *ast.FuncDecl:
			if isExported(d.Name) {
				if d.Recv != nil {
					P.Printf("<h3>func (");
					P.Expr(d.Recv.Typ);
					P.Printf(") %s</h3>\n", d.Name.Str);
				} else {
					P.Printf("<h2>func %s</h2>\n", d.Name.Str);
				}
				P.Printf("<p><code>");
				P.DoFuncDecl(d);
				P.String(noloc, "");
				P.Printf("</code></p>\n");
				if d.Comment != nil {
					P.printComment(d.Comment);
				}
			}
			
		case *ast.DeclList:
			
		}
	}
}


// ----------------------------------------------------------------------------
// Program

func (P *Printer) Program(p *ast.Program) {
	P.full = true;
	P.Token(p.Loc, token.PACKAGE);
	P.separator = blank;
	P.Expr(p.Name);
	P.newlines = 1;
	for i := 0; i < len(p.Decls); i++ {
		P.Decl(p.Decls[i]);
	}
	P.newlines = 1;
}


// ----------------------------------------------------------------------------
// External interface

var templ = template.NewTemplateOrDie("template.html");


func Print(writer io.Write, prog *ast.Program, html bool) {
	// setup
	var P Printer;
	padchar := byte(' ');
	if *usetabs {
		padchar = '\t';
	}
	flags := uint(0);
	if html {
		flags |= tabwriter.FilterHTML;
	}
	text := tabwriter.NewWriter(writer, *tabwidth, 1, padchar, flags);
	P.Init(text, nil /* prog.Comments */, html);

	if P.html {
		err := templ.Apply(text, "<!--", template.Substitution {
			"PACKAGE_NAME-->" : func() { P.Printf("%s", prog.Name.Str); },
			"PACKAGE_COMMENT-->": func() { P.printComment(prog.Comment); },
			"PACKAGE_INTERFACE-->" : func() { P.Interface(prog); },
			"PACKAGE_BODY-->" : func() { P.Program(prog); },
		});
		if err != nil {
			panic("print error - exiting");
		}
	} else {
		P.Program(prog);
	}

	P.String(noloc, "");  // flush pending separator/newlines
	err := text.Flush();
	if err != nil {
		panic("print error - exiting");
	}
}
