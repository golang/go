// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements printing of AST nodes; specifically
// expressions, statements, declarations, and files. It uses
// the print functionality implemented in printer.go.

package printer

import (
	"bytes";
	"container/vector";
	"go/ast";
	"go/token";
)


// ----------------------------------------------------------------------------
// Common AST nodes.

// Print as many newlines as necessary (but at least min and and at most
// max newlines) to get to the current line. ws is printed before the first
// line break. If newSection is set, the first line break is printed as
// formfeed. Returns true if any line break was printed; returns false otherwise.
//
// TODO(gri): Reconsider signature (provide position instead of line)
//
func (p *printer) linebreak(line, min, max int, ws whiteSpace, newSection bool) (printedBreak bool) {
	n := line - p.pos.Line;
	switch {
	case n < min: n = min;
	case n > max: n = max;
	}
	if n > 0 {
		p.print(ws);
		if newSection {
			p.print(formfeed);
			n--;
			printedBreak = true;
		}
	}
	for ; n > 0; n-- {
		p.print(newline);
		printedBreak = true;
	}
	return;
}


// TODO(gri): The code for printing lead and line comments
//            should be eliminated in favor of reusing the
//            comment intersperse mechanism above somehow.

// Print a list of individual comments.
func (p *printer) commentList(list []*ast.Comment) {
	for i, c := range list {
		t := c.Text;
		// TODO(gri): this needs to be styled like normal comments
		p.print(c.Pos(), t);
		if t[1] == '/' && i+1 < len(list) {
			//-style comment which is not at the end; print a newline
			p.print(newline);
		}
	}
}


// Print a lead comment followed by a newline.
func (p *printer) leadComment(d *ast.CommentGroup) {
	// Ignore the comment if we have comments interspersed (p.comment != nil).
	if p.comment == nil && d != nil {
		p.commentList(d.List);
		p.print(newline);
	}
}


// Print a tab followed by a line comment.
// A newline must be printed afterwards since
// the comment may be a //-style comment.
func (p *printer) lineComment(d *ast.CommentGroup) {
	// Ignore the comment if we have comments interspersed (p.comment != nil).
	if p.comment == nil && d != nil {
		p.print(vtab);
		p.commentList(d.List);
	}
}


// Sets multiLine to true if the identifier list spans multiple lines.
func (p *printer) identList(list []*ast.Ident, multiLine *bool) {
	// convert into an expression list
	xlist := make([]ast.Expr, len(list));
	for i, x := range list {
		xlist[i] = x;
	}
	p.exprList(noPos, xlist, commaSep, multiLine);
}


// Sets multiLine to true if the string list spans multiple lines.
func (p *printer) stringList(list []*ast.BasicLit, multiLine *bool) {
	// convert into an expression list
	xlist := make([]ast.Expr, len(list));
	for i, x := range list {
		xlist[i] = x;
	}
	p.exprList(noPos, xlist, noIndent, multiLine);
}


type exprListMode uint;
const (
	blankStart exprListMode = 1 << iota;  // print a blank before the list
	commaSep;  // elements are separated by commas
	commaTerm;  // elements are terminated by comma
	noIndent;  // no extra indentation in multi-line lists
)


// Print a list of expressions. If the list spans multiple
// source lines, the original line breaks are respected between
// expressions. Sets multiLine to true if the list spans multiple
// lines.
func (p *printer) exprList(prev token.Position, list []ast.Expr, mode exprListMode, multiLine *bool) {
	if len(list) == 0 {
		return;
	}

	if mode & blankStart != 0 {
		p.print(blank);
	}

	// TODO(gri): endLine may be incorrect as it is really the beginning
	//            of the last list entry. There may be only one, very long
	//            entry in which case line == endLine.
	line := list[0].Pos().Line;
	endLine := list[len(list)-1].Pos().Line;

	if prev.IsValid() && prev.Line == line && line == endLine {
		// all list entries on a single line
		for i, x := range list {
			if i > 0 {
				if mode & commaSep != 0 {
					p.print(token.COMMA);
				}
				p.print(blank);
			}
			p.expr(x, multiLine);
		}
		return;
	}

	// list entries span multiple lines;
	// use source code positions to guide line breaks

	// don't add extra indentation if noIndent is set;
	// i.e., pretend that the first line is already indented
	ws := ignore;
	if mode&noIndent == 0 {
		ws = indent;
	}

	if prev.IsValid() && prev.Line < line && p.linebreak(line, 1, 2, ws, true) {
		ws = ignore;
		*multiLine = true;
	}

	for i, x := range list {
		prev := line;
		line = x.Pos().Line;
		if i > 0 {
			if mode & commaSep != 0 {
				p.print(token.COMMA);
			}
			if prev < line {
				if p.linebreak(line, 1, 2, ws, true) {
					ws = ignore;
					*multiLine = true;
				}
			} else {
				p.print(blank);
			}
		}
		p.expr(x, multiLine);
	}
	if mode & commaTerm != 0 {
		p.print(token.COMMA);
		if ws == ignore && mode&noIndent == 0 {
			// should always be indented here since we have a multi-line
			// expression list - be conservative and check anyway
			p.print(unindent);
		}
		p.print(formfeed);  // terminating comma needs a line break to look good
	} else if ws == ignore && mode&noIndent == 0 {
		p.print(unindent);
	}
}


// Sets multiLine to true if the the parameter list spans multiple lines.
func (p *printer) parameters(list []*ast.Field, multiLine *bool) {
	p.print(token.LPAREN);
	if len(list) > 0 {
		for i, par := range list {
			if i > 0 {
				p.print(token.COMMA, blank);
			}
			if len(par.Names) > 0 {
				p.identList(par.Names, multiLine);
				p.print(blank);
			}
			p.expr(par.Type, multiLine);
		}
	}
	p.print(token.RPAREN);
}


// Returns true if a separating semicolon is optional.
// Sets multiLine to true if the signature spans multiple lines.
func (p *printer) signature(params, result []*ast.Field, multiLine *bool) (optSemi bool) {
	p.parameters(params, multiLine);
	if result != nil {
		p.print(blank);

		if len(result) == 1 && result[0].Names == nil {
			// single anonymous result; no ()'s unless it's a function type
			f := result[0];
			if _, isFtyp := f.Type.(*ast.FuncType); !isFtyp {
				optSemi = p.expr(f.Type, multiLine);
				return;
			}
		}

		p.parameters(result, multiLine);
	}
	return;
}


func (p *printer) fieldList(lbrace token.Position, list []*ast.Field, rbrace token.Position, isIncomplete, isStruct bool) {
	if len(list) == 0 && !isIncomplete && !p.commentBefore(rbrace) {
		// no blank between keyword and {} in this case
		p.print(lbrace, token.LBRACE, rbrace, token.RBRACE);
		return;
	}

	// at least one entry or incomplete
	p.print(blank, lbrace, token.LBRACE, indent, formfeed);
	if isStruct {

		sep := vtab;
		if len(list) == 1 {
			sep = blank;
		}
		var ml bool;
		for i, f := range list {
			if i > 0 {
				p.linebreak(f.Pos().Line, 1, 2, ignore, ml);
			}
			ml = false;
			extraTabs := 0;
			p.leadComment(f.Doc);
			if len(f.Names) > 0 {
				p.identList(f.Names, &ml);
				p.print(sep);
				p.expr(f.Type, &ml);
				extraTabs = 1;
			} else {
				p.expr(f.Type, &ml);
				extraTabs = 2;
			}
			if f.Tag != nil {
				if len(f.Names) > 0 && sep == vtab {
					p.print(sep);
				}
				p.print(sep);
				p.expr(&ast.StringList{f.Tag}, &ml);
				extraTabs = 0;
			}
			p.print(token.SEMICOLON);
			if f.Comment != nil {
				for ; extraTabs > 0; extraTabs-- {
					p.print(vtab);
				}
				p.lineComment(f.Comment);
			}
		}
		if isIncomplete {
			if len(list) > 0 {
				p.print(formfeed);
			}
			// TODO(gri): this needs to be styled like normal comments
			p.print("// contains unexported fields");
		}

	} else { // interface

		var ml bool;
		for i, f := range list {
			if i > 0 {
				p.linebreak(f.Pos().Line, 1, 2, ignore, ml);
			}
			ml = false;
			p.leadComment(f.Doc);
			if ftyp, isFtyp := f.Type.(*ast.FuncType); isFtyp {
				// method
				p.expr(f.Names[0], &ml);
				p.signature(ftyp.Params, ftyp.Results, &ml);
			} else {
				// embedded interface
				p.expr(f.Type, &ml);
			}
			p.print(token.SEMICOLON);
			p.lineComment(f.Comment);
		}
		if isIncomplete {
			if len(list) > 0 {
				p.print(formfeed);
			}
			// TODO(gri): this needs to be styled like normal comments
			p.print("// contains unexported methods");
		}

	}
	p.print(unindent, formfeed, rbrace, token.RBRACE);
}


// ----------------------------------------------------------------------------
// Expressions

func needsBlanks(expr ast.Expr) bool {
	switch x := expr.(type) {
	case *ast.Ident:
		// "long" identifiers look better with blanks around them
		return len(x.Value) > 8;
	case *ast.BasicLit:
		// "long" literals look better with blanks around them
		return len(x.Value) > 8;
	case *ast.ParenExpr:
		// parenthesized expressions don't need blanks around them
		return false;
	case *ast.IndexExpr:
		// index expressions don't need blanks if the indexed expressions are simple
		return needsBlanks(x.X)
	case *ast.CallExpr:
		// call expressions need blanks if they have more than one
		// argument or if the function expression needs blanks
		return len(x.Args) > 1 || needsBlanks(x.Fun);
	}
	return true;
}


// Sets multiLine to true if the binary expression spans multiple lines.
func (p *printer) binaryExpr(x *ast.BinaryExpr, prec1 int, multiLine *bool) {
	prec := x.Op.Precedence();
	if prec < prec1 {
		// parenthesis needed
		// Note: The parser inserts an ast.ParenExpr node; thus this case
		//       can only occur if the AST is created in a different way.
		p.print(token.LPAREN);
		p.expr(x, multiLine);
		p.print(token.RPAREN);
		return;
	}

	// Traverse left, collect all operations at same precedence
	// and determine if blanks should be printed around operators.
	//
	// This algorithm assumes that the right-hand side of a binary
	// operation has a different (higher) precedence then the current
	// node, which is how the parser creates the AST.
	var list vector.Vector;
	line := x.Y.Pos().Line;
	printBlanks := prec <= token.EQL.Precedence() || needsBlanks(x.Y);
	for {
		list.Push(x);
		if t, ok := x.X.(*ast.BinaryExpr); ok && t.Op.Precedence() == prec {
			x = t;
			prev := line;
			line = x.Y.Pos().Line;
			if needsBlanks(x.Y) || prev != line {
				printBlanks = true;
			}
		} else {
			break;
		}
	}
	prev := line;
	line = x.X.Pos().Line;
	if needsBlanks(x.X) || prev != line {
		printBlanks = true;
	}

	// Print collected operations left-to-right, with blanks if necessary.
	ws := indent;
	p.expr1(x.X, prec, multiLine);
	for list.Len() > 0 {
		x = list.Pop().(*ast.BinaryExpr);
		prev := line;
		line = x.Y.Pos().Line;
		if printBlanks {
			if prev != line {
				p.print(blank, x.OpPos, x.Op);
				// at least one line break, but respect an extra empty line
				// in the source
				if p.linebreak(line, 1, 2, ws, true) {
					ws = ignore;
					*multiLine = true;
				}
			} else {
				p.print(blank, x.OpPos, x.Op, blank);
			}
		} else {
			if prev != line {
				panic("internal error");
			}
			p.print(x.OpPos, x.Op);
		}
		p.expr1(x.Y, prec, multiLine);
	}
	if ws == ignore {
		p.print(unindent);
	}
}


// Returns true if a separating semicolon is optional.
// Sets multiLine to true if the expression spans multiple lines.
func (p *printer) expr1(expr ast.Expr, prec1 int, multiLine *bool) (optSemi bool) {
	p.print(expr.Pos());

	switch x := expr.(type) {
	case *ast.BadExpr:
		p.print("BadExpr");

	case *ast.Ident:
		p.print(x);

	case *ast.BinaryExpr:
		p.binaryExpr(x, prec1, multiLine);

	case *ast.KeyValueExpr:
		p.expr(x.Key, multiLine);
		p.print(x.Colon, token.COLON, blank);
		p.expr(x.Value, multiLine);

	case *ast.StarExpr:
		p.print(token.MUL);
		optSemi = p.expr(x.X, multiLine);

	case *ast.UnaryExpr:
		const prec = token.UnaryPrec;
		if prec < prec1 {
			// parenthesis needed
			p.print(token.LPAREN);
			p.expr(x, multiLine);
			p.print(token.RPAREN);
		} else {
			// no parenthesis needed
			p.print(x.Op);
			if x.Op == token.RANGE {
				p.print(blank);
			}
			p.expr1(x.X, prec, multiLine);
		}

	case *ast.BasicLit:
		p.print(x);

	case *ast.StringList:
		p.stringList(x.Strings, multiLine);

	case *ast.FuncLit:
		p.expr(x.Type, multiLine);
		p.funcBody(x.Body, true, multiLine);

	case *ast.ParenExpr:
		p.print(token.LPAREN);
		p.expr(x.X, multiLine);
		p.print(x.Rparen, token.RPAREN);

	case *ast.SelectorExpr:
		p.expr1(x.X, token.HighestPrec, multiLine);
		p.print(token.PERIOD);
		p.expr1(x.Sel, token.HighestPrec, multiLine);

	case *ast.TypeAssertExpr:
		p.expr1(x.X, token.HighestPrec, multiLine);
		p.print(token.PERIOD, token.LPAREN);
		if x.Type != nil {
			p.expr(x.Type, multiLine);
		} else {
			p.print(token.TYPE);
		}
		p.print(token.RPAREN);

	case *ast.IndexExpr:
		p.expr1(x.X, token.HighestPrec, multiLine);
		p.print(token.LBRACK);
		p.expr1(x.Index, token.LowestPrec, multiLine);
		if x.End != nil {
			if needsBlanks(x.Index) || needsBlanks(x.End) {
				// blanks around ":"
				p.print(blank, token.COLON, blank);
			} else {
				// no blanks around ":"
				p.print(token.COLON);
			}
			p.expr(x.End, multiLine);
		}
		p.print(token.RBRACK);

	case *ast.CallExpr:
		p.expr1(x.Fun, token.HighestPrec, multiLine);
		p.print(x.Lparen, token.LPAREN);
		p.exprList(x.Lparen, x.Args, commaSep, multiLine);
		p.print(x.Rparen, token.RPAREN);

	case *ast.CompositeLit:
		p.expr1(x.Type, token.HighestPrec, multiLine);
		p.print(x.Lbrace, token.LBRACE);
		p.exprList(x.Lbrace, x.Elts, commaSep|commaTerm, multiLine);
		p.print(x.Rbrace, token.RBRACE);

	case *ast.Ellipsis:
		p.print(token.ELLIPSIS);

	case *ast.ArrayType:
		p.print(token.LBRACK);
		if x.Len != nil {
			p.expr(x.Len, multiLine);
		}
		p.print(token.RBRACK);
		optSemi = p.expr(x.Elt, multiLine);

	case *ast.StructType:
		p.print(token.STRUCT);
		p.fieldList(x.Lbrace, x.Fields, x.Rbrace, x.Incomplete, true);
		optSemi = true;

	case *ast.FuncType:
		p.print(token.FUNC);
		optSemi = p.signature(x.Params, x.Results, multiLine);

	case *ast.InterfaceType:
		p.print(token.INTERFACE);
		p.fieldList(x.Lbrace, x.Methods, x.Rbrace, x.Incomplete, false);
		optSemi = true;

	case *ast.MapType:
		p.print(token.MAP, token.LBRACK);
		p.expr(x.Key, multiLine);
		p.print(token.RBRACK);
		optSemi = p.expr(x.Value, multiLine);

	case *ast.ChanType:
		switch x.Dir {
		case ast.SEND | ast.RECV:
			p.print(token.CHAN);
		case ast.RECV:
			p.print(token.ARROW, token.CHAN);
		case ast.SEND:
			p.print(token.CHAN, token.ARROW);
		}
		p.print(blank);
		optSemi = p.expr(x.Value, multiLine);

	default:
		panic("unreachable");
	}

	return;
}


// Returns true if a separating semicolon is optional.
// Sets multiLine to true if the expression spans multiple lines.
func (p *printer) expr(x ast.Expr, multiLine *bool) (optSemi bool) {
	return p.expr1(x, token.LowestPrec, multiLine);
}


// ----------------------------------------------------------------------------
// Statements

const maxStmtNewlines = 2  // maximum number of newlines between statements

// Print the statement list indented, but without a newline after the last statement.
// Extra line breaks between statements in the source are respected but at most one
// empty line is printed between statements.
func (p *printer) stmtList(list []ast.Stmt, _indent int) {
	// TODO(gri): fix _indent code
	if _indent > 0 {
		p.print(indent);
	}
	var multiLine bool;
	for i, s := range list {
		// _indent == 0 only for lists of switch/select case clauses;
		// in those cases each clause is a new section
		p.linebreak(s.Pos().Line, 1, maxStmtNewlines, ignore, i == 0 || _indent == 0 || multiLine);
		multiLine = false;
		if !p.stmt(s, &multiLine) {
			p.print(token.SEMICOLON);
		}
	}
	if _indent > 0 {
		p.print(unindent);
	}
}


// block prints an *ast.BlockStmt; it always spans at least two lines.
func (p *printer) block(s *ast.BlockStmt, indent int) {
	p.print(s.Pos(), token.LBRACE);
	p.stmtList(s.List, indent);
	p.linebreak(s.Rbrace.Line, 1, maxStmtNewlines, ignore, true);
	p.print(s.Rbrace, token.RBRACE);
}


// TODO(gri): Decide if this should be used more broadly. The printing code
//            knows when to insert parentheses for precedence reasons, but
//            need to be careful to keep them around type expressions.
func stripParens(x ast.Expr) ast.Expr {
	if px, hasParens := x.(*ast.ParenExpr); hasParens {
		return stripParens(px.X);
	}
	return x;
}


func (p *printer) controlClause(isForStmt bool, init ast.Stmt, expr ast.Expr, post ast.Stmt) {
	p.print(blank);
	needsBlank := false;
	if init == nil && post == nil {
		// no semicolons required
		if expr != nil {
			p.expr(stripParens(expr), ignoreMultiLine);
			needsBlank = true;
		}
	} else {
		// all semicolons required
		// (they are not separators, print them explicitly)
		if init != nil {
			p.stmt(init, ignoreMultiLine);
		}
		p.print(token.SEMICOLON, blank);
		if expr != nil {
			p.expr(stripParens(expr), ignoreMultiLine);
			needsBlank = true;
		}
		if isForStmt {
			p.print(token.SEMICOLON, blank);
			needsBlank = false;
			if post != nil {
				p.stmt(post, ignoreMultiLine);
				needsBlank = true;
			}
		}
	}
	if needsBlank {
		p.print(blank);
	}
}


// Returns true if a separating semicolon is optional.
// Sets multiLine to true if the statements spans multiple lines.
func (p *printer) stmt(stmt ast.Stmt, multiLine *bool) (optSemi bool) {
	p.print(stmt.Pos());

	switch s := stmt.(type) {
	case *ast.BadStmt:
		p.print("BadStmt");

	case *ast.DeclStmt:
		p.decl(s.Decl, inStmtList, multiLine);
		optSemi = true;  // decl prints terminating semicolon if necessary

	case *ast.EmptyStmt:
		// nothing to do

	case *ast.LabeledStmt:
		// a "correcting" unindent immediately following a line break
		// is applied before the line break  if there is no comment
		// between (see writeWhitespace)
		p.print(unindent);
		p.expr(s.Label, multiLine);
		p.print(token.COLON, vtab, indent);
		p.linebreak(s.Stmt.Pos().Line, 0, 1, ignore, true);
		optSemi = p.stmt(s.Stmt, multiLine);

	case *ast.ExprStmt:
		p.expr(s.X, multiLine);

	case *ast.IncDecStmt:
		p.expr(s.X, multiLine);
		p.print(s.Tok);

	case *ast.AssignStmt:
		p.exprList(s.Pos(), s.Lhs, commaSep, multiLine);
		p.print(blank, s.TokPos, s.Tok);
		p.exprList(s.TokPos, s.Rhs, blankStart | commaSep, multiLine);

	case *ast.GoStmt:
		p.print(token.GO, blank);
		p.expr(s.Call, multiLine);

	case *ast.DeferStmt:
		p.print(token.DEFER, blank);
		p.expr(s.Call, multiLine);

	case *ast.ReturnStmt:
		p.print(token.RETURN);
		if s.Results != nil {
			p.exprList(s.Pos(), s.Results, blankStart | commaSep, multiLine);
		}

	case *ast.BranchStmt:
		p.print(s.Tok);
		if s.Label != nil {
			p.print(blank);
			p.expr(s.Label, multiLine);
		}

	case *ast.BlockStmt:
		p.block(s, 1);
		*multiLine = true;
		optSemi = true;

	case *ast.IfStmt:
		p.print(token.IF);
		p.controlClause(false, s.Init, s.Cond, nil);
		p.block(s.Body, 1);
		*multiLine = true;
		optSemi = true;
		if s.Else != nil {
			p.print(blank, token.ELSE, blank);
			switch s.Else.(type) {
			case *ast.BlockStmt, *ast.IfStmt:
				optSemi = p.stmt(s.Else, ignoreMultiLine);
			default:
				p.print(token.LBRACE, indent, formfeed);
				p.stmt(s.Else, ignoreMultiLine);
				p.print(unindent, formfeed, token.RBRACE);
			}
		}

	case *ast.CaseClause:
		if s.Values != nil {
			p.print(token.CASE);
			p.exprList(s.Pos(), s.Values, blankStart | commaSep, multiLine);
		} else {
			p.print(token.DEFAULT);
		}
		p.print(s.Colon, token.COLON);
		p.stmtList(s.Body, 1);
		optSemi = true;  // "block" without {}'s

	case *ast.SwitchStmt:
		p.print(token.SWITCH);
		p.controlClause(false, s.Init, s.Tag, nil);
		p.block(s.Body, 0);
		*multiLine = true;
		optSemi = true;

	case *ast.TypeCaseClause:
		if s.Types != nil {
			p.print(token.CASE);
			p.exprList(s.Pos(), s.Types, blankStart | commaSep, multiLine);
		} else {
			p.print(token.DEFAULT);
		}
		p.print(s.Colon, token.COLON);
		p.stmtList(s.Body, 1);
		optSemi = true;  // "block" without {}'s

	case *ast.TypeSwitchStmt:
		p.print(token.SWITCH);
		if s.Init != nil {
			p.print(blank);
			p.stmt(s.Init, ignoreMultiLine);
			p.print(token.SEMICOLON);
		}
		p.print(blank);
		p.stmt(s.Assign, ignoreMultiLine);
		p.print(blank);
		p.block(s.Body, 0);
		*multiLine = true;
		optSemi = true;

	case *ast.CommClause:
		if s.Rhs != nil {
			p.print(token.CASE, blank);
			if s.Lhs != nil {
				p.expr(s.Lhs, multiLine);
				p.print(blank, s.Tok, blank);
			}
			p.expr(s.Rhs, multiLine);
		} else {
			p.print(token.DEFAULT);
		}
		p.print(s.Colon, token.COLON);
		p.stmtList(s.Body, 1);
		optSemi = true;  // "block" without {}'s

	case *ast.SelectStmt:
		p.print(token.SELECT, blank);
		p.block(s.Body, 0);
		*multiLine = true;
		optSemi = true;

	case *ast.ForStmt:
		p.print(token.FOR);
		p.controlClause(true, s.Init, s.Cond, s.Post);
		p.block(s.Body, 1);
		*multiLine = true;
		optSemi = true;

	case *ast.RangeStmt:
		p.print(token.FOR, blank);
		p.expr(s.Key, multiLine);
		if s.Value != nil {
			p.print(token.COMMA, blank);
			p.expr(s.Value, multiLine);
		}
		p.print(blank, s.TokPos, s.Tok, blank, token.RANGE, blank);
		p.expr(s.X, multiLine);
		p.print(blank);
		p.block(s.Body, 1);
		*multiLine = true;
		optSemi = true;

	default:
		panic("unreachable");
	}

	return;
}


// ----------------------------------------------------------------------------
// Declarations

type declContext uint;
const (
	atTop declContext = iota;
	inGroup;
	inStmtList;
)

// The parameter n is the number of specs in the group; context specifies
// the surroundings of the declaration. Separating semicolons are printed
// depending on the context. Sets multiLine to true if the spec spans
// multiple lines.
//
func (p *printer) spec(spec ast.Spec, n int, context declContext, multiLine *bool) {
	var (
		optSemi bool;  // true if a semicolon is optional
		comment *ast.CommentGroup;  // a line comment, if any
		extraTabs int;  // number of extra tabs before comment, if any
	)

	switch s := spec.(type) {
	case *ast.ImportSpec:
		p.leadComment(s.Doc);
		if s.Name != nil {
			p.expr(s.Name, multiLine);
			p.print(blank);
		}
		p.expr(&ast.StringList{s.Path}, multiLine);
		comment = s.Comment;

	case *ast.ValueSpec:
		p.leadComment(s.Doc);
		p.identList(s.Names, multiLine);  // always present
		if n == 1 {
			if s.Type != nil {
				p.print(blank);
				optSemi = p.expr(s.Type, multiLine);
			}
			if s.Values != nil {
				p.print(blank, token.ASSIGN);
				p.exprList(noPos, s.Values, blankStart | commaSep, multiLine);
				optSemi = false;
			}
		} else {
			extraTabs = 2;
			if s.Type != nil || s.Values != nil {
				p.print(vtab);
			}
			if s.Type != nil {
				optSemi = p.expr(s.Type, multiLine);
				extraTabs = 1;
			}
			if s.Values != nil {
				p.print(vtab);
				p.print(token.ASSIGN);
				p.exprList(noPos, s.Values, blankStart | commaSep, multiLine);
				optSemi = false;
				extraTabs = 0;
			}
		}
		comment = s.Comment;

	case *ast.TypeSpec:
		p.leadComment(s.Doc);
		p.expr(s.Name, multiLine);
		if n == 1 {
			p.print(blank);
		} else {
			p.print(vtab);
		}
		optSemi = p.expr(s.Type, multiLine);
		comment = s.Comment;

	default:
		panic("unreachable");
	}

	if context == inGroup || context == inStmtList && !optSemi {
		p.print(token.SEMICOLON);
	}

	if comment != nil {
		for ; extraTabs > 0; extraTabs-- {
			p.print(vtab);
		}
		p.lineComment(comment);
	}
}


// Sets multiLine to true if the declaration spans multiple lines.
func (p *printer) genDecl(d *ast.GenDecl, context declContext, multiLine *bool) {
	p.leadComment(d.Doc);
	p.print(d.Pos(), d.Tok, blank);

	if d.Lparen.IsValid() {
		// group of parenthesized declarations
		p.print(d.Lparen, token.LPAREN);
		if len(d.Specs) > 0 {
			p.print(indent, formfeed);
			var ml bool;
			for i, s := range d.Specs {
				if i > 0 {
					p.linebreak(s.Pos().Line, 1, 2, ignore, ml);
				}
				ml = false;
				p.spec(s, len(d.Specs), inGroup, &ml);
			}
			p.print(unindent, formfeed);
			*multiLine = true;
		}
		p.print(d.Rparen, token.RPAREN);

	} else {
		// single declaration
		p.spec(d.Specs[0], 1, context, multiLine);
	}
}


func (p *printer) isOneLiner(b *ast.BlockStmt) bool {
	switch {
	case len(b.List) > 1 || p.commentBefore(b.Rbrace):
		return false;  // too many statements or there is a comment - all bets are off
	case len(b.List) == 0:
		return true;  // empty block and no comments
	}

	// test-print the statement and see if it would fit
	var buf bytes.Buffer;
	_, err := p.Config.Fprint(&buf, b.List[0]);
	if err != nil {
		return false;  // don't try
	}

	if buf.Len() > 40 {
		return false;  // too long
	}

	for _, ch := range buf.Bytes() {
		if ch < ' ' {
			return false;  // contains control chars (tabs, newlines)
		}
	}

	return true;
}


// Sets multiLine to true if the function body spans multiple lines.
func (p *printer) funcBody(b *ast.BlockStmt, isLit bool, multiLine *bool) {
	if b == nil {
		return;
	}

	// TODO(gri): enable for function declarations, eventually.
	if isLit && p.isOneLiner(b) {
		sep := vtab;
		if isLit {
			sep = blank;
		}
		if len(b.List) > 0 {
			p.print(sep, b.Pos(), token.LBRACE, blank);
			p.stmt(b.List[0], ignoreMultiLine);
			p.print(blank, b.Rbrace, token.RBRACE);
		} else {
			p.print(sep, b.Pos(), token.LBRACE, b.Rbrace, token.RBRACE);
		}
		return;
	}

	p.print(blank);
	p.block(b, 1);
	*multiLine = true;
}


// Sets multiLine to true if the declaration spans multiple lines.
func (p *printer) funcDecl(d *ast.FuncDecl, multiLine *bool) {
	p.leadComment(d.Doc);
	p.print(d.Pos(), token.FUNC, blank);
	if recv := d.Recv; recv != nil {
		// method: print receiver
		p.print(token.LPAREN);
		if len(recv.Names) > 0 {
			p.expr(recv.Names[0], multiLine);
			p.print(blank);
		}
		p.expr(recv.Type, multiLine);
		p.print(token.RPAREN, blank);
	}
	p.expr(d.Name, multiLine);
	p.signature(d.Type.Params, d.Type.Results, multiLine);
	p.funcBody(d.Body, false, multiLine);
}


// Sets multiLine to true if the declaration spans multiple lines.
func (p *printer) decl(decl ast.Decl, context declContext, multiLine *bool) {
	switch d := decl.(type) {
	case *ast.BadDecl:
		p.print(d.Pos(), "BadDecl");
	case *ast.GenDecl:
		p.genDecl(d, context, multiLine);
	case *ast.FuncDecl:
		p.funcDecl(d, multiLine);
	default:
		panic("unreachable");
	}
}


// ----------------------------------------------------------------------------
// Files

const maxDeclNewlines = 3  // maximum number of newlines between declarations

func declToken(decl ast.Decl) (tok token.Token) {
	tok = token.ILLEGAL;
	switch d := decl.(type) {
	case *ast.GenDecl:
		tok = d.Tok;
	case *ast.FuncDecl:
		tok = token.FUNC;
	}
	return;
}


func (p *printer) file(src *ast.File) {
	p.leadComment(src.Doc);
	p.print(src.Pos(), token.PACKAGE, blank);
	p.expr(src.Name, ignoreMultiLine);

	if len(src.Decls) > 0 {
		tok := token.ILLEGAL;
		for _, d := range src.Decls {
			prev := tok;
			tok = declToken(d);
			// if the declaration token changed (e.g., from CONST to TYPE)
			// print an empty line between top-level declarations
			min := 1;
			if prev != tok {
				min = 2;
			}
			p.linebreak(d.Pos().Line, min, maxDeclNewlines, ignore, false);
			p.decl(d, atTop, ignoreMultiLine);
		}
	}

	p.print(newline);
}
