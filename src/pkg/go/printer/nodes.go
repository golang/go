// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements printing of AST nodes; specifically
// expressions, statements, declarations, and files. It uses
// the print functionality implemented in printer.go.

package printer

import (
	"bytes"
	"go/ast"
	"go/token"
)


// Other formatting issues:
// - better comment formatting for /*-style comments at the end of a line (e.g. a declaration)
//   when the comment spans multiple lines; if such a comment is just two lines, formatting is
//   not idempotent
// - formatting of expression lists
// - should use blank instead of tab to separate one-line function bodies from
//   the function header unless there is a group of consecutive one-liners


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
	n := line - p.pos.Line
	switch {
	case n < min:
		n = min
	case n > max:
		n = max
	}

	if n > 0 {
		p.print(ws)
		if newSection {
			p.print(formfeed)
			n--
			printedBreak = true
		}
	}
	for ; n > 0; n-- {
		p.print(newline)
		printedBreak = true
	}
	return
}


// setComment sets g as the next comment if g != nil and if node comments
// are enabled - this mode is used when printing source code fragments such
// as exports only. It assumes that there are no other pending comments to
// intersperse.
func (p *printer) setComment(g *ast.CommentGroup) {
	if g == nil || !p.useNodeComments {
		return
	}
	if p.comments == nil {
		// initialize p.comments lazily
		p.comments = make([]*ast.CommentGroup, 1)
	} else if p.cindex < len(p.comments) {
		// for some reason there are pending comments; this
		// should never happen - handle gracefully and flush
		// all comments up to g, ignore anything after that
		p.flush(g.List[0].Pos(), false)
	}
	p.comments[0] = g
	p.cindex = 0
}


// Sets multiLine to true if the identifier list spans multiple lines.
func (p *printer) identList(list []*ast.Ident, multiLine *bool) {
	// convert into an expression list so we can re-use exprList formatting
	xlist := make([]ast.Expr, len(list))
	for i, x := range list {
		xlist[i] = x
	}
	p.exprList(noPos, xlist, 1, commaSep, multiLine, noPos)
}


type exprListMode uint

const (
	blankStart exprListMode = 1 << iota // print a blank before a non-empty list
	blankEnd                // print a blank after a non-empty list
	commaSep                // elements are separated by commas
	commaTerm               // list is optionally terminated by a comma
	noIndent                // no extra indentation in multi-line lists
)


// isOneLineExpr returns true if x is "small enough" to fit onto a single line.
func (p *printer) isOneLineExpr(x ast.Expr) bool {
	const maxSize = 60 // aproximate value, excluding space for comments
	return p.nodeSize(x, maxSize) <= maxSize
}


// Print a list of expressions. If the list spans multiple
// source lines, the original line breaks are respected between
// expressions. Sets multiLine to true if the list spans multiple
// lines.
func (p *printer) exprList(prev token.Position, list []ast.Expr, depth int, mode exprListMode, multiLine *bool, next token.Position) {
	if len(list) == 0 {
		return
	}

	if mode&blankStart != 0 {
		p.print(blank)
	}

	line := list[0].Pos().Line
	endLine := next.Line
	if endLine == 0 {
		// TODO(gri): endLine may be incorrect as it is really the beginning
		//            of the last list entry. There may be only one, very long
		//            entry in which case line == endLine.
		endLine = list[len(list)-1].Pos().Line
	}

	if prev.IsValid() && prev.Line == line && line == endLine {
		// all list entries on a single line
		for i, x := range list {
			if i > 0 {
				if mode&commaSep != 0 {
					p.print(token.COMMA)
				}
				p.print(blank)
			}
			p.expr0(x, depth, multiLine)
		}
		if mode&blankEnd != 0 {
			p.print(blank)
		}
		return
	}

	// list entries span multiple lines;
	// use source code positions to guide line breaks

	// don't add extra indentation if noIndent is set;
	// i.e., pretend that the first line is already indented
	ws := ignore
	if mode&noIndent == 0 {
		ws = indent
	}

	oneLiner := false // true if the previous expression fit on a single line
	prevBreak := -1   // index of last expression that was followed by a linebreak

	// the first linebreak is always a formfeed since this section must not
	// depend on any previous formatting
	if prev.IsValid() && prev.Line < line && p.linebreak(line, 1, 2, ws, true) {
		ws = ignore
		*multiLine = true
		prevBreak = 0
	}

	for i, x := range list {
		prev := line
		line = x.Pos().Line
		if i > 0 {
			if mode&commaSep != 0 {
				p.print(token.COMMA)
			}
			if prev < line && prev > 0 && line > 0 {
				// lines are broken using newlines so comments remain aligned,
				// but if an expression is not a "one-line" expression, or if
				// multiple expressions are on the same line, the section is
				// broken with a formfeed
				if p.linebreak(line, 1, 2, ws, !oneLiner || prevBreak+1 < i) {
					ws = ignore
					*multiLine = true
					prevBreak = i
				}
			} else {
				p.print(blank)
			}
		}
		// determine if x satisfies the "one-liner" criteria
		// TODO(gri): determine if the multiline information returned
		//            from p.expr0 is precise enough so it could be
		//            used instead
		oneLiner = p.isOneLineExpr(x)
		if t, isPair := x.(*ast.KeyValueExpr); isPair && oneLiner && len(list) > 1 {
			// we have a key:value expression that fits onto one line, and
			// is a list with more then one entry: align all the values
			p.expr(t.Key, multiLine)
			p.print(t.Colon, token.COLON, vtab)
			p.expr(t.Value, multiLine)
		} else {
			p.expr0(x, depth, multiLine)
		}
	}

	if mode&commaTerm != 0 && next.IsValid() && p.pos.Line < next.Line {
		// print a terminating comma if the next token is on a new line
		p.print(token.COMMA)
		if ws == ignore && mode&noIndent == 0 {
			// unindent if we indented
			p.print(unindent)
		}
		p.print(formfeed) // terminating comma needs a line break to look good
		return
	}

	if mode&blankEnd != 0 {
		p.print(blank)
	}

	if ws == ignore && mode&noIndent == 0 {
		// unindent if we indented
		p.print(unindent)
	}
}


// Sets multiLine to true if the the parameter list spans multiple lines.
func (p *printer) parameters(fields *ast.FieldList, multiLine *bool) {
	p.print(fields.Opening, token.LPAREN)
	if len(fields.List) > 0 {
		for i, par := range fields.List {
			if i > 0 {
				p.print(token.COMMA, blank)
			}
			if len(par.Names) > 0 {
				p.identList(par.Names, multiLine)
				p.print(blank)
			}
			p.expr(par.Type, multiLine)
		}
	}
	p.print(fields.Closing, token.RPAREN)
}


// Sets multiLine to true if the signature spans multiple lines.
func (p *printer) signature(params, result *ast.FieldList, multiLine *bool) {
	p.parameters(params, multiLine)
	n := result.NumFields()
	if n > 0 {
		p.print(blank)
		if n == 1 && result.List[0].Names == nil {
			// single anonymous result; no ()'s
			p.expr(result.List[0].Type, multiLine)
			return
		}
		p.parameters(result, multiLine)
	}
}


func identListSize(list []*ast.Ident, maxSize int) (size int) {
	for i, x := range list {
		if i > 0 {
			size += 2 // ", "
		}
		size += len(x.Name())
		if size >= maxSize {
			break
		}
	}
	return
}


func (p *printer) isOneLineFieldList(list []*ast.Field) bool {
	if len(list) != 1 {
		return false // allow only one field
	}
	f := list[0]
	if f.Tag != nil || f.Comment != nil {
		return false // don't allow tags or comments
	}
	// only name(s) and type
	const maxSize = 30 // adjust as appropriate, this is an approximate value
	namesSize := identListSize(f.Names, maxSize)
	if namesSize > 0 {
		namesSize = 1 // blank between names and types
	}
	typeSize := p.nodeSize(f.Type, maxSize)
	return namesSize+typeSize <= maxSize
}


func (p *printer) setLineComment(text string) {
	p.setComment(&ast.CommentGroup{[]*ast.Comment{&ast.Comment{noPos, []byte(text)}}})
}


func (p *printer) fieldList(fields *ast.FieldList, isIncomplete bool, ctxt exprContext) {
	lbrace := fields.Opening
	list := fields.List
	rbrace := fields.Closing

	if !isIncomplete && !p.commentBefore(rbrace) {
		// possibly a one-line struct/interface
		if len(list) == 0 {
			// no blank between keyword and {} in this case
			p.print(lbrace, token.LBRACE, rbrace, token.RBRACE)
			return
		} else if ctxt&(compositeLit|structType) == compositeLit|structType &&
			p.isOneLineFieldList(list) { // for now ignore interfaces
			// small enough - print on one line
			// (don't use identList and ignore source line breaks)
			p.print(lbrace, token.LBRACE, blank)
			f := list[0]
			for i, x := range f.Names {
				if i > 0 {
					p.print(token.COMMA, blank)
				}
				p.expr(x, ignoreMultiLine)
			}
			if len(f.Names) > 0 {
				p.print(blank)
			}
			p.expr(f.Type, ignoreMultiLine)
			p.print(blank, rbrace, token.RBRACE)
			return
		}
	}

	// at least one entry or incomplete
	p.print(blank, lbrace, token.LBRACE, indent, formfeed)
	if ctxt&structType != 0 {

		sep := vtab
		if len(list) == 1 {
			sep = blank
		}
		var ml bool
		for i, f := range list {
			if i > 0 {
				p.linebreak(f.Pos().Line, 1, 2, ignore, ml)
			}
			ml = false
			extraTabs := 0
			p.setComment(f.Doc)
			if len(f.Names) > 0 {
				// named fields
				p.identList(f.Names, &ml)
				p.print(sep)
				p.expr(f.Type, &ml)
				extraTabs = 1
			} else {
				// anonymous field
				p.expr(f.Type, &ml)
				extraTabs = 2
			}
			if f.Tag != nil {
				if len(f.Names) > 0 && sep == vtab {
					p.print(sep)
				}
				p.print(sep)
				p.expr(f.Tag, &ml)
				extraTabs = 0
			}
			if f.Comment != nil {
				for ; extraTabs > 0; extraTabs-- {
					p.print(sep)
				}
				p.setComment(f.Comment)
			}
		}
		if isIncomplete {
			if len(list) > 0 {
				p.print(formfeed)
			}
			p.flush(rbrace, false) // make sure we don't loose the last line comment
			p.setLineComment("// contains unexported fields")
		}

	} else { // interface

		var ml bool
		for i, f := range list {
			if i > 0 {
				p.linebreak(f.Pos().Line, 1, 2, ignore, ml)
			}
			ml = false
			p.setComment(f.Doc)
			if ftyp, isFtyp := f.Type.(*ast.FuncType); isFtyp {
				// method
				p.expr(f.Names[0], &ml)
				p.signature(ftyp.Params, ftyp.Results, &ml)
			} else {
				// embedded interface
				p.expr(f.Type, &ml)
			}
			p.setComment(f.Comment)
		}
		if isIncomplete {
			if len(list) > 0 {
				p.print(formfeed)
			}
			p.flush(rbrace, false) // make sure we don't loose the last line comment
			p.setLineComment("// contains unexported methods")
		}

	}
	p.print(unindent, formfeed, rbrace, token.RBRACE)
}


// ----------------------------------------------------------------------------
// Expressions

// exprContext describes the syntactic environment in which an expression node is printed.
type exprContext uint

const (
	compositeLit = 1 << iota
	structType
)


func walkBinary(e *ast.BinaryExpr) (has5, has6 bool, maxProblem int) {
	switch e.Op.Precedence() {
	case 5:
		has5 = true
	case 6:
		has6 = true
	}

	switch l := e.X.(type) {
	case *ast.BinaryExpr:
		if l.Op.Precedence() < e.Op.Precedence() {
			// parens will be inserted.
			// pretend this is an *ast.ParenExpr and do nothing.
			break
		}
		h5, h6, mp := walkBinary(l)
		has5 = has5 || h5
		has6 = has6 || h6
		if maxProblem < mp {
			maxProblem = mp
		}
	}

	switch r := e.Y.(type) {
	case *ast.BinaryExpr:
		if r.Op.Precedence() <= e.Op.Precedence() {
			// parens will be inserted.
			// pretend this is an *ast.ParenExpr and do nothing.
			break
		}
		h5, h6, mp := walkBinary(r)
		has5 = has5 || h5
		has6 = has6 || h6
		if maxProblem < mp {
			maxProblem = mp
		}

	case *ast.StarExpr:
		if e.Op.String() == "/" {
			maxProblem = 6
		}

	case *ast.UnaryExpr:
		switch e.Op.String() + r.Op.String() {
		case "/*":
			maxProblem = 6
		case "++", "--":
			if maxProblem < 5 {
				maxProblem = 5
			}
		}
	}
	return
}


func cutoff(e *ast.BinaryExpr, depth int) int {
	has5, has6, maxProblem := walkBinary(e)
	if maxProblem > 0 {
		return maxProblem + 1
	}
	if has5 && has6 {
		if depth == 1 {
			return 6
		}
		return 5
	}
	if depth == 1 {
		return 7
	}
	return 5
}


func diffPrec(expr ast.Expr, prec int) int {
	x, ok := expr.(*ast.BinaryExpr)
	if !ok || prec != x.Op.Precedence() {
		return 1
	}
	return 0
}


// Format the binary expression: decide the cutoff and then format.
// Let's call depth == 1 Normal mode, and depth > 1 Compact mode.
// (Algorithm suggestion by Russ Cox.)
//
// The precedences are:
//	6             *  /  %  <<  >>  &  &^
//	5             +  -  |  ^
//	4             ==  !=  <  <=  >  >=
//	3             <-
//	2             &&
//	1             ||
//
// The only decision is whether there will be spaces around levels 5 and 6.
// There are never spaces at level 7 (unary), and always spaces at levels 4 and below.
//
// To choose the cutoff, look at the whole expression but excluding primary
// expressions (function calls, parenthesized exprs), and apply these rules:
//
//	1) If there is a binary operator with a right side unary operand
//	   that would clash without a space, the cutoff must be (in order):
//
//		&^	7
//		/*	7
//		++	6
//		--	6
//
//	2) If there is a mix of level 6 and level 5 operators, then the cutoff
//	   is 6 (use spaces to distinguish precedence) in Normal mode
//	   and 5 (never use spaces) in Compact mode.
//
//	3) If there are no level 5 operators or no level 6 operators, then the
//	   cutoff is 7 (always use spaces) in Normal mode
//	   and 5 (never use spaces) in Compact mode.
//
// Sets multiLine to true if the binary expression spans multiple lines.
func (p *printer) binaryExpr(x *ast.BinaryExpr, prec1, cutoff, depth int, multiLine *bool) {
	prec := x.Op.Precedence()
	if prec < prec1 {
		// parenthesis needed
		// Note: The parser inserts an ast.ParenExpr node; thus this case
		//       can only occur if the AST is created in a different way.
		p.print(token.LPAREN)
		p.expr0(x, depth-1, multiLine) // parentheses undo one level of depth
		p.print(token.RPAREN)
		return
	}

	printBlank := prec < cutoff

	ws := indent
	p.expr1(x.X, prec, depth+diffPrec(x.X, prec), 0, multiLine)
	if printBlank {
		p.print(blank)
	}
	xline := p.pos.Line // before the operator (it may be on the next line!)
	yline := x.Y.Pos().Line
	p.print(x.OpPos, x.Op)
	if xline != yline && xline > 0 && yline > 0 {
		// at least one line break, but respect an extra empty line
		// in the source
		if p.linebreak(yline, 1, 2, ws, true) {
			ws = ignore
			*multiLine = true
			printBlank = false // no blank after line break
		}
	}
	if printBlank {
		p.print(blank)
	}
	p.expr1(x.Y, prec+1, depth+1, 0, multiLine)
	if ws == ignore {
		p.print(unindent)
	}
}


func isBinary(expr ast.Expr) bool {
	_, ok := expr.(*ast.BinaryExpr)
	return ok
}


// Sets multiLine to true if the expression spans multiple lines.
func (p *printer) expr1(expr ast.Expr, prec1, depth int, ctxt exprContext, multiLine *bool) {
	p.print(expr.Pos())

	switch x := expr.(type) {
	case *ast.BadExpr:
		p.print("BadExpr")

	case *ast.Ident:
		p.print(x)

	case *ast.BinaryExpr:
		if depth < 1 {
			p.internalError("depth < 1:", depth)
			depth = 1
		}
		p.binaryExpr(x, prec1, cutoff(x, depth), depth, multiLine)

	case *ast.KeyValueExpr:
		p.expr(x.Key, multiLine)
		p.print(x.Colon, token.COLON, blank)
		p.expr(x.Value, multiLine)

	case *ast.StarExpr:
		const prec = token.UnaryPrec
		if prec < prec1 {
			// parenthesis needed
			p.print(token.LPAREN)
			p.print(token.MUL)
			p.expr(x.X, multiLine)
			p.print(token.RPAREN)
		} else {
			// no parenthesis needed
			p.print(token.MUL)
			p.expr(x.X, multiLine)
		}

	case *ast.UnaryExpr:
		const prec = token.UnaryPrec
		if prec < prec1 {
			// parenthesis needed
			p.print(token.LPAREN)
			p.expr(x, multiLine)
			p.print(token.RPAREN)
		} else {
			// no parenthesis needed
			p.print(x.Op)
			if x.Op == token.RANGE {
				// TODO(gri) Remove this code if it cannot be reached.
				p.print(blank)
			}
			p.expr1(x.X, prec, depth, 0, multiLine)
		}

	case *ast.BasicLit:
		p.print(x)

	case *ast.FuncLit:
		p.expr(x.Type, multiLine)
		p.funcBody(x.Body, distance(x.Type.Pos(), p.pos), true, multiLine)

	case *ast.ParenExpr:
		p.print(token.LPAREN)
		p.expr0(x.X, depth-1, multiLine) // parentheses undo one level of depth
		p.print(x.Rparen, token.RPAREN)

	case *ast.SelectorExpr:
		p.expr1(x.X, token.HighestPrec, depth, 0, multiLine)
		p.print(token.PERIOD)
		p.expr1(x.Sel, token.HighestPrec, depth, 0, multiLine)

	case *ast.TypeAssertExpr:
		p.expr1(x.X, token.HighestPrec, depth, 0, multiLine)
		p.print(token.PERIOD, token.LPAREN)
		if x.Type != nil {
			p.expr(x.Type, multiLine)
		} else {
			p.print(token.TYPE)
		}
		p.print(token.RPAREN)

	case *ast.IndexExpr:
		// TODO(gri): should treat[] like parentheses and undo one level of depth
		p.expr1(x.X, token.HighestPrec, 1, 0, multiLine)
		p.print(token.LBRACK)
		p.expr0(x.Index, depth+1, multiLine)
		p.print(token.RBRACK)

	case *ast.SliceExpr:
		// TODO(gri): should treat[] like parentheses and undo one level of depth
		p.expr1(x.X, token.HighestPrec, 1, 0, multiLine)
		p.print(token.LBRACK)
		p.expr0(x.Index, depth+1, multiLine)
		// blanks around ":" if both sides exist and either side is a binary expression
		if depth <= 1 && x.End != nil && (isBinary(x.Index) || isBinary(x.End)) {
			p.print(blank, token.COLON, blank)
		} else {
			p.print(token.COLON)
		}
		if x.End != nil {
			p.expr0(x.End, depth+1, multiLine)
		}
		p.print(token.RBRACK)

	case *ast.CallExpr:
		if len(x.Args) > 1 {
			depth++
		}
		p.expr1(x.Fun, token.HighestPrec, depth, 0, multiLine)
		p.print(x.Lparen, token.LPAREN)
		p.exprList(x.Lparen, x.Args, depth, commaSep|commaTerm, multiLine, x.Rparen)
		p.print(x.Rparen, token.RPAREN)

	case *ast.CompositeLit:
		p.expr1(x.Type, token.HighestPrec, depth, compositeLit, multiLine)
		p.print(x.Lbrace, token.LBRACE)
		p.exprList(x.Lbrace, x.Elts, 1, commaSep|commaTerm, multiLine, x.Rbrace)
		p.print(x.Rbrace, token.RBRACE)

	case *ast.Ellipsis:
		p.print(token.ELLIPSIS)
		if x.Elt != nil {
			p.expr(x.Elt, multiLine)
		}

	case *ast.ArrayType:
		p.print(token.LBRACK)
		if x.Len != nil {
			p.expr(x.Len, multiLine)
		}
		p.print(token.RBRACK)
		p.expr(x.Elt, multiLine)

	case *ast.StructType:
		p.print(token.STRUCT)
		p.fieldList(x.Fields, x.Incomplete, ctxt|structType)

	case *ast.FuncType:
		p.print(token.FUNC)
		p.signature(x.Params, x.Results, multiLine)

	case *ast.InterfaceType:
		p.print(token.INTERFACE)
		p.fieldList(x.Methods, x.Incomplete, ctxt)

	case *ast.MapType:
		p.print(token.MAP, token.LBRACK)
		p.expr(x.Key, multiLine)
		p.print(token.RBRACK)
		p.expr(x.Value, multiLine)

	case *ast.ChanType:
		switch x.Dir {
		case ast.SEND | ast.RECV:
			p.print(token.CHAN)
		case ast.RECV:
			p.print(token.ARROW, token.CHAN)
		case ast.SEND:
			p.print(token.CHAN, token.ARROW)
		}
		p.print(blank)
		p.expr(x.Value, multiLine)

	default:
		panic("unreachable")
	}

	return
}


func (p *printer) expr0(x ast.Expr, depth int, multiLine *bool) {
	p.expr1(x, token.LowestPrec, depth, 0, multiLine)
}


// Sets multiLine to true if the expression spans multiple lines.
func (p *printer) expr(x ast.Expr, multiLine *bool) {
	const depth = 1
	p.expr1(x, token.LowestPrec, depth, 0, multiLine)
}


// ----------------------------------------------------------------------------
// Statements

const maxStmtNewlines = 2 // maximum number of newlines between statements

// Print the statement list indented, but without a newline after the last statement.
// Extra line breaks between statements in the source are respected but at most one
// empty line is printed between statements.
func (p *printer) stmtList(list []ast.Stmt, _indent int) {
	// TODO(gri): fix _indent code
	if _indent > 0 {
		p.print(indent)
	}
	var multiLine bool
	for i, s := range list {
		// _indent == 0 only for lists of switch/select case clauses;
		// in those cases each clause is a new section
		p.linebreak(s.Pos().Line, 1, maxStmtNewlines, ignore, i == 0 || _indent == 0 || multiLine)
		multiLine = false
		p.stmt(s, &multiLine)
	}
	if _indent > 0 {
		p.print(unindent)
	}
}


// block prints an *ast.BlockStmt; it always spans at least two lines.
func (p *printer) block(s *ast.BlockStmt, indent int) {
	p.print(s.Pos(), token.LBRACE)
	p.stmtList(s.List, indent)
	p.linebreak(s.Rbrace.Line, 1, maxStmtNewlines, ignore, true)
	p.print(s.Rbrace, token.RBRACE)
}


// TODO(gri): Decide if this should be used more broadly. The printing code
//            knows when to insert parentheses for precedence reasons, but
//            need to be careful to keep them around type expressions.
func stripParens(x ast.Expr) ast.Expr {
	if px, hasParens := x.(*ast.ParenExpr); hasParens {
		return stripParens(px.X)
	}
	return x
}


func (p *printer) controlClause(isForStmt bool, init ast.Stmt, expr ast.Expr, post ast.Stmt) {
	p.print(blank)
	needsBlank := false
	if init == nil && post == nil {
		// no semicolons required
		if expr != nil {
			p.expr(stripParens(expr), ignoreMultiLine)
			needsBlank = true
		}
	} else {
		// all semicolons required
		// (they are not separators, print them explicitly)
		if init != nil {
			p.stmt(init, ignoreMultiLine)
		}
		p.print(token.SEMICOLON, blank)
		if expr != nil {
			p.expr(stripParens(expr), ignoreMultiLine)
			needsBlank = true
		}
		if isForStmt {
			p.print(token.SEMICOLON, blank)
			needsBlank = false
			if post != nil {
				p.stmt(post, ignoreMultiLine)
				needsBlank = true
			}
		}
	}
	if needsBlank {
		p.print(blank)
	}
}


// Sets multiLine to true if the statements spans multiple lines.
func (p *printer) stmt(stmt ast.Stmt, multiLine *bool) {
	p.print(stmt.Pos())

	switch s := stmt.(type) {
	case *ast.BadStmt:
		p.print("BadStmt")

	case *ast.DeclStmt:
		p.decl(s.Decl, inStmtList, multiLine)

	case *ast.EmptyStmt:
		// nothing to do

	case *ast.LabeledStmt:
		// a "correcting" unindent immediately following a line break
		// is applied before the line break  if there is no comment
		// between (see writeWhitespace)
		p.print(unindent)
		p.expr(s.Label, multiLine)
		p.print(token.COLON, vtab, indent)
		p.linebreak(s.Stmt.Pos().Line, 0, 1, ignore, true)
		p.stmt(s.Stmt, multiLine)

	case *ast.ExprStmt:
		const depth = 1
		p.expr0(s.X, depth, multiLine)

	case *ast.IncDecStmt:
		const depth = 1
		p.expr0(s.X, depth+1, multiLine)
		p.print(s.Tok)

	case *ast.AssignStmt:
		var depth = 1
		if len(s.Lhs) > 1 && len(s.Rhs) > 1 {
			depth++
		}
		p.exprList(s.Pos(), s.Lhs, depth, commaSep, multiLine, s.TokPos)
		p.print(blank, s.TokPos, s.Tok)
		p.exprList(s.TokPos, s.Rhs, depth, blankStart|commaSep, multiLine, noPos)

	case *ast.GoStmt:
		p.print(token.GO, blank)
		p.expr(s.Call, multiLine)

	case *ast.DeferStmt:
		p.print(token.DEFER, blank)
		p.expr(s.Call, multiLine)

	case *ast.ReturnStmt:
		p.print(token.RETURN)
		if s.Results != nil {
			p.exprList(s.Pos(), s.Results, 1, blankStart|commaSep, multiLine, noPos)
		}

	case *ast.BranchStmt:
		p.print(s.Tok)
		if s.Label != nil {
			p.print(blank)
			p.expr(s.Label, multiLine)
		}

	case *ast.BlockStmt:
		p.block(s, 1)
		*multiLine = true

	case *ast.IfStmt:
		p.print(token.IF)
		p.controlClause(false, s.Init, s.Cond, nil)
		p.block(s.Body, 1)
		*multiLine = true
		if s.Else != nil {
			p.print(blank, token.ELSE, blank)
			switch s.Else.(type) {
			case *ast.BlockStmt, *ast.IfStmt:
				p.stmt(s.Else, ignoreMultiLine)
			default:
				p.print(token.LBRACE, indent, formfeed)
				p.stmt(s.Else, ignoreMultiLine)
				p.print(unindent, formfeed, token.RBRACE)
			}
		}

	case *ast.CaseClause:
		if s.Values != nil {
			p.print(token.CASE)
			p.exprList(s.Pos(), s.Values, 1, blankStart|commaSep, multiLine, s.Colon)
		} else {
			p.print(token.DEFAULT)
		}
		p.print(s.Colon, token.COLON)
		p.stmtList(s.Body, 1)

	case *ast.SwitchStmt:
		p.print(token.SWITCH)
		p.controlClause(false, s.Init, s.Tag, nil)
		p.block(s.Body, 0)
		*multiLine = true

	case *ast.TypeCaseClause:
		if s.Types != nil {
			p.print(token.CASE)
			p.exprList(s.Pos(), s.Types, 1, blankStart|commaSep, multiLine, s.Colon)
		} else {
			p.print(token.DEFAULT)
		}
		p.print(s.Colon, token.COLON)
		p.stmtList(s.Body, 1)

	case *ast.TypeSwitchStmt:
		p.print(token.SWITCH)
		if s.Init != nil {
			p.print(blank)
			p.stmt(s.Init, ignoreMultiLine)
			p.print(token.SEMICOLON)
		}
		p.print(blank)
		p.stmt(s.Assign, ignoreMultiLine)
		p.print(blank)
		p.block(s.Body, 0)
		*multiLine = true

	case *ast.CommClause:
		if s.Rhs != nil {
			p.print(token.CASE, blank)
			if s.Lhs != nil {
				p.expr(s.Lhs, multiLine)
				p.print(blank, s.Tok, blank)
			}
			p.expr(s.Rhs, multiLine)
		} else {
			p.print(token.DEFAULT)
		}
		p.print(s.Colon, token.COLON)
		p.stmtList(s.Body, 1)

	case *ast.SelectStmt:
		p.print(token.SELECT, blank)
		p.block(s.Body, 0)
		*multiLine = true

	case *ast.ForStmt:
		p.print(token.FOR)
		p.controlClause(true, s.Init, s.Cond, s.Post)
		p.block(s.Body, 1)
		*multiLine = true

	case *ast.RangeStmt:
		p.print(token.FOR, blank)
		p.expr(s.Key, multiLine)
		if s.Value != nil {
			p.print(token.COMMA, blank)
			p.expr(s.Value, multiLine)
		}
		p.print(blank, s.TokPos, s.Tok, blank, token.RANGE, blank)
		p.expr(stripParens(s.X), multiLine)
		p.print(blank)
		p.block(s.Body, 1)
		*multiLine = true

	default:
		panic("unreachable")
	}

	return
}


// ----------------------------------------------------------------------------
// Declarations

type declContext uint

const (
	atTop declContext = iota
	inGroup
	inStmtList
)

// The parameter n is the number of specs in the group; context specifies
// the surroundings of the declaration. Separating semicolons are printed
// depending on the context. Sets multiLine to true if the spec spans
// multiple lines.
//
func (p *printer) spec(spec ast.Spec, n int, context declContext, multiLine *bool) {
	var (
		comment   *ast.CommentGroup // a line comment, if any
		extraTabs int               // number of extra tabs before comment, if any
	)

	switch s := spec.(type) {
	case *ast.ImportSpec:
		p.setComment(s.Doc)
		if s.Name != nil {
			p.expr(s.Name, multiLine)
			p.print(blank)
		}
		p.expr(s.Path, multiLine)
		comment = s.Comment

	case *ast.ValueSpec:
		p.setComment(s.Doc)
		p.identList(s.Names, multiLine) // always present
		if n == 1 {
			if s.Type != nil {
				p.print(blank)
				p.expr(s.Type, multiLine)
			}
			if s.Values != nil {
				p.print(blank, token.ASSIGN)
				p.exprList(noPos, s.Values, 1, blankStart|commaSep, multiLine, noPos)
			}
		} else {
			extraTabs = 2
			if s.Type != nil || s.Values != nil {
				p.print(vtab)
			}
			if s.Type != nil {
				p.expr(s.Type, multiLine)
				extraTabs = 1
			}
			if s.Values != nil {
				p.print(vtab)
				p.print(token.ASSIGN)
				p.exprList(noPos, s.Values, 1, blankStart|commaSep, multiLine, noPos)
				extraTabs = 0
			}
		}
		comment = s.Comment

	case *ast.TypeSpec:
		p.setComment(s.Doc)
		p.expr(s.Name, multiLine)
		if n == 1 {
			p.print(blank)
		} else {
			p.print(vtab)
		}
		p.expr(s.Type, multiLine)
		comment = s.Comment

	default:
		panic("unreachable")
	}

	if comment != nil {
		for ; extraTabs > 0; extraTabs-- {
			p.print(vtab)
		}
		p.setComment(comment)
	}
}


// Sets multiLine to true if the declaration spans multiple lines.
func (p *printer) genDecl(d *ast.GenDecl, context declContext, multiLine *bool) {
	p.setComment(d.Doc)
	p.print(d.Pos(), d.Tok, blank)

	if d.Lparen.IsValid() {
		// group of parenthesized declarations
		p.print(d.Lparen, token.LPAREN)
		if len(d.Specs) > 0 {
			p.print(indent, formfeed)
			var ml bool
			for i, s := range d.Specs {
				if i > 0 {
					p.linebreak(s.Pos().Line, 1, 2, ignore, ml)
				}
				ml = false
				p.spec(s, len(d.Specs), inGroup, &ml)
			}
			p.print(unindent, formfeed)
			*multiLine = true
		}
		p.print(d.Rparen, token.RPAREN)

	} else {
		// single declaration
		p.spec(d.Specs[0], 1, context, multiLine)
	}
}


// nodeSize determines the size of n in chars after formatting.
// The result is <= maxSize if the node fits on one line with at
// most maxSize chars and the formatted output doesn't contain
// any control chars. Otherwise, the result is > maxSize.
//
func (p *printer) nodeSize(n ast.Node, maxSize int) (size int) {
	size = maxSize + 1 // assume n doesn't fit
	// nodeSize computation must be indendent of particular
	// style so that we always get the same decision; print
	// in RawFormat
	cfg := Config{Mode: RawFormat}
	var buf bytes.Buffer
	if _, err := cfg.Fprint(&buf, n); err != nil {
		return
	}
	if buf.Len() <= maxSize {
		for _, ch := range buf.Bytes() {
			if ch < ' ' {
				return
			}
		}
		size = buf.Len() // n fits
	}
	return
}


func (p *printer) isOneLineFunc(b *ast.BlockStmt, headerSize int) bool {
	const maxSize = 90 // adjust as appropriate, this is an approximate value
	bodySize := 0
	switch {
	case len(b.List) > 1 || p.commentBefore(b.Rbrace):
		return false // too many statements or there is a comment - all bets are off
	case len(b.List) == 1:
		bodySize = p.nodeSize(b.List[0], maxSize)
	}
	// require both headers and overall size to be not "too large"
	return headerSize <= maxSize/2 && headerSize+bodySize <= maxSize
}


// Sets multiLine to true if the function body spans multiple lines.
func (p *printer) funcBody(b *ast.BlockStmt, headerSize int, isLit bool, multiLine *bool) {
	if b == nil {
		return
	}

	if p.isOneLineFunc(b, headerSize) {
		sep := vtab
		if isLit {
			sep = blank
		}
		if len(b.List) > 0 {
			p.print(sep, b.Pos(), token.LBRACE, blank)
			p.stmt(b.List[0], ignoreMultiLine)
			p.print(blank, b.Rbrace, token.RBRACE)
		} else {
			p.print(sep, b.Pos(), token.LBRACE, b.Rbrace, token.RBRACE)
		}
		return
	}

	p.print(blank)
	p.block(b, 1)
	*multiLine = true
}


// distance returns the column difference between from and to if both
// are on the same line; if they are on different lines (or unknown)
// the result is infinity (1<<30).
func distance(from, to token.Position) int {
	if from.IsValid() && to.IsValid() && from.Line == to.Line {
		return to.Column - from.Column
	}
	return 1 << 30
}


// Sets multiLine to true if the declaration spans multiple lines.
func (p *printer) funcDecl(d *ast.FuncDecl, multiLine *bool) {
	p.setComment(d.Doc)
	p.print(d.Pos(), token.FUNC, blank)
	if d.Recv != nil {
		p.parameters(d.Recv, multiLine) // method: print receiver
		p.print(blank)
	}
	p.expr(d.Name, multiLine)
	p.signature(d.Type.Params, d.Type.Results, multiLine)
	p.funcBody(d.Body, distance(d.Pos(), p.pos), false, multiLine)
}


// Sets multiLine to true if the declaration spans multiple lines.
func (p *printer) decl(decl ast.Decl, context declContext, multiLine *bool) {
	switch d := decl.(type) {
	case *ast.BadDecl:
		p.print(d.Pos(), "BadDecl")
	case *ast.GenDecl:
		p.genDecl(d, context, multiLine)
	case *ast.FuncDecl:
		p.funcDecl(d, multiLine)
	default:
		panic("unreachable")
	}
}


// ----------------------------------------------------------------------------
// Files

const maxDeclNewlines = 3 // maximum number of newlines between declarations

func declToken(decl ast.Decl) (tok token.Token) {
	tok = token.ILLEGAL
	switch d := decl.(type) {
	case *ast.GenDecl:
		tok = d.Tok
	case *ast.FuncDecl:
		tok = token.FUNC
	}
	return
}


func (p *printer) file(src *ast.File) {
	p.setComment(src.Doc)
	p.print(src.Pos(), token.PACKAGE, blank)
	p.expr(src.Name, ignoreMultiLine)

	if len(src.Decls) > 0 {
		tok := token.ILLEGAL
		for _, d := range src.Decls {
			prev := tok
			tok = declToken(d)
			// if the declaration token changed (e.g., from CONST to TYPE)
			// print an empty line between top-level declarations
			min := 1
			if prev != tok {
				min = 2
			}
			p.linebreak(d.Pos().Line, min, maxDeclNewlines, ignore, false)
			p.decl(d, atTop, ignoreMultiLine)
		}
	}

	p.print(newline)
}
