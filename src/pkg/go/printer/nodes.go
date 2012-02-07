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

// Print as many newlines as necessary (but at least min newlines) to get to
// the current line. ws is printed before the first line break. If newSection
// is set, the first line break is printed as formfeed. Returns true if any
// line break was printed; returns false otherwise.
//
// TODO(gri): linebreak may add too many lines if the next statement at "line"
//            is preceded by comments because the computation of n assumes
//            the current position before the comment and the target position
//            after the comment. Thus, after interspersing such comments, the
//            space taken up by them is not considered to reduce the number of
//            linebreaks. At the moment there is no easy way to know about
//            future (not yet interspersed) comments in this function.
//
func (p *printer) linebreak(line, min int, ws whiteSpace, newSection bool) (printedBreak bool) {
	n := nlimit(line - p.pos.Line)
	if n < min {
		n = min
	}
	if n > 0 {
		p.print(ws)
		if newSection {
			p.print(formfeed)
			n--
		}
		for ; n > 0; n-- {
			p.print(newline)
		}
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
		p.flush(p.posFor(g.List[0].Pos()), token.ILLEGAL)
	}
	p.comments[0] = g
	p.cindex = 0
	p.nextComment() // get comment ready for use
}

type exprListMode uint

const (
	blankStart exprListMode = 1 << iota // print a blank before a non-empty list
	blankEnd                            // print a blank after a non-empty list
	commaSep                            // elements are separated by commas
	commaTerm                           // list is optionally terminated by a comma
	noIndent                            // no extra indentation in multi-line lists
	periodSep                           // elements are separated by periods
)

// Sets multiLine to true if the identifier list spans multiple lines.
// If indent is set, a multi-line identifier list is indented after the
// first linebreak encountered.
func (p *printer) identList(list []*ast.Ident, indent bool, multiLine *bool) {
	// convert into an expression list so we can re-use exprList formatting
	xlist := make([]ast.Expr, len(list))
	for i, x := range list {
		xlist[i] = x
	}
	mode := commaSep
	if !indent {
		mode |= noIndent
	}
	p.exprList(token.NoPos, xlist, 1, mode, multiLine, token.NoPos)
}

// Print a list of expressions. If the list spans multiple
// source lines, the original line breaks are respected between
// expressions. Sets multiLine to true if the list spans multiple
// lines.
//
// TODO(gri) Consider rewriting this to be independent of []ast.Expr
//           so that we can use the algorithm for any kind of list
//           (e.g., pass list via a channel over which to range).
func (p *printer) exprList(prev0 token.Pos, list []ast.Expr, depth int, mode exprListMode, multiLine *bool, next0 token.Pos) {
	if len(list) == 0 {
		return
	}

	if mode&blankStart != 0 {
		p.print(blank)
	}

	prev := p.posFor(prev0)
	next := p.posFor(next0)
	line := p.lineFor(list[0].Pos())
	endLine := p.lineFor(list[len(list)-1].End())

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

	// the first linebreak is always a formfeed since this section must not
	// depend on any previous formatting
	prevBreak := -1 // index of last expression that was followed by a linebreak
	if prev.IsValid() && prev.Line < line && p.linebreak(line, 0, ws, true) {
		ws = ignore
		*multiLine = true
		prevBreak = 0
	}

	// initialize expression/key size: a zero value indicates expr/key doesn't fit on a single line
	size := 0

	// print all list elements
	for i, x := range list {
		prevLine := line
		line = p.lineFor(x.Pos())

		// determine if the next linebreak, if any, needs to use formfeed:
		// in general, use the entire node size to make the decision; for
		// key:value expressions, use the key size
		// TODO(gri) for a better result, should probably incorporate both
		//           the key and the node size into the decision process
		useFF := true

		// determine element size: all bets are off if we don't have
		// position information for the previous and next token (likely
		// generated code - simply ignore the size in this case by setting
		// it to 0)
		prevSize := size
		const infinity = 1e6 // larger than any source line
		size = p.nodeSize(x, infinity)
		pair, isPair := x.(*ast.KeyValueExpr)
		if size <= infinity && prev.IsValid() && next.IsValid() {
			// x fits on a single line
			if isPair {
				size = p.nodeSize(pair.Key, infinity) // size <= infinity
			}
		} else {
			// size too large or we don't have good layout information
			size = 0
		}

		// if the previous line and the current line had single-
		// line-expressions and the key sizes are small or the
		// the ratio between the key sizes does not exceed a
		// threshold, align columns and do not use formfeed
		if prevSize > 0 && size > 0 {
			const smallSize = 20
			if prevSize <= smallSize && size <= smallSize {
				useFF = false
			} else {
				const r = 4 // threshold
				ratio := float64(size) / float64(prevSize)
				useFF = ratio <= 1/r || r <= ratio
			}
		}

		if i > 0 {
			switch {
			case mode&commaSep != 0:
				p.print(token.COMMA)
			case mode&periodSep != 0:
				p.print(token.PERIOD)
			}
			needsBlank := mode&periodSep == 0 // period-separated list elements don't need a blank
			if prevLine < line && prevLine > 0 && line > 0 {
				// lines are broken using newlines so comments remain aligned
				// unless forceFF is set or there are multiple expressions on
				// the same line in which case formfeed is used
				if p.linebreak(line, 0, ws, useFF || prevBreak+1 < i) {
					ws = ignore
					*multiLine = true
					prevBreak = i
					needsBlank = false // we got a line break instead
				}
			}
			if needsBlank {
				p.print(blank)
			}
		}

		if isPair && size > 0 && len(list) > 1 {
			// we have a key:value expression that fits onto one line and
			// is in a list with more then one entry: use a column for the
			// key such that consecutive entries can align if possible
			p.expr(pair.Key, multiLine)
			p.print(pair.Colon, token.COLON, vtab)
			p.expr(pair.Value, multiLine)
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
		prevLine := p.lineFor(fields.Opening)
		ws := indent
		for i, par := range fields.List {
			// determine par begin and end line (may be different
			// if there are multiple parameter names for this par
			// or the type is on a separate line)
			var parLineBeg int
			var parLineEnd = p.lineFor(par.Type.Pos())
			if len(par.Names) > 0 {
				parLineBeg = p.lineFor(par.Names[0].Pos())
			} else {
				parLineBeg = parLineEnd
			}
			// separating "," if needed
			if i > 0 {
				p.print(token.COMMA)
			}
			// separator if needed (linebreak or blank)
			if 0 < prevLine && prevLine < parLineBeg && p.linebreak(parLineBeg, 0, ws, true) {
				// break line if the opening "(" or previous parameter ended on a different line
				ws = ignore
				*multiLine = true
			} else if i > 0 {
				p.print(blank)
			}
			// parameter names
			if len(par.Names) > 0 {
				// Very subtle: If we indented before (ws == ignore), identList
				// won't indent again. If we didn't (ws == indent), identList will
				// indent if the identList spans multiple lines, and it will outdent
				// again at the end (and still ws == indent). Thus, a subsequent indent
				// by a linebreak call after a type, or in the next multi-line identList
				// will do the right thing.
				p.identList(par.Names, ws == indent, multiLine)
				p.print(blank)
			}
			// parameter type
			p.expr(par.Type, multiLine)
			prevLine = parLineEnd
		}
		// if the closing ")" is on a separate line from the last parameter,
		// print an additional "," and line break
		if closing := p.lineFor(fields.Closing); 0 < prevLine && prevLine < closing {
			p.print(",")
			p.linebreak(closing, 0, ignore, true)
		}
		// unindent if we indented
		if ws == ignore {
			p.print(unindent)
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
		size += len(x.Name)
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
	p.setComment(&ast.CommentGroup{[]*ast.Comment{{token.NoPos, text}}})
}

func (p *printer) fieldList(fields *ast.FieldList, isStruct, isIncomplete bool) {
	lbrace := fields.Opening
	list := fields.List
	rbrace := fields.Closing
	hasComments := isIncomplete || p.commentBefore(p.posFor(rbrace))
	srcIsOneLine := lbrace.IsValid() && rbrace.IsValid() && p.lineFor(lbrace) == p.lineFor(rbrace)

	if !hasComments && srcIsOneLine {
		// possibly a one-line struct/interface
		if len(list) == 0 {
			// no blank between keyword and {} in this case
			p.print(lbrace, token.LBRACE, rbrace, token.RBRACE)
			return
		} else if isStruct && p.isOneLineFieldList(list) { // for now ignore interfaces
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
	// hasComments || !srcIsOneLine

	p.print(blank, lbrace, token.LBRACE, indent)
	if hasComments || len(list) > 0 {
		p.print(formfeed)
	}

	if isStruct {

		sep := vtab
		if len(list) == 1 {
			sep = blank
		}
		var ml bool
		for i, f := range list {
			if i > 0 {
				p.linebreak(p.lineFor(f.Pos()), 1, ignore, ml)
			}
			ml = false
			extraTabs := 0
			p.setComment(f.Doc)
			if len(f.Names) > 0 {
				// named fields
				p.identList(f.Names, false, &ml)
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
			p.flush(p.posFor(rbrace), token.RBRACE) // make sure we don't lose the last line comment
			p.setLineComment("// contains filtered or unexported fields")
		}

	} else { // interface

		var ml bool
		for i, f := range list {
			if i > 0 {
				p.linebreak(p.lineFor(f.Pos()), 1, ignore, ml)
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
			p.flush(p.posFor(rbrace), token.RBRACE) // make sure we don't lose the last line comment
			p.setLineComment("// contains filtered or unexported methods")
		}

	}
	p.print(unindent, formfeed, rbrace, token.RBRACE)
}

// ----------------------------------------------------------------------------
// Expressions

func walkBinary(e *ast.BinaryExpr) (has4, has5 bool, maxProblem int) {
	switch e.Op.Precedence() {
	case 4:
		has4 = true
	case 5:
		has5 = true
	}

	switch l := e.X.(type) {
	case *ast.BinaryExpr:
		if l.Op.Precedence() < e.Op.Precedence() {
			// parens will be inserted.
			// pretend this is an *ast.ParenExpr and do nothing.
			break
		}
		h4, h5, mp := walkBinary(l)
		has4 = has4 || h4
		has5 = has5 || h5
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
		h4, h5, mp := walkBinary(r)
		has4 = has4 || h4
		has5 = has5 || h5
		if maxProblem < mp {
			maxProblem = mp
		}

	case *ast.StarExpr:
		if e.Op == token.QUO { // `*/`
			maxProblem = 5
		}

	case *ast.UnaryExpr:
		switch e.Op.String() + r.Op.String() {
		case "/*", "&&", "&^":
			maxProblem = 5
		case "++", "--":
			if maxProblem < 4 {
				maxProblem = 4
			}
		}
	}
	return
}

func cutoff(e *ast.BinaryExpr, depth int) int {
	has4, has5, maxProblem := walkBinary(e)
	if maxProblem > 0 {
		return maxProblem + 1
	}
	if has4 && has5 {
		if depth == 1 {
			return 5
		}
		return 4
	}
	if depth == 1 {
		return 6
	}
	return 4
}

func diffPrec(expr ast.Expr, prec int) int {
	x, ok := expr.(*ast.BinaryExpr)
	if !ok || prec != x.Op.Precedence() {
		return 1
	}
	return 0
}

func reduceDepth(depth int) int {
	depth--
	if depth < 1 {
		depth = 1
	}
	return depth
}

// Format the binary expression: decide the cutoff and then format.
// Let's call depth == 1 Normal mode, and depth > 1 Compact mode.
// (Algorithm suggestion by Russ Cox.)
//
// The precedences are:
//	5             *  /  %  <<  >>  &  &^
//	4             +  -  |  ^
//	3             ==  !=  <  <=  >  >=
//	2             &&
//	1             ||
//
// The only decision is whether there will be spaces around levels 4 and 5.
// There are never spaces at level 6 (unary), and always spaces at levels 3 and below.
//
// To choose the cutoff, look at the whole expression but excluding primary
// expressions (function calls, parenthesized exprs), and apply these rules:
//
//	1) If there is a binary operator with a right side unary operand
//	   that would clash without a space, the cutoff must be (in order):
//
//		/*	6
//		&&	6
//		&^	6
//		++	5
//		--	5
//
//         (Comparison operators always have spaces around them.)
//
//	2) If there is a mix of level 5 and level 4 operators, then the cutoff
//	   is 5 (use spaces to distinguish precedence) in Normal mode
//	   and 4 (never use spaces) in Compact mode.
//
//	3) If there are no level 4 operators or no level 5 operators, then the
//	   cutoff is 6 (always use spaces) in Normal mode
//	   and 4 (never use spaces) in Compact mode.
//
// Sets multiLine to true if the binary expression spans multiple lines.
func (p *printer) binaryExpr(x *ast.BinaryExpr, prec1, cutoff, depth int, multiLine *bool) {
	prec := x.Op.Precedence()
	if prec < prec1 {
		// parenthesis needed
		// Note: The parser inserts an ast.ParenExpr node; thus this case
		//       can only occur if the AST is created in a different way.
		p.print(token.LPAREN)
		p.expr0(x, reduceDepth(depth), multiLine) // parentheses undo one level of depth
		p.print(token.RPAREN)
		return
	}

	printBlank := prec < cutoff

	ws := indent
	p.expr1(x.X, prec, depth+diffPrec(x.X, prec), multiLine)
	if printBlank {
		p.print(blank)
	}
	xline := p.pos.Line // before the operator (it may be on the next line!)
	yline := p.lineFor(x.Y.Pos())
	p.print(x.OpPos, x.Op)
	if xline != yline && xline > 0 && yline > 0 {
		// at least one line break, but respect an extra empty line
		// in the source
		if p.linebreak(yline, 1, ws, true) {
			ws = ignore
			*multiLine = true
			printBlank = false // no blank after line break
		}
	}
	if printBlank {
		p.print(blank)
	}
	p.expr1(x.Y, prec+1, depth+1, multiLine)
	if ws == ignore {
		p.print(unindent)
	}
}

func isBinary(expr ast.Expr) bool {
	_, ok := expr.(*ast.BinaryExpr)
	return ok
}

// If the expression contains one or more selector expressions, splits it into
// two expressions at the rightmost period. Writes entire expr to suffix when
// selector isn't found. Rewrites AST nodes for calls, index expressions and
// type assertions, all of which may be found in selector chains, to make them
// parts of the chain.
func splitSelector(expr ast.Expr) (body, suffix ast.Expr) {
	switch x := expr.(type) {
	case *ast.SelectorExpr:
		body, suffix = x.X, x.Sel
		return
	case *ast.CallExpr:
		body, suffix = splitSelector(x.Fun)
		if body != nil {
			suffix = &ast.CallExpr{suffix, x.Lparen, x.Args, x.Ellipsis, x.Rparen}
			return
		}
	case *ast.IndexExpr:
		body, suffix = splitSelector(x.X)
		if body != nil {
			suffix = &ast.IndexExpr{suffix, x.Lbrack, x.Index, x.Rbrack}
			return
		}
	case *ast.SliceExpr:
		body, suffix = splitSelector(x.X)
		if body != nil {
			suffix = &ast.SliceExpr{suffix, x.Lbrack, x.Low, x.High, x.Rbrack}
			return
		}
	case *ast.TypeAssertExpr:
		body, suffix = splitSelector(x.X)
		if body != nil {
			suffix = &ast.TypeAssertExpr{suffix, x.Type}
			return
		}
	}
	suffix = expr
	return
}

// Convert an expression into an expression list split at the periods of
// selector expressions.
func selectorExprList(expr ast.Expr) (list []ast.Expr) {
	// split expression
	for expr != nil {
		var suffix ast.Expr
		expr, suffix = splitSelector(expr)
		list = append(list, suffix)
	}

	// reverse list
	for i, j := 0, len(list)-1; i < j; i, j = i+1, j-1 {
		list[i], list[j] = list[j], list[i]
	}

	return
}

// Sets multiLine to true if the expression spans multiple lines.
func (p *printer) expr1(expr ast.Expr, prec1, depth int, multiLine *bool) {
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
			p.expr1(x.X, prec, depth, multiLine)
		}

	case *ast.BasicLit:
		p.print(x)

	case *ast.FuncLit:
		p.expr(x.Type, multiLine)
		p.funcBody(x.Body, p.distance(x.Type.Pos(), p.pos), true, multiLine)

	case *ast.ParenExpr:
		if _, hasParens := x.X.(*ast.ParenExpr); hasParens {
			// don't print parentheses around an already parenthesized expression
			// TODO(gri) consider making this more general and incorporate precedence levels
			p.expr0(x.X, reduceDepth(depth), multiLine) // parentheses undo one level of depth
		} else {
			p.print(token.LPAREN)
			p.expr0(x.X, reduceDepth(depth), multiLine) // parentheses undo one level of depth
			p.print(x.Rparen, token.RPAREN)
		}

	case *ast.SelectorExpr:
		parts := selectorExprList(expr)
		p.exprList(token.NoPos, parts, depth, periodSep, multiLine, token.NoPos)

	case *ast.TypeAssertExpr:
		p.expr1(x.X, token.HighestPrec, depth, multiLine)
		p.print(token.PERIOD, token.LPAREN)
		if x.Type != nil {
			p.expr(x.Type, multiLine)
		} else {
			p.print(token.TYPE)
		}
		p.print(token.RPAREN)

	case *ast.IndexExpr:
		// TODO(gri): should treat[] like parentheses and undo one level of depth
		p.expr1(x.X, token.HighestPrec, 1, multiLine)
		p.print(x.Lbrack, token.LBRACK)
		p.expr0(x.Index, depth+1, multiLine)
		p.print(x.Rbrack, token.RBRACK)

	case *ast.SliceExpr:
		// TODO(gri): should treat[] like parentheses and undo one level of depth
		p.expr1(x.X, token.HighestPrec, 1, multiLine)
		p.print(x.Lbrack, token.LBRACK)
		if x.Low != nil {
			p.expr0(x.Low, depth+1, multiLine)
		}
		// blanks around ":" if both sides exist and either side is a binary expression
		if depth <= 1 && x.Low != nil && x.High != nil && (isBinary(x.Low) || isBinary(x.High)) {
			p.print(blank, token.COLON, blank)
		} else {
			p.print(token.COLON)
		}
		if x.High != nil {
			p.expr0(x.High, depth+1, multiLine)
		}
		p.print(x.Rbrack, token.RBRACK)

	case *ast.CallExpr:
		if len(x.Args) > 1 {
			depth++
		}
		p.expr1(x.Fun, token.HighestPrec, depth, multiLine)
		p.print(x.Lparen, token.LPAREN)
		p.exprList(x.Lparen, x.Args, depth, commaSep|commaTerm, multiLine, x.Rparen)
		if x.Ellipsis.IsValid() {
			p.print(x.Ellipsis, token.ELLIPSIS)
		}
		p.print(x.Rparen, token.RPAREN)

	case *ast.CompositeLit:
		// composite literal elements that are composite literals themselves may have the type omitted
		if x.Type != nil {
			p.expr1(x.Type, token.HighestPrec, depth, multiLine)
		}
		p.print(x.Lbrace, token.LBRACE)
		p.exprList(x.Lbrace, x.Elts, 1, commaSep|commaTerm, multiLine, x.Rbrace)
		// do not insert extra line breaks because of comments before
		// the closing '}' as it might break the code if there is no
		// trailing ','
		p.print(noExtraLinebreak, x.Rbrace, token.RBRACE, noExtraLinebreak)

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
		p.fieldList(x.Fields, true, x.Incomplete)

	case *ast.FuncType:
		p.print(token.FUNC)
		p.signature(x.Params, x.Results, multiLine)

	case *ast.InterfaceType:
		p.print(token.INTERFACE)
		p.fieldList(x.Methods, false, x.Incomplete)

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
	p.expr1(x, token.LowestPrec, depth, multiLine)
}

// Sets multiLine to true if the expression spans multiple lines.
func (p *printer) expr(x ast.Expr, multiLine *bool) {
	const depth = 1
	p.expr1(x, token.LowestPrec, depth, multiLine)
}

// ----------------------------------------------------------------------------
// Statements

// Print the statement list indented, but without a newline after the last statement.
// Extra line breaks between statements in the source are respected but at most one
// empty line is printed between statements.
func (p *printer) stmtList(list []ast.Stmt, _indent int, nextIsRBrace bool) {
	// TODO(gri): fix _indent code
	if _indent > 0 {
		p.print(indent)
	}
	var multiLine bool
	for i, s := range list {
		// _indent == 0 only for lists of switch/select case clauses;
		// in those cases each clause is a new section
		p.linebreak(p.lineFor(s.Pos()), 1, ignore, i == 0 || _indent == 0 || multiLine)
		multiLine = false
		p.stmt(s, nextIsRBrace && i == len(list)-1, &multiLine)
	}
	if _indent > 0 {
		p.print(unindent)
	}
}

// block prints an *ast.BlockStmt; it always spans at least two lines.
func (p *printer) block(s *ast.BlockStmt, indent int) {
	p.print(s.Pos(), token.LBRACE)
	p.stmtList(s.List, indent, true)
	p.linebreak(p.lineFor(s.Rbrace), 1, ignore, true)
	p.print(s.Rbrace, token.RBRACE)
}

func isTypeName(x ast.Expr) bool {
	switch t := x.(type) {
	case *ast.Ident:
		return true
	case *ast.SelectorExpr:
		return isTypeName(t.X)
	}
	return false
}

func stripParens(x ast.Expr) ast.Expr {
	if px, strip := x.(*ast.ParenExpr); strip {
		// parentheses must not be stripped if there are any
		// unparenthesized composite literals starting with
		// a type name
		ast.Inspect(px.X, func(node ast.Node) bool {
			switch x := node.(type) {
			case *ast.ParenExpr:
				// parentheses protect enclosed composite literals
				return false
			case *ast.CompositeLit:
				if isTypeName(x.Type) {
					strip = false // do not strip parentheses
				}
				return false
			}
			// in all other cases, keep inspecting
			return true
		})
		if strip {
			return stripParens(px.X)
		}
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
			p.stmt(init, false, ignoreMultiLine)
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
				p.stmt(post, false, ignoreMultiLine)
				needsBlank = true
			}
		}
	}
	if needsBlank {
		p.print(blank)
	}
}

// Sets multiLine to true if the statements spans multiple lines.
func (p *printer) stmt(stmt ast.Stmt, nextIsRBrace bool, multiLine *bool) {
	p.print(stmt.Pos())

	switch s := stmt.(type) {
	case *ast.BadStmt:
		p.print("BadStmt")

	case *ast.DeclStmt:
		p.decl(s.Decl, multiLine)

	case *ast.EmptyStmt:
		// nothing to do

	case *ast.LabeledStmt:
		// a "correcting" unindent immediately following a line break
		// is applied before the line break if there is no comment
		// between (see writeWhitespace)
		p.print(unindent)
		p.expr(s.Label, multiLine)
		p.print(s.Colon, token.COLON, indent)
		if e, isEmpty := s.Stmt.(*ast.EmptyStmt); isEmpty {
			if !nextIsRBrace {
				p.print(newline, e.Pos(), token.SEMICOLON)
				break
			}
		} else {
			p.linebreak(p.lineFor(s.Stmt.Pos()), 1, ignore, true)
		}
		p.stmt(s.Stmt, nextIsRBrace, multiLine)

	case *ast.ExprStmt:
		const depth = 1
		p.expr0(s.X, depth, multiLine)

	case *ast.SendStmt:
		const depth = 1
		p.expr0(s.Chan, depth, multiLine)
		p.print(blank, s.Arrow, token.ARROW, blank)
		p.expr0(s.Value, depth, multiLine)

	case *ast.IncDecStmt:
		const depth = 1
		p.expr0(s.X, depth+1, multiLine)
		p.print(s.TokPos, s.Tok)

	case *ast.AssignStmt:
		var depth = 1
		if len(s.Lhs) > 1 && len(s.Rhs) > 1 {
			depth++
		}
		p.exprList(s.Pos(), s.Lhs, depth, commaSep, multiLine, s.TokPos)
		p.print(blank, s.TokPos, s.Tok)
		p.exprList(s.TokPos, s.Rhs, depth, blankStart|commaSep, multiLine, token.NoPos)

	case *ast.GoStmt:
		p.print(token.GO, blank)
		p.expr(s.Call, multiLine)

	case *ast.DeferStmt:
		p.print(token.DEFER, blank)
		p.expr(s.Call, multiLine)

	case *ast.ReturnStmt:
		p.print(token.RETURN)
		if s.Results != nil {
			p.exprList(s.Pos(), s.Results, 1, blankStart|commaSep, multiLine, token.NoPos)
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
				p.stmt(s.Else, nextIsRBrace, ignoreMultiLine)
			default:
				p.print(token.LBRACE, indent, formfeed)
				p.stmt(s.Else, true, ignoreMultiLine)
				p.print(unindent, formfeed, token.RBRACE)
			}
		}

	case *ast.CaseClause:
		if s.List != nil {
			p.print(token.CASE)
			p.exprList(s.Pos(), s.List, 1, blankStart|commaSep, multiLine, s.Colon)
		} else {
			p.print(token.DEFAULT)
		}
		p.print(s.Colon, token.COLON)
		p.stmtList(s.Body, 1, nextIsRBrace)

	case *ast.SwitchStmt:
		p.print(token.SWITCH)
		p.controlClause(false, s.Init, s.Tag, nil)
		p.block(s.Body, 0)
		*multiLine = true

	case *ast.TypeSwitchStmt:
		p.print(token.SWITCH)
		if s.Init != nil {
			p.print(blank)
			p.stmt(s.Init, false, ignoreMultiLine)
			p.print(token.SEMICOLON)
		}
		p.print(blank)
		p.stmt(s.Assign, false, ignoreMultiLine)
		p.print(blank)
		p.block(s.Body, 0)
		*multiLine = true

	case *ast.CommClause:
		if s.Comm != nil {
			p.print(token.CASE, blank)
			p.stmt(s.Comm, false, ignoreMultiLine)
		} else {
			p.print(token.DEFAULT)
		}
		p.print(s.Colon, token.COLON)
		p.stmtList(s.Body, 1, nextIsRBrace)

	case *ast.SelectStmt:
		p.print(token.SELECT, blank)
		body := s.Body
		if len(body.List) == 0 && !p.commentBefore(p.posFor(body.Rbrace)) {
			// print empty select statement w/o comments on one line
			p.print(body.Lbrace, token.LBRACE, body.Rbrace, token.RBRACE)
		} else {
			p.block(body, 0)
			*multiLine = true
		}

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

// The keepTypeColumn function determines if the type column of a series of
// consecutive const or var declarations must be kept, or if initialization
// values (V) can be placed in the type column (T) instead. The i'th entry
// in the result slice is true if the type column in spec[i] must be kept.
//
// For example, the declaration:
//
//	const (
//		foobar int = 42 // comment
//		x          = 7  // comment
//		foo
//              bar = 991
//	)
//
// leads to the type/values matrix below. A run of value columns (V) can
// be moved into the type column if there is no type for any of the values
// in that column (we only move entire columns so that they align properly).
//
//	matrix        formatted     result
//                    matrix
//	T  V    ->    T  V     ->   true      there is a T and so the type
//	-  V          -  V          true      column must be kept
//	-  -          -  -          false
//	-  V          V  -          false     V is moved into T column
//
func keepTypeColumn(specs []ast.Spec) []bool {
	m := make([]bool, len(specs))

	populate := func(i, j int, keepType bool) {
		if keepType {
			for ; i < j; i++ {
				m[i] = true
			}
		}
	}

	i0 := -1 // if i0 >= 0 we are in a run and i0 is the start of the run
	var keepType bool
	for i, s := range specs {
		t := s.(*ast.ValueSpec)
		if t.Values != nil {
			if i0 < 0 {
				// start of a run of ValueSpecs with non-nil Values
				i0 = i
				keepType = false
			}
		} else {
			if i0 >= 0 {
				// end of a run
				populate(i0, i, keepType)
				i0 = -1
			}
		}
		if t.Type != nil {
			keepType = true
		}
	}
	if i0 >= 0 {
		// end of a run
		populate(i0, len(specs), keepType)
	}

	return m
}

func (p *printer) valueSpec(s *ast.ValueSpec, keepType, doIndent bool, multiLine *bool) {
	p.setComment(s.Doc)
	p.identList(s.Names, doIndent, multiLine) // always present
	extraTabs := 3
	if s.Type != nil || keepType {
		p.print(vtab)
		extraTabs--
	}
	if s.Type != nil {
		p.expr(s.Type, multiLine)
	}
	if s.Values != nil {
		p.print(vtab, token.ASSIGN)
		p.exprList(token.NoPos, s.Values, 1, blankStart|commaSep, multiLine, token.NoPos)
		extraTabs--
	}
	if s.Comment != nil {
		for ; extraTabs > 0; extraTabs-- {
			p.print(vtab)
		}
		p.setComment(s.Comment)
	}
}

// The parameter n is the number of specs in the group. If doIndent is set,
// multi-line identifier lists in the spec are indented when the first
// linebreak is encountered.
// Sets multiLine to true if the spec spans multiple lines.
//
func (p *printer) spec(spec ast.Spec, n int, doIndent bool, multiLine *bool) {
	switch s := spec.(type) {
	case *ast.ImportSpec:
		p.setComment(s.Doc)
		if s.Name != nil {
			p.expr(s.Name, multiLine)
			p.print(blank)
		}
		p.expr(s.Path, multiLine)
		p.setComment(s.Comment)
		p.print(s.EndPos)

	case *ast.ValueSpec:
		if n != 1 {
			p.internalError("expected n = 1; got", n)
		}
		p.setComment(s.Doc)
		p.identList(s.Names, doIndent, multiLine) // always present
		if s.Type != nil {
			p.print(blank)
			p.expr(s.Type, multiLine)
		}
		if s.Values != nil {
			p.print(blank, token.ASSIGN)
			p.exprList(token.NoPos, s.Values, 1, blankStart|commaSep, multiLine, token.NoPos)
		}
		p.setComment(s.Comment)

	case *ast.TypeSpec:
		p.setComment(s.Doc)
		p.expr(s.Name, multiLine)
		if n == 1 {
			p.print(blank)
		} else {
			p.print(vtab)
		}
		p.expr(s.Type, multiLine)
		p.setComment(s.Comment)

	default:
		panic("unreachable")
	}
}

// Sets multiLine to true if the declaration spans multiple lines.
func (p *printer) genDecl(d *ast.GenDecl, multiLine *bool) {
	p.setComment(d.Doc)
	p.print(d.Pos(), d.Tok, blank)

	if d.Lparen.IsValid() {
		// group of parenthesized declarations
		p.print(d.Lparen, token.LPAREN)
		if n := len(d.Specs); n > 0 {
			p.print(indent, formfeed)
			if n > 1 && (d.Tok == token.CONST || d.Tok == token.VAR) {
				// two or more grouped const/var declarations:
				// determine if the type column must be kept
				keepType := keepTypeColumn(d.Specs)
				var ml bool
				for i, s := range d.Specs {
					if i > 0 {
						p.linebreak(p.lineFor(s.Pos()), 1, ignore, ml)
					}
					ml = false
					p.valueSpec(s.(*ast.ValueSpec), keepType[i], false, &ml)
				}
			} else {
				var ml bool
				for i, s := range d.Specs {
					if i > 0 {
						p.linebreak(p.lineFor(s.Pos()), 1, ignore, ml)
					}
					ml = false
					p.spec(s, n, false, &ml)
				}
			}
			p.print(unindent, formfeed)
			*multiLine = true
		}
		p.print(d.Rparen, token.RPAREN)

	} else {
		// single declaration
		p.spec(d.Specs[0], 1, true, multiLine)
	}
}

// nodeSize determines the size of n in chars after formatting.
// The result is <= maxSize if the node fits on one line with at
// most maxSize chars and the formatted output doesn't contain
// any control chars. Otherwise, the result is > maxSize.
//
func (p *printer) nodeSize(n ast.Node, maxSize int) (size int) {
	// nodeSize invokes the printer, which may invoke nodeSize
	// recursively. For deep composite literal nests, this can
	// lead to an exponential algorithm. Remember previous
	// results to prune the recursion (was issue 1628).
	if size, found := p.nodeSizes[n]; found {
		return size
	}

	size = maxSize + 1 // assume n doesn't fit
	p.nodeSizes[n] = size

	// nodeSize computation must be independent of particular
	// style so that we always get the same decision; print
	// in RawFormat
	cfg := Config{Mode: RawFormat}
	var buf bytes.Buffer
	if err := cfg.fprint(&buf, p.fset, n, p.nodeSizes); err != nil {
		return
	}
	if buf.Len() <= maxSize {
		for _, ch := range buf.Bytes() {
			if ch < ' ' {
				return
			}
		}
		size = buf.Len() // n fits
		p.nodeSizes[n] = size
	}
	return
}

func (p *printer) isOneLineFunc(b *ast.BlockStmt, headerSize int) bool {
	pos1 := b.Pos()
	pos2 := b.Rbrace
	if pos1.IsValid() && pos2.IsValid() && p.lineFor(pos1) != p.lineFor(pos2) {
		// opening and closing brace are on different lines - don't make it a one-liner
		return false
	}
	if len(b.List) > 5 || p.commentBefore(p.posFor(pos2)) {
		// too many statements or there is a comment inside - don't make it a one-liner
		return false
	}
	// otherwise, estimate body size
	const maxSize = 100
	bodySize := 0
	for i, s := range b.List {
		if i > 0 {
			bodySize += 2 // space for a semicolon and blank
		}
		bodySize += p.nodeSize(s, maxSize)
	}
	return headerSize+bodySize <= maxSize
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
		p.print(sep, b.Lbrace, token.LBRACE)
		if len(b.List) > 0 {
			p.print(blank)
			for i, s := range b.List {
				if i > 0 {
					p.print(token.SEMICOLON, blank)
				}
				p.stmt(s, i == len(b.List)-1, ignoreMultiLine)
			}
			p.print(blank)
		}
		p.print(b.Rbrace, token.RBRACE)
		return
	}

	p.print(blank)
	p.block(b, 1)
	*multiLine = true
}

// distance returns the column difference between from and to if both
// are on the same line; if they are on different lines (or unknown)
// the result is infinity.
func (p *printer) distance(from0 token.Pos, to token.Position) int {
	from := p.posFor(from0)
	if from.IsValid() && to.IsValid() && from.Line == to.Line {
		return to.Column - from.Column
	}
	return infinity
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
	p.funcBody(d.Body, p.distance(d.Pos(), p.pos), false, multiLine)
}

// Sets multiLine to true if the declaration spans multiple lines.
func (p *printer) decl(decl ast.Decl, multiLine *bool) {
	switch d := decl.(type) {
	case *ast.BadDecl:
		p.print(d.Pos(), "BadDecl")
	case *ast.GenDecl:
		p.genDecl(d, multiLine)
	case *ast.FuncDecl:
		p.funcDecl(d, multiLine)
	default:
		panic("unreachable")
	}
}

// ----------------------------------------------------------------------------
// Files

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
			// or the next declaration has documentation associated with it,
			// print an empty line between top-level declarations
			// (because p.linebreak is called with the position of d, which
			// is past any documentation, the minimum requirement is satisfied
			// even w/o the extra getDoc(d) nil-check - leave it in case the
			// linebreak logic improves - there's already a TODO).
			min := 1
			if prev != tok || getDoc(d) != nil {
				min = 2
			}
			p.linebreak(p.lineFor(d.Pos()), min, ignore, false)
			p.decl(d, ignoreMultiLine)
		}
	}

	p.print(newline)
}
