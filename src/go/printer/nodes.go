// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements printing of AST nodes; specifically
// expressions, statements, declarations, and files. It uses
// the print functionality implemented in printer.go.

package printer

import (
	"go/ast"
	"go/token"
	"math"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Formatting issues:
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
// is set, the first line break is printed as formfeed. Returns 0 if no line
// breaks were printed, returns 1 if there was exactly one newline printed,
// and returns a value > 1 if there was a formfeed or more than one newline
// printed.
//
// TODO(gri): linebreak may add too many lines if the next statement at "line"
// is preceded by comments because the computation of n assumes
// the current position before the comment and the target position
// after the comment. Thus, after interspersing such comments, the
// space taken up by them is not considered to reduce the number of
// linebreaks. At the moment there is no easy way to know about
// future (not yet interspersed) comments in this function.
func (p *printer) linebreak(line, min int, ws whiteSpace, newSection bool) (nbreaks int) {
	n := max(nlimit(line-p.pos.Line), min)
	if n > 0 {
		p.print(ws)
		if newSection {
			p.print(formfeed)
			n--
			nbreaks = 2
		}
		nbreaks += n
		for ; n > 0; n-- {
			p.print(newline)
		}
	}
	return
}

// setComment sets g as the next comment if g != nil and if node comments
// are enabled - this mode is used when printing source code fragments such
// as exports only. It assumes that there is no pending comment in p.comments
// and at most one pending comment in the p.comment cache.
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
		p.comments = p.comments[0:1]
		// in debug mode, report error
		p.internalError("setComment found pending comments")
	}
	p.comments[0] = g
	p.cindex = 0
	// don't overwrite any pending comment in the p.comment cache
	// (there may be a pending comment when a line comment is
	// immediately followed by a lead comment with no other
	// tokens between)
	if p.commentOffset == infinity {
		p.nextComment() // get comment ready for use
	}
}

type exprListMode uint

const (
	commaTerm exprListMode = 1 << iota // list is optionally terminated by a comma
	noIndent                           // no extra indentation in multi-line lists
)

// If indent is set, a multi-line identifier list is indented after the
// first linebreak encountered.
func (p *printer) identList(list []*ast.Ident, indent bool) {
	// convert into an expression list so we can re-use exprList formatting
	xlist := make([]ast.Expr, len(list))
	for i, x := range list {
		xlist[i] = x
	}
	var mode exprListMode
	if !indent {
		mode = noIndent
	}
	p.exprList(token.NoPos, xlist, 1, mode, token.NoPos, false)
}

const filteredMsg = "contains filtered or unexported fields"

// Print a list of expressions. If the list spans multiple
// source lines, the original line breaks are respected between
// expressions.
//
// TODO(gri) Consider rewriting this to be independent of []ast.Expr
// so that we can use the algorithm for any kind of list
//
//	(e.g., pass list via a channel over which to range).
func (p *printer) exprList(prev0 token.Pos, list []ast.Expr, depth int, mode exprListMode, next0 token.Pos, isIncomplete bool) {
	if len(list) == 0 {
		if isIncomplete {
			prev := p.posFor(prev0)
			next := p.posFor(next0)
			if prev.IsValid() && prev.Line == next.Line {
				p.print("/* " + filteredMsg + " */")
			} else {
				p.print(newline)
				p.print(indent, "// "+filteredMsg, unindent, newline)
			}
		}
		return
	}

	prev := p.posFor(prev0)
	next := p.posFor(next0)
	line := p.lineFor(list[0].Pos())
	endLine := p.lineFor(list[len(list)-1].End())

	if prev.IsValid() && prev.Line == line && line == endLine {
		// all list entries on a single line
		for i, x := range list {
			if i > 0 {
				// use position of expression following the comma as
				// comma position for correct comment placement
				p.setPos(x.Pos())
				p.print(token.COMMA, blank)
			}
			p.expr0(x, depth)
		}
		if isIncomplete {
			p.print(token.COMMA, blank, "/* "+filteredMsg+" */")
		}
		return
	}

	// list entries span multiple lines;
	// use source code positions to guide line breaks

	// Don't add extra indentation if noIndent is set;
	// i.e., pretend that the first line is already indented.
	ws := ignore
	if mode&noIndent == 0 {
		ws = indent
	}

	// The first linebreak is always a formfeed since this section must not
	// depend on any previous formatting.
	prevBreak := -1 // index of last expression that was followed by a linebreak
	if prev.IsValid() && prev.Line < line && p.linebreak(line, 0, ws, true) > 0 {
		ws = ignore
		prevBreak = 0
	}

	// initialize expression/key size: a zero value indicates expr/key doesn't fit on a single line
	size := 0

	// We use the ratio between the geometric mean of the previous key sizes and
	// the current size to determine if there should be a break in the alignment.
	// To compute the geometric mean we accumulate the ln(size) values (lnsum)
	// and the number of sizes included (count).
	lnsum := 0.0
	count := 0

	// print all list elements
	prevLine := prev.Line
	for i, x := range list {
		line = p.lineFor(x.Pos())

		// Determine if the next linebreak, if any, needs to use formfeed:
		// in general, use the entire node size to make the decision; for
		// key:value expressions, use the key size.
		// TODO(gri) for a better result, should probably incorporate both
		//           the key and the node size into the decision process
		useFF := true

		// Determine element size: All bets are off if we don't have
		// position information for the previous and next token (likely
		// generated code - simply ignore the size in this case by setting
		// it to 0).
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

		// If the previous line and the current line had single-
		// line-expressions and the key sizes are small or the
		// ratio between the current key and the geometric mean
		// if the previous key sizes does not exceed a threshold,
		// align columns and do not use formfeed.
		if prevSize > 0 && size > 0 {
			const smallSize = 40
			if count == 0 || prevSize <= smallSize && size <= smallSize {
				useFF = false
			} else {
				const r = 2.5                               // threshold
				geomean := math.Exp(lnsum / float64(count)) // count > 0
				ratio := float64(size) / geomean
				useFF = r*ratio <= 1 || r <= ratio
			}
		}

		needsLinebreak := 0 < prevLine && prevLine < line
		if i > 0 {
			// Use position of expression following the comma as
			// comma position for correct comment placement, but
			// only if the expression is on the same line.
			if !needsLinebreak {
				p.setPos(x.Pos())
			}
			p.print(token.COMMA)
			needsBlank := true
			if needsLinebreak {
				// Lines are broken using newlines so comments remain aligned
				// unless useFF is set or there are multiple expressions on
				// the same line in which case formfeed is used.
				nbreaks := p.linebreak(line, 0, ws, useFF || prevBreak+1 < i)
				if nbreaks > 0 {
					ws = ignore
					prevBreak = i
					needsBlank = false // we got a line break instead
				}
				// If there was a new section or more than one new line
				// (which means that the tabwriter will implicitly break
				// the section), reset the geomean variables since we are
				// starting a new group of elements with the next element.
				if nbreaks > 1 {
					lnsum = 0
					count = 0
				}
			}
			if needsBlank {
				p.print(blank)
			}
		}

		if len(list) > 1 && isPair && size > 0 && needsLinebreak {
			// We have a key:value expression that fits onto one line
			// and it's not on the same line as the prior expression:
			// Use a column for the key such that consecutive entries
			// can align if possible.
			// (needsLinebreak is set if we started a new line before)
			p.expr(pair.Key)
			p.setPos(pair.Colon)
			p.print(token.COLON, vtab)
			p.expr(pair.Value)
		} else {
			p.expr0(x, depth)
		}

		if size > 0 {
			lnsum += math.Log(float64(size))
			count++
		}

		prevLine = line
	}

	if mode&commaTerm != 0 && next.IsValid() && p.pos.Line < next.Line {
		// Print a terminating comma if the next token is on a new line.
		p.print(token.COMMA)
		if isIncomplete {
			p.print(newline)
			p.print("// " + filteredMsg)
		}
		if ws == ignore && mode&noIndent == 0 {
			// unindent if we indented
			p.print(unindent)
		}
		p.print(formfeed) // terminating comma needs a line break to look good
		return
	}

	if isIncomplete {
		p.print(token.COMMA, newline)
		p.print("// "+filteredMsg, newline)
	}

	if ws == ignore && mode&noIndent == 0 {
		// unindent if we indented
		p.print(unindent)
	}
}

type paramMode int

const (
	funcParam paramMode = iota
	funcTParam
	typeTParam
)

func (p *printer) parameters(fields *ast.FieldList, mode paramMode) {
	openTok, closeTok := token.LPAREN, token.RPAREN
	if mode != funcParam {
		openTok, closeTok = token.LBRACK, token.RBRACK
	}
	p.setPos(fields.Opening)
	p.print(openTok)
	if len(fields.List) > 0 {
		prevLine := p.lineFor(fields.Opening)
		ws := indent
		for i, par := range fields.List {
			// determine par begin and end line (may be different
			// if there are multiple parameter names for this par
			// or the type is on a separate line)
			parLineBeg := p.lineFor(par.Pos())
			parLineEnd := p.lineFor(par.End())
			// separating "," if needed
			needsLinebreak := 0 < prevLine && prevLine < parLineBeg
			if i > 0 {
				// use position of parameter following the comma as
				// comma position for correct comma placement, but
				// only if the next parameter is on the same line
				if !needsLinebreak {
					p.setPos(par.Pos())
				}
				p.print(token.COMMA)
			}
			// separator if needed (linebreak or blank)
			if needsLinebreak && p.linebreak(parLineBeg, 0, ws, true) > 0 {
				// break line if the opening "(" or previous parameter ended on a different line
				ws = ignore
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
				p.identList(par.Names, ws == indent)
				p.print(blank)
			}
			// parameter type
			p.expr(stripParensAlways(par.Type))
			prevLine = parLineEnd
		}

		// if the closing ")" is on a separate line from the last parameter,
		// print an additional "," and line break
		if closing := p.lineFor(fields.Closing); 0 < prevLine && prevLine < closing {
			p.print(token.COMMA)
			p.linebreak(closing, 0, ignore, true)
		} else if mode == typeTParam && fields.NumFields() == 1 && combinesWithName(stripParensAlways(fields.List[0].Type)) {
			// A type parameter list [P T] where the name P and the type expression T syntactically
			// combine to another valid (value) expression requires a trailing comma, as in [P *T,]
			// (or an enclosing interface as in [P interface(*T)]), so that the type parameter list
			// is not parsed as an array length [P*T].
			p.print(token.COMMA)
		}

		// unindent if we indented
		if ws == ignore {
			p.print(unindent)
		}
	}

	p.setPos(fields.Closing)
	p.print(closeTok)
}

// combinesWithName reports whether a name followed by the expression x
// syntactically combines to another valid (value) expression. For instance
// using *T for x, "name *T" syntactically appears as the expression x*T.
// On the other hand, using  P|Q or *P|~Q for x, "name P|Q" or "name *P|~Q"
// cannot be combined into a valid (value) expression.
func combinesWithName(x ast.Expr) bool {
	switch x := x.(type) {
	case *ast.StarExpr:
		// name *x.X combines to name*x.X if x.X is not a type element
		return !isTypeElem(x.X)
	case *ast.BinaryExpr:
		return combinesWithName(x.X) && !isTypeElem(x.Y)
	case *ast.ParenExpr:
		return !isTypeElem(x.X)
	}
	return false
}

// isTypeElem reports whether x is a (possibly parenthesized) type element expression.
// The result is false if x could be a type element OR an ordinary (value) expression.
func isTypeElem(x ast.Expr) bool {
	switch x := x.(type) {
	case *ast.ArrayType, *ast.StructType, *ast.FuncType, *ast.InterfaceType, *ast.MapType, *ast.ChanType:
		return true
	case *ast.UnaryExpr:
		return x.Op == token.TILDE
	case *ast.BinaryExpr:
		return isTypeElem(x.X) || isTypeElem(x.Y)
	case *ast.ParenExpr:
		return isTypeElem(x.X)
	}
	return false
}

func (p *printer) signature(sig *ast.FuncType) {
	if sig.TypeParams != nil {
		p.parameters(sig.TypeParams, funcTParam)
	}
	if sig.Params != nil {
		p.parameters(sig.Params, funcParam)
	} else {
		p.print(token.LPAREN, token.RPAREN)
	}
	res := sig.Results
	n := res.NumFields()
	if n > 0 {
		// res != nil
		p.print(blank)
		if n == 1 && res.List[0].Names == nil {
			// single anonymous res; no ()'s
			p.expr(stripParensAlways(res.List[0].Type))
			return
		}
		p.parameters(res, funcParam)
	}
}

func identListSize(list []*ast.Ident, maxSize int) (size int) {
	for i, x := range list {
		if i > 0 {
			size += len(", ")
		}
		size += utf8.RuneCountInString(x.Name)
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
	p.setComment(&ast.CommentGroup{List: []*ast.Comment{{Slash: token.NoPos, Text: text}}})
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
			p.setPos(lbrace)
			p.print(token.LBRACE)
			p.setPos(rbrace)
			p.print(token.RBRACE)
			return
		} else if p.isOneLineFieldList(list) {
			// small enough - print on one line
			// (don't use identList and ignore source line breaks)
			p.setPos(lbrace)
			p.print(token.LBRACE, blank)
			f := list[0]
			if isStruct {
				for i, x := range f.Names {
					if i > 0 {
						// no comments so no need for comma position
						p.print(token.COMMA, blank)
					}
					p.expr(x)
				}
				if len(f.Names) > 0 {
					p.print(blank)
				}
				p.expr(f.Type)
			} else { // interface
				if len(f.Names) > 0 {
					name := f.Names[0] // method name
					p.expr(name)
					p.signature(f.Type.(*ast.FuncType)) // don't print "func"
				} else {
					// embedded interface
					p.expr(f.Type)
				}
			}
			p.print(blank)
			p.setPos(rbrace)
			p.print(token.RBRACE)
			return
		}
	}
	// hasComments || !srcIsOneLine

	p.print(blank)
	p.setPos(lbrace)
	p.print(token.LBRACE, indent)
	if hasComments || len(list) > 0 {
		p.print(formfeed)
	}

	if isStruct {

		sep := vtab
		if len(list) == 1 {
			sep = blank
		}
		var line int
		for i, f := range list {
			if i > 0 {
				p.linebreak(p.lineFor(f.Pos()), 1, ignore, p.linesFrom(line) > 0)
			}
			extraTabs := 0
			p.setComment(f.Doc)
			p.recordLine(&line)
			if len(f.Names) > 0 {
				// named fields
				p.identList(f.Names, false)
				p.print(sep)
				p.expr(f.Type)
				extraTabs = 1
			} else {
				// anonymous field
				p.expr(f.Type)
				extraTabs = 2
			}
			if f.Tag != nil {
				if len(f.Names) > 0 && sep == vtab {
					p.print(sep)
				}
				p.print(sep)
				p.expr(f.Tag)
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
			p.setLineComment("// " + filteredMsg)
		}

	} else { // interface

		var line int
		var prev *ast.Ident // previous "type" identifier
		for i, f := range list {
			var name *ast.Ident // first name, or nil
			if len(f.Names) > 0 {
				name = f.Names[0]
			}
			if i > 0 {
				// don't do a line break (min == 0) if we are printing a list of types
				// TODO(gri) this doesn't work quite right if the list of types is
				//           spread across multiple lines
				min := 1
				if prev != nil && name == prev {
					min = 0
				}
				p.linebreak(p.lineFor(f.Pos()), min, ignore, p.linesFrom(line) > 0)
			}
			p.setComment(f.Doc)
			p.recordLine(&line)
			if name != nil {
				// method
				p.expr(name)
				p.signature(f.Type.(*ast.FuncType)) // don't print "func"
				prev = nil
			} else {
				// embedded interface
				p.expr(f.Type)
				prev = nil
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
	p.print(unindent, formfeed)
	p.setPos(rbrace)
	p.print(token.RBRACE)
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
		maxProblem = max(maxProblem, mp)
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
		maxProblem = max(maxProblem, mp)

	case *ast.StarExpr:
		if e.Op == token.QUO { // `*/`
			maxProblem = 5
		}

	case *ast.UnaryExpr:
		switch e.Op.String() + r.Op.String() {
		case "/*", "&&", "&^":
			maxProblem = 5
		case "++", "--":
			maxProblem = max(maxProblem, 4)
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
//
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
//  1. If there is a binary operator with a right side unary operand
//     that would clash without a space, the cutoff must be (in order):
//
//     /*	6
//     &&	6
//     &^	6
//     ++	5
//     --	5
//
//     (Comparison operators always have spaces around them.)
//
//  2. If there is a mix of level 5 and level 4 operators, then the cutoff
//     is 5 (use spaces to distinguish precedence) in Normal mode
//     and 4 (never use spaces) in Compact mode.
//
//  3. If there are no level 4 operators or no level 5 operators, then the
//     cutoff is 6 (always use spaces) in Normal mode
//     and 4 (never use spaces) in Compact mode.
func (p *printer) binaryExpr(x *ast.BinaryExpr, prec1, cutoff, depth int) {
	prec := x.Op.Precedence()
	if prec < prec1 {
		// parenthesis needed
		// Note: The parser inserts an ast.ParenExpr node; thus this case
		//       can only occur if the AST is created in a different way.
		p.print(token.LPAREN)
		p.expr0(x, reduceDepth(depth)) // parentheses undo one level of depth
		p.print(token.RPAREN)
		return
	}

	printBlank := prec < cutoff

	ws := indent
	p.expr1(x.X, prec, depth+diffPrec(x.X, prec))
	if printBlank {
		p.print(blank)
	}
	xline := p.pos.Line // before the operator (it may be on the next line!)
	yline := p.lineFor(x.Y.Pos())
	p.setPos(x.OpPos)
	p.print(x.Op)
	if xline != yline && xline > 0 && yline > 0 {
		// at least one line break, but respect an extra empty line
		// in the source
		if p.linebreak(yline, 1, ws, true) > 0 {
			ws = ignore
			printBlank = false // no blank after line break
		}
	}
	if printBlank {
		p.print(blank)
	}
	p.expr1(x.Y, prec+1, depth+1)
	if ws == ignore {
		p.print(unindent)
	}
}

func isBinary(expr ast.Expr) bool {
	_, ok := expr.(*ast.BinaryExpr)
	return ok
}

func (p *printer) expr1(expr ast.Expr, prec1, depth int) {
	p.setPos(expr.Pos())

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
		p.binaryExpr(x, prec1, cutoff(x, depth), depth)

	case *ast.KeyValueExpr:
		p.expr(x.Key)
		p.setPos(x.Colon)
		p.print(token.COLON, blank)
		p.expr(x.Value)

	case *ast.StarExpr:
		const prec = token.UnaryPrec
		if prec < prec1 {
			// parenthesis needed
			p.print(token.LPAREN)
			p.print(token.MUL)
			p.expr(x.X)
			p.print(token.RPAREN)
		} else {
			// no parenthesis needed
			p.print(token.MUL)
			p.expr(x.X)
		}

	case *ast.UnaryExpr:
		const prec = token.UnaryPrec
		if prec < prec1 {
			// parenthesis needed
			p.print(token.LPAREN)
			p.expr(x)
			p.print(token.RPAREN)
		} else {
			// no parenthesis needed
			p.print(x.Op)
			if x.Op == token.RANGE {
				// TODO(gri) Remove this code if it cannot be reached.
				p.print(blank)
			}
			p.expr1(x.X, prec, depth)
		}

	case *ast.BasicLit:
		if p.Config.Mode&normalizeNumbers != 0 {
			x = normalizedNumber(x)
		}
		p.print(x)

	case *ast.FuncLit:
		p.setPos(x.Type.Pos())
		p.print(token.FUNC)
		// See the comment in funcDecl about how the header size is computed.
		startCol := p.out.Column - len("func")
		p.signature(x.Type)
		p.funcBody(p.distanceFrom(x.Type.Pos(), startCol), blank, x.Body)

	case *ast.ParenExpr:
		if _, hasParens := x.X.(*ast.ParenExpr); hasParens {
			// don't print parentheses around an already parenthesized expression
			// TODO(gri) consider making this more general and incorporate precedence levels
			p.expr0(x.X, depth)
		} else {
			p.print(token.LPAREN)
			p.expr0(x.X, reduceDepth(depth)) // parentheses undo one level of depth
			p.setPos(x.Rparen)
			p.print(token.RPAREN)
		}

	case *ast.SelectorExpr:
		p.selectorExpr(x, depth, false)

	case *ast.TypeAssertExpr:
		p.expr1(x.X, token.HighestPrec, depth)
		p.print(token.PERIOD)
		p.setPos(x.Lparen)
		p.print(token.LPAREN)
		if x.Type != nil {
			p.expr(x.Type)
		} else {
			p.print(token.TYPE)
		}
		p.setPos(x.Rparen)
		p.print(token.RPAREN)

	case *ast.IndexExpr:
		// TODO(gri): should treat[] like parentheses and undo one level of depth
		p.expr1(x.X, token.HighestPrec, 1)
		p.setPos(x.Lbrack)
		p.print(token.LBRACK)
		p.expr0(x.Index, depth+1)
		p.setPos(x.Rbrack)
		p.print(token.RBRACK)

	case *ast.IndexListExpr:
		// TODO(gri): as for IndexExpr, should treat [] like parentheses and undo
		// one level of depth
		p.expr1(x.X, token.HighestPrec, 1)
		p.setPos(x.Lbrack)
		p.print(token.LBRACK)
		p.exprList(x.Lbrack, x.Indices, depth+1, commaTerm, x.Rbrack, false)
		p.setPos(x.Rbrack)
		p.print(token.RBRACK)

	case *ast.SliceExpr:
		// TODO(gri): should treat[] like parentheses and undo one level of depth
		p.expr1(x.X, token.HighestPrec, 1)
		p.setPos(x.Lbrack)
		p.print(token.LBRACK)
		indices := []ast.Expr{x.Low, x.High}
		if x.Max != nil {
			indices = append(indices, x.Max)
		}
		// determine if we need extra blanks around ':'
		var needsBlanks bool
		if depth <= 1 {
			var indexCount int
			var hasBinaries bool
			for _, x := range indices {
				if x != nil {
					indexCount++
					if isBinary(x) {
						hasBinaries = true
					}
				}
			}
			if indexCount > 1 && hasBinaries {
				needsBlanks = true
			}
		}
		for i, x := range indices {
			if i > 0 {
				if indices[i-1] != nil && needsBlanks {
					p.print(blank)
				}
				p.print(token.COLON)
				if x != nil && needsBlanks {
					p.print(blank)
				}
			}
			if x != nil {
				p.expr0(x, depth+1)
			}
		}
		p.setPos(x.Rbrack)
		p.print(token.RBRACK)

	case *ast.CallExpr:
		if len(x.Args) > 1 {
			depth++
		}

		// Conversions to literal function types or <-chan
		// types require parentheses around the type.
		paren := false
		switch t := x.Fun.(type) {
		case *ast.FuncType:
			paren = true
		case *ast.ChanType:
			paren = t.Dir == ast.RECV
		}
		if paren {
			p.print(token.LPAREN)
		}
		wasIndented := p.possibleSelectorExpr(x.Fun, token.HighestPrec, depth)
		if paren {
			p.print(token.RPAREN)
		}

		p.setPos(x.Lparen)
		p.print(token.LPAREN)
		if x.Ellipsis.IsValid() {
			p.exprList(x.Lparen, x.Args, depth, 0, x.Ellipsis, false)
			p.setPos(x.Ellipsis)
			p.print(token.ELLIPSIS)
			if x.Rparen.IsValid() && p.lineFor(x.Ellipsis) < p.lineFor(x.Rparen) {
				p.print(token.COMMA, formfeed)
			}
		} else {
			p.exprList(x.Lparen, x.Args, depth, commaTerm, x.Rparen, false)
		}
		p.setPos(x.Rparen)
		p.print(token.RPAREN)
		if wasIndented {
			p.print(unindent)
		}

	case *ast.CompositeLit:
		// composite literal elements that are composite literals themselves may have the type omitted
		if x.Type != nil {
			p.expr1(x.Type, token.HighestPrec, depth)
		}
		p.level++
		p.setPos(x.Lbrace)
		p.print(token.LBRACE)
		p.exprList(x.Lbrace, x.Elts, 1, commaTerm, x.Rbrace, x.Incomplete)
		// do not insert extra line break following a /*-style comment
		// before the closing '}' as it might break the code if there
		// is no trailing ','
		mode := noExtraLinebreak
		// do not insert extra blank following a /*-style comment
		// before the closing '}' unless the literal is empty
		if len(x.Elts) > 0 {
			mode |= noExtraBlank
		}
		// need the initial indent to print lone comments with
		// the proper level of indentation
		p.print(indent, unindent, mode)
		p.setPos(x.Rbrace)
		p.print(token.RBRACE, mode)
		p.level--

	case *ast.Ellipsis:
		p.print(token.ELLIPSIS)
		if x.Elt != nil {
			p.expr(x.Elt)
		}

	case *ast.ArrayType:
		p.print(token.LBRACK)
		if x.Len != nil {
			p.expr(x.Len)
		}
		p.print(token.RBRACK)
		p.expr(x.Elt)

	case *ast.StructType:
		p.print(token.STRUCT)
		p.fieldList(x.Fields, true, x.Incomplete)

	case *ast.FuncType:
		p.print(token.FUNC)
		p.signature(x)

	case *ast.InterfaceType:
		p.print(token.INTERFACE)
		p.fieldList(x.Methods, false, x.Incomplete)

	case *ast.MapType:
		p.print(token.MAP, token.LBRACK)
		p.expr(x.Key)
		p.print(token.RBRACK)
		p.expr(x.Value)

	case *ast.ChanType:
		switch x.Dir {
		case ast.SEND | ast.RECV:
			p.print(token.CHAN)
		case ast.RECV:
			p.print(token.ARROW, token.CHAN) // x.Arrow and x.Pos() are the same
		case ast.SEND:
			p.print(token.CHAN)
			p.setPos(x.Arrow)
			p.print(token.ARROW)
		}
		p.print(blank)
		p.expr(x.Value)

	default:
		panic("unreachable")
	}
}

// normalizedNumber rewrites base prefixes and exponents
// of numbers to use lower-case letters (0X123 to 0x123 and 1.2E3 to 1.2e3),
// and removes leading 0's from integer imaginary literals (0765i to 765i).
// It leaves hexadecimal digits alone.
//
// normalizedNumber doesn't modify the ast.BasicLit value lit points to.
// If lit is not a number or a number in canonical format already,
// lit is returned as is. Otherwise a new ast.BasicLit is created.
func normalizedNumber(lit *ast.BasicLit) *ast.BasicLit {
	if lit.Kind != token.INT && lit.Kind != token.FLOAT && lit.Kind != token.IMAG {
		return lit // not a number - nothing to do
	}
	if len(lit.Value) < 2 {
		return lit // only one digit (common case) - nothing to do
	}
	// len(lit.Value) >= 2

	// We ignore lit.Kind because for lit.Kind == token.IMAG the literal may be an integer
	// or floating-point value, decimal or not. Instead, just consider the literal pattern.
	x := lit.Value
	switch x[:2] {
	default:
		// 0-prefix octal, decimal int, or float (possibly with 'i' suffix)
		if i := strings.LastIndexByte(x, 'E'); i >= 0 {
			x = x[:i] + "e" + x[i+1:]
			break
		}
		// remove leading 0's from integer (but not floating-point) imaginary literals
		if x[len(x)-1] == 'i' && !strings.ContainsAny(x, ".e") {
			x = strings.TrimLeft(x, "0_")
			if x == "i" {
				x = "0i"
			}
		}
	case "0X":
		x = "0x" + x[2:]
		// possibly a hexadecimal float
		if i := strings.LastIndexByte(x, 'P'); i >= 0 {
			x = x[:i] + "p" + x[i+1:]
		}
	case "0x":
		// possibly a hexadecimal float
		i := strings.LastIndexByte(x, 'P')
		if i == -1 {
			return lit // nothing to do
		}
		x = x[:i] + "p" + x[i+1:]
	case "0O":
		x = "0o" + x[2:]
	case "0o":
		return lit // nothing to do
	case "0B":
		x = "0b" + x[2:]
	case "0b":
		return lit // nothing to do
	}

	return &ast.BasicLit{ValuePos: lit.ValuePos, Kind: lit.Kind, Value: x}
}

func (p *printer) possibleSelectorExpr(expr ast.Expr, prec1, depth int) bool {
	if x, ok := expr.(*ast.SelectorExpr); ok {
		return p.selectorExpr(x, depth, true)
	}
	p.expr1(expr, prec1, depth)
	return false
}

// selectorExpr handles an *ast.SelectorExpr node and reports whether x spans
// multiple lines.
func (p *printer) selectorExpr(x *ast.SelectorExpr, depth int, isMethod bool) bool {
	p.expr1(x.X, token.HighestPrec, depth)

	// We don't have the position of the dot, so we have to predict it,
	// to avoid issues with comment handling.
	// See https://go.dev/issue/70978 and TestIssue70978 for more details.
	if x.Sel.Pos().IsValid() && p.pos.Offset > p.posFor(x.Sel.Pos()).Offset {
		p.setPos(x.Sel.Pos())
	}

	p.print(token.PERIOD)
	if line := p.lineFor(x.Sel.Pos()); p.pos.IsValid() && p.pos.Line < line {
		p.print(indent, newline)
		p.setPos(x.Sel.Pos())
		p.print(x.Sel)
		if !isMethod {
			p.print(unindent)
		}
		return true
	}
	p.setPos(x.Sel.Pos())
	p.print(x.Sel)
	return false
}

func (p *printer) expr0(x ast.Expr, depth int) {
	p.expr1(x, token.LowestPrec, depth)
}

func (p *printer) expr(x ast.Expr) {
	const depth = 1
	p.expr1(x, token.LowestPrec, depth)
}

// ----------------------------------------------------------------------------
// Statements

// Print the statement list indented, but without a newline after the last statement.
// Extra line breaks between statements in the source are respected but at most one
// empty line is printed between statements.
func (p *printer) stmtList(list []ast.Stmt, nindent int, nextIsRBrace bool) {
	if nindent > 0 {
		p.print(indent)
	}
	var line int
	i := 0
	for _, s := range list {
		// ignore empty statements (was issue 3466)
		if _, isEmpty := s.(*ast.EmptyStmt); !isEmpty {
			// nindent == 0 only for lists of switch/select case clauses;
			// in those cases each clause is a new section
			if len(p.output) > 0 {
				// only print line break if we are not at the beginning of the output
				// (i.e., we are not printing only a partial program)
				p.linebreak(p.lineFor(s.Pos()), 1, ignore, i == 0 || nindent == 0 || p.linesFrom(line) > 0)
			}
			p.recordLine(&line)
			p.stmt(s, nextIsRBrace && i == len(list)-1)
			// labeled statements put labels on a separate line, but here
			// we only care about the start line of the actual statement
			// without label - correct line for each label
			for t := s; ; {
				lt, _ := t.(*ast.LabeledStmt)
				if lt == nil {
					break
				}
				line++
				t = lt.Stmt
			}
			i++
		}
	}
	if nindent > 0 {
		p.print(unindent)
	}
}

// block prints an *ast.BlockStmt; it always spans at least two lines.
func (p *printer) block(b *ast.BlockStmt, nindent int) {
	p.setPos(b.Lbrace)
	p.print(token.LBRACE)
	p.stmtList(b.List, nindent, true)
	p.linebreak(p.lineFor(b.Rbrace), 1, ignore, true)
	p.setPos(b.Rbrace)
	p.print(token.RBRACE)
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

func stripParensAlways(x ast.Expr) ast.Expr {
	if x, ok := x.(*ast.ParenExpr); ok {
		return stripParensAlways(x.X)
	}
	return x
}

func (p *printer) controlClause(isForStmt bool, init ast.Stmt, expr ast.Expr, post ast.Stmt) {
	p.print(blank)
	needsBlank := false
	if init == nil && post == nil {
		// no semicolons required
		if expr != nil {
			p.expr(stripParens(expr))
			needsBlank = true
		}
	} else {
		// all semicolons required
		// (they are not separators, print them explicitly)
		if init != nil {
			p.stmt(init, false)
		}
		p.print(token.SEMICOLON, blank)
		if expr != nil {
			p.expr(stripParens(expr))
			needsBlank = true
		}
		if isForStmt {
			p.print(token.SEMICOLON, blank)
			needsBlank = false
			if post != nil {
				p.stmt(post, false)
				needsBlank = true
			}
		}
	}
	if needsBlank {
		p.print(blank)
	}
}

// indentList reports whether an expression list would look better if it
// were indented wholesale (starting with the very first element, rather
// than starting at the first line break).
func (p *printer) indentList(list []ast.Expr) bool {
	// Heuristic: indentList reports whether there are more than one multi-
	// line element in the list, or if there is any element that is not
	// starting on the same line as the previous one ends.
	if len(list) >= 2 {
		var b = p.lineFor(list[0].Pos())
		var e = p.lineFor(list[len(list)-1].End())
		if 0 < b && b < e {
			// list spans multiple lines
			n := 0 // multi-line element count
			line := b
			for _, x := range list {
				xb := p.lineFor(x.Pos())
				xe := p.lineFor(x.End())
				if line < xb {
					// x is not starting on the same
					// line as the previous one ended
					return true
				}
				if xb < xe {
					// x is a multi-line element
					n++
				}
				line = xe
			}
			return n > 1
		}
	}
	return false
}

func (p *printer) stmt(stmt ast.Stmt, nextIsRBrace bool) {
	p.setPos(stmt.Pos())

	switch s := stmt.(type) {
	case *ast.BadStmt:
		p.print("BadStmt")

	case *ast.DeclStmt:
		p.decl(s.Decl)

	case *ast.EmptyStmt:
		// nothing to do

	case *ast.LabeledStmt:
		// a "correcting" unindent immediately following a line break
		// is applied before the line break if there is no comment
		// between (see writeWhitespace)
		p.print(unindent)
		p.expr(s.Label)
		p.setPos(s.Colon)
		p.print(token.COLON, indent)
		if e, isEmpty := s.Stmt.(*ast.EmptyStmt); isEmpty {
			if !nextIsRBrace {
				p.print(newline)
				p.setPos(e.Pos())
				p.print(token.SEMICOLON)
				break
			}
		} else {
			p.linebreak(p.lineFor(s.Stmt.Pos()), 1, ignore, true)
		}
		p.stmt(s.Stmt, nextIsRBrace)

	case *ast.ExprStmt:
		const depth = 1
		p.expr0(s.X, depth)

	case *ast.SendStmt:
		const depth = 1
		p.expr0(s.Chan, depth)
		p.print(blank)
		p.setPos(s.Arrow)
		p.print(token.ARROW, blank)
		p.expr0(s.Value, depth)

	case *ast.IncDecStmt:
		const depth = 1
		p.expr0(s.X, depth+1)
		p.setPos(s.TokPos)
		p.print(s.Tok)

	case *ast.AssignStmt:
		var depth = 1
		if len(s.Lhs) > 1 && len(s.Rhs) > 1 {
			depth++
		}
		p.exprList(s.Pos(), s.Lhs, depth, 0, s.TokPos, false)
		p.print(blank)
		p.setPos(s.TokPos)
		p.print(s.Tok, blank)
		p.exprList(s.TokPos, s.Rhs, depth, 0, token.NoPos, false)

	case *ast.GoStmt:
		p.print(token.GO, blank)
		p.expr(s.Call)

	case *ast.DeferStmt:
		p.print(token.DEFER, blank)
		p.expr(s.Call)

	case *ast.ReturnStmt:
		p.print(token.RETURN)
		if s.Results != nil {
			p.print(blank)
			// Use indentList heuristic to make corner cases look
			// better (issue 1207). A more systematic approach would
			// always indent, but this would cause significant
			// reformatting of the code base and not necessarily
			// lead to more nicely formatted code in general.
			if p.indentList(s.Results) {
				p.print(indent)
				// Use NoPos so that a newline never goes before
				// the results (see issue #32854).
				p.exprList(token.NoPos, s.Results, 1, noIndent, token.NoPos, false)
				p.print(unindent)
			} else {
				p.exprList(token.NoPos, s.Results, 1, 0, token.NoPos, false)
			}
		}

	case *ast.BranchStmt:
		p.print(s.Tok)
		if s.Label != nil {
			p.print(blank)
			p.expr(s.Label)
		}

	case *ast.BlockStmt:
		p.block(s, 1)

	case *ast.IfStmt:
		p.print(token.IF)
		p.controlClause(false, s.Init, s.Cond, nil)
		p.block(s.Body, 1)
		if s.Else != nil {
			p.print(blank, token.ELSE, blank)
			switch s.Else.(type) {
			case *ast.BlockStmt, *ast.IfStmt:
				p.stmt(s.Else, nextIsRBrace)
			default:
				// This can only happen with an incorrectly
				// constructed AST. Permit it but print so
				// that it can be parsed without errors.
				p.print(token.LBRACE, indent, formfeed)
				p.stmt(s.Else, true)
				p.print(unindent, formfeed, token.RBRACE)
			}
		}

	case *ast.CaseClause:
		if s.List != nil {
			p.print(token.CASE, blank)
			p.exprList(s.Pos(), s.List, 1, 0, s.Colon, false)
		} else {
			p.print(token.DEFAULT)
		}
		p.setPos(s.Colon)
		p.print(token.COLON)
		p.stmtList(s.Body, 1, nextIsRBrace)

	case *ast.SwitchStmt:
		p.print(token.SWITCH)
		p.controlClause(false, s.Init, s.Tag, nil)
		p.block(s.Body, 0)

	case *ast.TypeSwitchStmt:
		p.print(token.SWITCH)
		if s.Init != nil {
			p.print(blank)
			p.stmt(s.Init, false)
			p.print(token.SEMICOLON)
		}
		p.print(blank)
		p.stmt(s.Assign, false)
		p.print(blank)
		p.block(s.Body, 0)

	case *ast.CommClause:
		if s.Comm != nil {
			p.print(token.CASE, blank)
			p.stmt(s.Comm, false)
		} else {
			p.print(token.DEFAULT)
		}
		p.setPos(s.Colon)
		p.print(token.COLON)
		p.stmtList(s.Body, 1, nextIsRBrace)

	case *ast.SelectStmt:
		p.print(token.SELECT, blank)
		body := s.Body
		if len(body.List) == 0 && !p.commentBefore(p.posFor(body.Rbrace)) {
			// print empty select statement w/o comments on one line
			p.setPos(body.Lbrace)
			p.print(token.LBRACE)
			p.setPos(body.Rbrace)
			p.print(token.RBRACE)
		} else {
			p.block(body, 0)
		}

	case *ast.ForStmt:
		p.print(token.FOR)
		p.controlClause(true, s.Init, s.Cond, s.Post)
		p.block(s.Body, 1)

	case *ast.RangeStmt:
		p.print(token.FOR, blank)
		if s.Key != nil {
			p.expr(s.Key)
			if s.Value != nil {
				// use position of value following the comma as
				// comma position for correct comment placement
				p.setPos(s.Value.Pos())
				p.print(token.COMMA, blank)
				p.expr(s.Value)
			}
			p.print(blank)
			p.setPos(s.TokPos)
			p.print(s.Tok, blank)
		}
		p.print(token.RANGE, blank)
		p.expr(stripParens(s.X))
		p.print(blank)
		p.block(s.Body, 1)

	default:
		panic("unreachable")
	}
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
//		const (
//			foobar int = 42 // comment
//			x          = 7  // comment
//			foo
//	             bar = 991
//		)
//
// leads to the type/values matrix below. A run of value columns (V) can
// be moved into the type column if there is no type for any of the values
// in that column (we only move entire columns so that they align properly).
//
//		matrix        formatted     result
//	                   matrix
//		T  V    ->    T  V     ->   true      there is a T and so the type
//		-  V          -  V          true      column must be kept
//		-  -          -  -          false
//		-  V          V  -          false     V is moved into T column
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

func (p *printer) valueSpec(s *ast.ValueSpec, keepType bool) {
	p.setComment(s.Doc)
	p.identList(s.Names, false) // always present
	extraTabs := 3
	if s.Type != nil || keepType {
		p.print(vtab)
		extraTabs--
	}
	if s.Type != nil {
		p.expr(s.Type)
	}
	if s.Values != nil {
		p.print(vtab, token.ASSIGN, blank)
		p.exprList(token.NoPos, s.Values, 1, 0, token.NoPos, false)
		extraTabs--
	}
	if s.Comment != nil {
		for ; extraTabs > 0; extraTabs-- {
			p.print(vtab)
		}
		p.setComment(s.Comment)
	}
}

func sanitizeImportPath(lit *ast.BasicLit) *ast.BasicLit {
	// Note: An unmodified AST generated by go/parser will already
	// contain a backward- or double-quoted path string that does
	// not contain any invalid characters, and most of the work
	// here is not needed. However, a modified or generated AST
	// may possibly contain non-canonical paths. Do the work in
	// all cases since it's not too hard and not speed-critical.

	// if we don't have a proper string, be conservative and return whatever we have
	if lit.Kind != token.STRING {
		return lit
	}
	s, err := strconv.Unquote(lit.Value)
	if err != nil {
		return lit
	}

	// if the string is an invalid path, return whatever we have
	//
	// spec: "Implementation restriction: A compiler may restrict
	// ImportPaths to non-empty strings using only characters belonging
	// to Unicode's L, M, N, P, and S general categories (the Graphic
	// characters without spaces) and may also exclude the characters
	// !"#$%&'()*,:;<=>?[\]^`{|} and the Unicode replacement character
	// U+FFFD."
	if s == "" {
		return lit
	}
	const illegalChars = `!"#$%&'()*,:;<=>?[\]^{|}` + "`\uFFFD"
	for _, r := range s {
		if !unicode.IsGraphic(r) || unicode.IsSpace(r) || strings.ContainsRune(illegalChars, r) {
			return lit
		}
	}

	// otherwise, return the double-quoted path
	s = strconv.Quote(s)
	if s == lit.Value {
		return lit // nothing wrong with lit
	}
	return &ast.BasicLit{ValuePos: lit.ValuePos, Kind: token.STRING, Value: s}
}

// The parameter n is the number of specs in the group. If doIndent is set,
// multi-line identifier lists in the spec are indented when the first
// linebreak is encountered.
func (p *printer) spec(spec ast.Spec, n int, doIndent bool) {
	switch s := spec.(type) {
	case *ast.ImportSpec:
		p.setComment(s.Doc)
		if s.Name != nil {
			p.expr(s.Name)
			p.print(blank)
		}
		p.expr(sanitizeImportPath(s.Path))
		p.setComment(s.Comment)
		p.setPos(s.EndPos)

	case *ast.ValueSpec:
		if n != 1 {
			p.internalError("expected n = 1; got", n)
		}
		p.setComment(s.Doc)
		p.identList(s.Names, doIndent) // always present
		if s.Type != nil {
			p.print(blank)
			p.expr(s.Type)
		}
		if s.Values != nil {
			p.print(blank, token.ASSIGN, blank)
			p.exprList(token.NoPos, s.Values, 1, 0, token.NoPos, false)
		}
		p.setComment(s.Comment)

	case *ast.TypeSpec:
		p.setComment(s.Doc)
		p.expr(s.Name)
		if s.TypeParams != nil {
			p.parameters(s.TypeParams, typeTParam)
		}
		if n == 1 {
			p.print(blank)
		} else {
			p.print(vtab)
		}
		if s.Assign.IsValid() {
			p.print(token.ASSIGN, blank)
		}
		p.expr(s.Type)
		p.setComment(s.Comment)

	default:
		panic("unreachable")
	}
}

func (p *printer) genDecl(d *ast.GenDecl) {
	p.setComment(d.Doc)
	p.setPos(d.Pos())
	p.print(d.Tok, blank)

	if d.Lparen.IsValid() || len(d.Specs) != 1 {
		// group of parenthesized declarations
		p.setPos(d.Lparen)
		p.print(token.LPAREN)
		if n := len(d.Specs); n > 0 {
			p.print(indent, formfeed)
			if n > 1 && (d.Tok == token.CONST || d.Tok == token.VAR) {
				// two or more grouped const/var declarations:
				// determine if the type column must be kept
				keepType := keepTypeColumn(d.Specs)
				var line int
				for i, s := range d.Specs {
					if i > 0 {
						p.linebreak(p.lineFor(s.Pos()), 1, ignore, p.linesFrom(line) > 0)
					}
					p.recordLine(&line)
					p.valueSpec(s.(*ast.ValueSpec), keepType[i])
				}
			} else {
				var line int
				for i, s := range d.Specs {
					if i > 0 {
						p.linebreak(p.lineFor(s.Pos()), 1, ignore, p.linesFrom(line) > 0)
					}
					p.recordLine(&line)
					p.spec(s, n, false)
				}
			}
			p.print(unindent, formfeed)
		}
		p.setPos(d.Rparen)
		p.print(token.RPAREN)

	} else if len(d.Specs) > 0 {
		// single declaration
		p.spec(d.Specs[0], 1, true)
	}
}

// sizeCounter is an io.Writer which counts the number of bytes written,
// as well as whether a newline character was seen.
type sizeCounter struct {
	hasNewline bool
	size       int
}

func (c *sizeCounter) Write(p []byte) (int, error) {
	if !c.hasNewline {
		for _, b := range p {
			if b == '\n' || b == '\f' {
				c.hasNewline = true
				break
			}
		}
	}
	c.size += len(p)
	return len(p), nil
}

// nodeSize determines the size of n in chars after formatting.
// The result is <= maxSize if the node fits on one line with at
// most maxSize chars and the formatted output doesn't contain
// any control chars. Otherwise, the result is > maxSize.
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
	var counter sizeCounter
	if err := cfg.fprint(&counter, p.fset, n, p.nodeSizes); err != nil {
		return
	}
	if counter.size <= maxSize && !counter.hasNewline {
		// n fits in a single line
		size = counter.size
		p.nodeSizes[n] = size
	}
	return
}

// numLines returns the number of lines spanned by node n in the original source.
func (p *printer) numLines(n ast.Node) int {
	if from := n.Pos(); from.IsValid() {
		if to := n.End(); to.IsValid() {
			return p.lineFor(to) - p.lineFor(from) + 1
		}
	}
	return infinity
}

// bodySize is like nodeSize but it is specialized for *ast.BlockStmt's.
func (p *printer) bodySize(b *ast.BlockStmt, maxSize int) int {
	pos1 := b.Pos()
	pos2 := b.Rbrace
	if pos1.IsValid() && pos2.IsValid() && p.lineFor(pos1) != p.lineFor(pos2) {
		// opening and closing brace are on different lines - don't make it a one-liner
		return maxSize + 1
	}
	if len(b.List) > 5 {
		// too many statements - don't make it a one-liner
		return maxSize + 1
	}
	// otherwise, estimate body size
	bodySize := p.commentSizeBefore(p.posFor(pos2))
	for i, s := range b.List {
		if bodySize > maxSize {
			break // no need to continue
		}
		if i > 0 {
			bodySize += 2 // space for a semicolon and blank
		}
		bodySize += p.nodeSize(s, maxSize)
	}
	return bodySize
}

// funcBody prints a function body following a function header of given headerSize.
// If the header's and block's size are "small enough" and the block is "simple enough",
// the block is printed on the current line, without line breaks, spaced from the header
// by sep. Otherwise the block's opening "{" is printed on the current line, followed by
// lines for the block's statements and its closing "}".
func (p *printer) funcBody(headerSize int, sep whiteSpace, b *ast.BlockStmt) {
	if b == nil {
		return
	}

	// save/restore composite literal nesting level
	defer func(level int) {
		p.level = level
	}(p.level)
	p.level = 0

	const maxSize = 100
	if headerSize+p.bodySize(b, maxSize) <= maxSize {
		p.print(sep)
		p.setPos(b.Lbrace)
		p.print(token.LBRACE)
		if len(b.List) > 0 {
			p.print(blank)
			for i, s := range b.List {
				if i > 0 {
					p.print(token.SEMICOLON, blank)
				}
				p.stmt(s, i == len(b.List)-1)
			}
			p.print(blank)
		}
		p.print(noExtraLinebreak)
		p.setPos(b.Rbrace)
		p.print(token.RBRACE, noExtraLinebreak)
		return
	}

	if sep != ignore {
		p.print(blank) // always use blank
	}
	p.block(b, 1)
}

// distanceFrom returns the column difference between p.out (the current output
// position) and startOutCol. If the start position is on a different line from
// the current position (or either is unknown), the result is infinity.
func (p *printer) distanceFrom(startPos token.Pos, startOutCol int) int {
	if startPos.IsValid() && p.pos.IsValid() && p.posFor(startPos).Line == p.pos.Line {
		return p.out.Column - startOutCol
	}
	return infinity
}

func (p *printer) funcDecl(d *ast.FuncDecl) {
	p.setComment(d.Doc)
	p.setPos(d.Pos())
	p.print(token.FUNC, blank)
	// We have to save startCol only after emitting FUNC; otherwise it can be on a
	// different line (all whitespace preceding the FUNC is emitted only when the
	// FUNC is emitted).
	startCol := p.out.Column - len("func ")
	if d.Recv != nil {
		p.parameters(d.Recv, funcParam) // method: print receiver
		p.print(blank)
	}
	p.expr(d.Name)
	p.signature(d.Type)
	p.funcBody(p.distanceFrom(d.Pos(), startCol), vtab, d.Body)
}

func (p *printer) decl(decl ast.Decl) {
	switch d := decl.(type) {
	case *ast.BadDecl:
		p.setPos(d.Pos())
		p.print("BadDecl")
	case *ast.GenDecl:
		p.genDecl(d)
	case *ast.FuncDecl:
		p.funcDecl(d)
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

func (p *printer) declList(list []ast.Decl) {
	tok := token.ILLEGAL
	for _, d := range list {
		prev := tok
		tok = declToken(d)
		// If the declaration token changed (e.g., from CONST to TYPE)
		// or the next declaration has documentation associated with it,
		// print an empty line between top-level declarations.
		// (because p.linebreak is called with the position of d, which
		// is past any documentation, the minimum requirement is satisfied
		// even w/o the extra getDoc(d) nil-check - leave it in case the
		// linebreak logic improves - there's already a TODO).
		if len(p.output) > 0 {
			// only print line break if we are not at the beginning of the output
			// (i.e., we are not printing only a partial program)
			min := 1
			if prev != tok || getDoc(d) != nil {
				min = 2
			}
			// start a new section if the next declaration is a function
			// that spans multiple lines (see also issue #19544)
			p.linebreak(p.lineFor(d.Pos()), min, ignore, tok == token.FUNC && p.numLines(d) > 1)
		}
		p.decl(d)
	}
}

func (p *printer) file(src *ast.File) {
	p.setComment(src.Doc)
	p.setPos(src.Pos())
	p.print(token.PACKAGE, blank)
	p.expr(src.Name)
	p.declList(src.Decls)
	p.print(newline)
}
