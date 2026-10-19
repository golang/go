// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"go/token"
	"slices"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// This file applies the M1-S5 templates: every lowering builds the native
// statements and the eject-text twin from one plan, so the two agree by
// construction.

// twin collects the marker substitutions of one eject text.
type twin struct {
	subs []textSub
	seq  int
}

// mark returns a placeholder standing for text, positioned at pos so the
// printer sees known positions while rendering the eject text.
func (tw *twin) mark(pos syntax.Pos, text string) *syntax.Name {
	return markerText(&tw.subs, &tw.seq, pos, text)
}

// tryAssignOf matches lhs = f()? and lhs := f()?.
func tryAssignOf(s *syntax.AssignStmt) (*syntax.TryExpr, *syntax.CallExpr, bool) {
	if s.Op != syntax.Def && s.Op != 0 {
		return nil, nil, false
	}
	t, ok := syntax.Unparen(s.Rhs).(*syntax.TryExpr)
	if !ok {
		return nil, nil, false
	}
	call, ok := syntax.Unparen(t.X).(*syntax.CallExpr)
	if !ok {
		return nil, nil, false
	}
	return t, call, true
}

// tryExprOf matches f()? as an expression statement.
func tryExprOf(s *syntax.ExprStmt) (*syntax.TryExpr, *syntax.CallExpr, bool) {
	t, ok := syntax.Unparen(s.X).(*syntax.TryExpr)
	if !ok {
		return nil, nil, false
	}
	call, ok := syntax.Unparen(t.X).(*syntax.CallExpr)
	if !ok {
		return nil, nil, false
	}
	return t, call, true
}

// tryCallOf matches defer f()?.
func tryCallOf(s *syntax.CallStmt) (*syntax.TryExpr, *syntax.CallExpr, bool) {
	t, ok := syntax.Unparen(s.Call).(*syntax.TryExpr)
	if !ok {
		return nil, nil, false
	}
	call, ok := syntax.Unparen(t.X).(*syntax.CallExpr)
	if !ok {
		return nil, nil, false
	}
	return t, call, true
}

// callResults counts the non-error results of a checked ? call. A call
// returning only error has no tuple type and counts zero.
func (fc *funcLower) callResults(call *syntax.CallExpr) int {
	typ := fc.fl.pkg.info.TypeOf(call)
	if tup, ok := typ.(*types2.Tuple); ok {
		return tup.Len() - 1
	}
	return 0
}

// nextErr returns the next per-function error temporary spelling.
func (fc *funcLower) nextErr() string {
	spelling := fmt.Sprintf("_logos_e%d", fc.errSeq)
	fc.errSeq++
	return spelling
}

// nextFun returns the next per-function callee temporary spelling.
func (fc *funcLower) nextFun() string {
	spelling := fmt.Sprintf("_logos_f%d", fc.funSeq)
	fc.funSeq++
	return spelling
}

// nextArg returns the next per-function argument temporary spelling.
func (fc *funcLower) nextArg() string {
	spelling := fmt.Sprintf("_logos_a%d", fc.argSeq)
	fc.argSeq++
	return spelling
}

// zerosAt builds fresh zero values for the non-error results.
func (fc *funcLower) zerosAt(pos syntax.Pos) []syntax.Expr {
	var out []syntax.Expr
	for i := range fc.specs {
		sp := &fc.specs[i]
		if sp.isErr {
			continue
		}
		out = append(out, zeroForType(sp.typ, sp.typeExpr, pos))
	}
	return out
}

// exitPair builds the ? error path twice: once for the native tree and
// once for the eject twin.
func (fc *funcLower) exitPair(pos syntax.Pos, errSpelling, siteSpelling string) ([]syntax.Stmt, []syntax.Stmt) {
	native := buildExit(exitSpec{
		pos:   pos,
		err:   nameOf(pos, errSpelling),
		site:  nameOf(pos, siteSpelling),
		zeros: fc.zerosAt(pos),
	})
	text := buildExit(exitSpec{
		pos:   pos,
		err:   nameOf(pos, errSpelling),
		site:  nameOf(pos, siteSpelling),
		zeros: fc.zerosAt(pos),
	})
	return native, text
}

// lowerList lowers a statement list.
func (fc *funcLower) lowerList(list []syntax.Stmt) []syntax.Stmt {
	var out []syntax.Stmt
	for _, s := range list {
		out = append(out, fc.lowerStmt(s)...)
	}
	return out
}

// lowerStmt lowers one statement, splaying multi-statement results.
func (fc *funcLower) lowerStmt(s syntax.Stmt) []syntax.Stmt {
	switch s := s.(type) {
	case *syntax.AssignStmt,
		*syntax.ExprStmt,
		*syntax.CallStmt,
		*syntax.ErrdeferStmt,
		*syntax.ReturnStmt:
		return fc.lowerSimple(s)
	default:
		return fc.lowerCompound(s)
	}
}

// lowerSimple lowers the statements that directly hold ? and errdefer.
func (fc *funcLower) lowerSimple(s syntax.Stmt) []syntax.Stmt {
	switch s := s.(type) {
	case *syntax.AssignStmt:
		if t, call, ok := tryAssignOf(s); ok {
			return fc.lowerTryAssign(s, t, call)
		}
		scanForLiterals(fc.fl, s)
		return []syntax.Stmt{s}
	case *syntax.ExprStmt:
		if t, call, ok := tryExprOf(s); ok {
			return fc.lowerTryExpr(s, t, call)
		}
		scanForLiterals(fc.fl, s)
		return []syntax.Stmt{s}
	case *syntax.CallStmt:
		if s.Tok == syntax.Defer {
			if t, call, ok := tryCallOf(s); ok {
				return fc.lowerDeferTry(s, t, call)
			}
		}
		scanForLiterals(fc.fl, s)
		return []syntax.Stmt{s}
	case *syntax.ErrdeferStmt:
		return fc.lowerErrdefer(s)
	case *syntax.ReturnStmt:
		if fc.hasErrdefer {
			return fc.lowerReturn(s)
		}
		scanForLiterals(fc.fl, s)
		return []syntax.Stmt{s}
	default:
		return []syntax.Stmt{s}
	}
}

// lowerCompound lowers the statements that only contain nested forms.
func (fc *funcLower) lowerCompound(s syntax.Stmt) []syntax.Stmt {
	switch s := s.(type) {
	case *syntax.BlockStmt:
		s.List = fc.lowerList(s.List)
		return []syntax.Stmt{s}
	case *syntax.IfStmt:
		return fc.lowerIf(s)
	case *syntax.ForStmt:
		return fc.lowerFor(s)
	case *syntax.SwitchStmt:
		return fc.lowerSwitch(s)
	case *syntax.SelectStmt:
		for _, clause := range s.Body {
			scanForLiterals(fc.fl, clause.Comm)
			clause.Body = fc.lowerList(clause.Body)
		}
		return []syntax.Stmt{s}
	default:
		return fc.lowerLeaf(s)
	}
}

// lowerLeaf lowers labeled, declaration, send, and range statements.
func (fc *funcLower) lowerLeaf(s syntax.Stmt) []syntax.Stmt {
	switch s := s.(type) {
	case *syntax.LabeledStmt:
		inner := fc.lowerStmt(s.Stmt)
		s.Stmt = inner[0]
		return append([]syntax.Stmt{s}, inner[1:]...)
	case *syntax.DeclStmt:
		scanForLiterals(fc.fl, s)
		return []syntax.Stmt{s}
	case *syntax.SendStmt:
		scanForLiterals(fc.fl, s)
		return []syntax.Stmt{s}
	case *syntax.RangeClause:
		scanForLiterals(fc.fl, s)
		return []syntax.Stmt{s}
	default:
		return []syntax.Stmt{s}
	}
}

// lowerIf lowers an if statement and a try in its init clause.
func (fc *funcLower) lowerIf(s *syntax.IfStmt) []syntax.Stmt {
	scanForLiterals(fc.fl, s.Cond)
	if wrapped := fc.wrapInitStmt(s, s.Init, "if"); wrapped != nil {
		s.Then.List = fc.lowerList(s.Then.List)
		fc.lowerElse(s)
		return []syntax.Stmt{wrapped}
	}
	scanForLiterals(fc.fl, s.Init)
	s.Then.List = fc.lowerList(s.Then.List)
	fc.lowerElse(s)
	return []syntax.Stmt{s}
}

// lowerElse lowers the else branch of an if statement.
func (fc *funcLower) lowerElse(s *syntax.IfStmt) {
	switch els := s.Else.(type) {
	case nil:
	case *syntax.BlockStmt:
		els.List = fc.lowerList(els.List)
	case *syntax.IfStmt:
		// A try in an else-if init hoists into a wrapper block that
		// replaces the else branch.
		s.Else = fc.lowerIf(els)[0]
	}
}

// lowerFor lowers a for statement and a try in its init clause. A try in
// the post statement is left for the survivor check: no SimpleStmt can
// hold the error branch.
func (fc *funcLower) lowerFor(s *syntax.ForStmt) []syntax.Stmt {
	switch init := s.Init.(type) {
	case nil:
	case *syntax.RangeClause:
		scanForLiterals(fc.fl, init)
	default:
		if wrapped := fc.wrapInitStmt(s, init, "for"); wrapped != nil {
			scanForLiterals(fc.fl, s.Cond)
			scanForLiterals(fc.fl, s.Post)
			s.Body.List = fc.lowerList(s.Body.List)
			return []syntax.Stmt{wrapped}
		}
		scanForLiterals(fc.fl, init)
	}
	scanForLiterals(fc.fl, s.Cond)
	scanForLiterals(fc.fl, s.Post)
	s.Body.List = fc.lowerList(s.Body.List)
	return []syntax.Stmt{s}
}

// lowerSwitch lowers a switch statement and a try in its init clause.
func (fc *funcLower) lowerSwitch(s *syntax.SwitchStmt) []syntax.Stmt {
	if wrapped := fc.wrapInitStmt(s, s.Init, "switch"); wrapped != nil {
		scanForLiterals(fc.fl, s.Tag)
		fc.lowerCases(s.Body)
		return []syntax.Stmt{wrapped}
	}
	scanForLiterals(fc.fl, s.Init)
	scanForLiterals(fc.fl, s.Tag)
	fc.lowerCases(s.Body)
	return []syntax.Stmt{s}
}

// lowerCases lowers every case body of a switch statement.
func (fc *funcLower) lowerCases(body []*syntax.CaseClause) {
	for _, clause := range body {
		scanForLiterals(fc.fl, clause.Cases)
		clause.Body = fc.lowerList(clause.Body)
	}
}

// initCore is one lowering core for a try init clause.
type initCore func() (native []syntax.Stmt, text string, spanStart, spanEnd int)

// initCoreOf matches a try statement in init position.
func (fc *funcLower) initCoreOf(init syntax.SimpleStmt) (initCore, bool) {
	if s, isAssign := init.(*syntax.AssignStmt); isAssign {
		if t, call, matched := tryAssignOf(s); matched {
			return func() ([]syntax.Stmt, string, int, int) {
				return fc.tryAssignCore(s, t, call)
			}, true
		}
		return nil, false
	}
	if s, isExpr := init.(*syntax.ExprStmt); isExpr {
		if t, call, matched := tryExprOf(s); matched {
			return func() ([]syntax.Stmt, string, int, int) {
				return fc.tryExprCore(s, t, call)
			}, true
		}
	}
	return nil, false
}

// wrapInitStmt hoists a try init clause into a block around its host.
func (fc *funcLower) wrapInitStmt(host syntax.Stmt, init syntax.SimpleStmt, kw string) *syntax.BlockStmt {
	core, ok := fc.initCoreOf(init)
	if !ok {
		return nil
	}
	fl := fc.fl
	scanForLiterals(fl, init)
	native, text, _, spanEnd := core()
	// The hoisted statements sit left of the init clause in the lowered
	// tree, so their scopes open at the host keyword: the writer requires
	// scope positions never to move backward.
	anchorScopes(native, host.Pos())
	kwOff := fl.off(host.Pos())
	semi := skipSemi(fl.src, spanEnd)
	clearInit(host)
	end := blockEndOff(fl, host)
	fl.add(kwOff, semi, "{ "+text+kw+" ")
	fl.add(end, end, "} ")
	wrapper := blockOf(host.Pos(), append(native, host))
	wrapper.Rbrace = hostRbrace(host)
	return wrapper
}

// anchorScopes moves the scope-opening nodes of synthesized statements to
// pos. It visits only blocks and if chains, never user expressions, so
// user nodes keep their positions.
func anchorScopes(list []syntax.Stmt, pos syntax.Pos) {
	for _, s := range list {
		switch s := s.(type) {
		case *syntax.BlockStmt:
			s.SetPos(pos)
			s.Rbrace = pos
			anchorScopes(s.List, pos)
		case *syntax.IfStmt:
			s.SetPos(pos)
			anchorScopes([]syntax.Stmt{s.Then}, pos)
			if s.Else != nil {
				anchorScopes([]syntax.Stmt{s.Else}, pos)
			}
		}
	}
}

// hostRbrace returns the position of the closing brace that ends host. The
// wrapper block of a hoisted init clause closes there: scope marks must not
// run backward, and the host's own blocks close at their real braces.
func hostRbrace(host syntax.Stmt) syntax.Pos {
	switch host := host.(type) {
	case *syntax.IfStmt:
		for {
			switch els := host.Else.(type) {
			case *syntax.BlockStmt:
				return els.Rbrace
			case *syntax.IfStmt:
				host = els
			default:
				return host.Then.Rbrace
			}
		}
	case *syntax.ForStmt:
		return host.Body.Rbrace
	case *syntax.SwitchStmt:
		return host.Rbrace
	}
	return host.Pos()
}

// clearInit removes the init clause of an if, for, or switch statement.
func clearInit(host syntax.Stmt) {
	switch host := host.(type) {
	case *syntax.IfStmt:
		host.Init = nil
	case *syntax.ForStmt:
		host.Init = nil
	case *syntax.SwitchStmt:
		host.Init = nil
	}
}

// hostBlockEnd returns the offset just past the closing brace of host.
func blockEndOff(fl *fileLower, host syntax.Stmt) int {
	switch host := host.(type) {
	case *syntax.IfStmt:
		return ifEndOff(fl, host)
	case *syntax.ForStmt:
		return fl.off(host.Body.Rbrace) + 1
	case *syntax.SwitchStmt:
		return fl.off(host.Rbrace) + 1
	default:
		return len(fl.src)
	}
}

// ifEndOff returns the offset just past the final block of an if chain.
func ifEndOff(fl *fileLower, s *syntax.IfStmt) int {
	for {
		switch els := s.Else.(type) {
		case nil:
			return fl.off(s.Then.Rbrace) + 1
		case *syntax.BlockStmt:
			return fl.off(els.Rbrace) + 1
		case *syntax.IfStmt:
			s = els
		default:
			return len(fl.src)
		}
	}
}

// skipSemi returns the offset just past the first semicolon at or after
// from.
func skipSemi(src []byte, from int) int {
	end := from
	scanSegment(src, from, len(src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		if t.kind == token.SEMICOLON {
			end = t.end
			return false
		}
		return true
	})
	return end
}

// lowerTryAssign lowers lhs = f()? and lhs := f()? in a statement list.
func (fc *funcLower) lowerTryAssign(s *syntax.AssignStmt, t *syntax.TryExpr, call *syntax.CallExpr) []syntax.Stmt {
	fl := fc.fl
	native, text, spanStart, spanEnd := fc.tryAssignCore(s, t, call)
	fl.add(spanStart, spanEnd, text)
	return native
}

// tryAssignCore builds the native and text forms of a try assignment and
// returns them with the replaced span.
func (fc *funcLower) tryAssignCore(s *syntax.AssignStmt, t *syntax.TryExpr, call *syntax.CallExpr) ([]syntax.Stmt, string, int, int) {
	fl := fc.fl
	src := fl.src
	tpos := t.Pos()
	stmtPos := leftmostOf(syntax.UnpackListExpr(s.Lhs)[0])
	spanStart := fl.off(stmtPos)
	opOff := fl.off(s.Pos())
	qOff := fl.off(tpos)
	spanEnd := skipCloseParens(src, qOff+1)
	lhs := syntax.UnpackListExpr(s.Lhs)
	lhsText := string(trimSpanBytes(src[spanStart:opOff]))
	callText := fl.textOf(span{start: fl.off(leftmostOf(call.Fun)), end: qOff})
	errSpelling := fc.nextErr()
	siteSpelling := fl.newSite(tpos, tpos.Line())
	exitNative, exitText := fc.exitPair(tpos, errSpelling, siteSpelling)
	var native []syntax.Stmt
	tw := &twin{}
	lhsMark := tw.mark(stmtPos, lhsText)
	callMark := tw.mark(tpos, callText)
	if s.Op == syntax.Def {
		native = []syntax.Stmt{
			errDefineOf(stmtPos, lhs, call, nameOf(tpos, errSpelling)),
			errCheckOf(stmtPos, nameOf(tpos, errSpelling), exitNative),
		}
		text := renderText([]syntax.Stmt{
			errDefineOf(stmtPos, []syntax.Expr{lhsMark}, callMark, nameOf(tpos, errSpelling)),
			errCheckOf(stmtPos, nameOf(tpos, errSpelling), exitText),
		}, tw.subs)
		return native, text, spanStart, spanEnd
	}
	errDecl := nameOf(tpos, errSpelling)
	errUse := nameOf(tpos, errSpelling)
	errChk := nameOf(tpos, errSpelling)
	withErr := append(slices.Clone(lhs), errUse)
	native = []syntax.Stmt{blockOf(stmtPos, []syntax.Stmt{
		errorVarDecl(stmtPos, errDecl),
		assignOf(stmtPos, withErr, []syntax.Expr{call}),
		errCheckOf(stmtPos, errChk, exitNative),
	})}
	withErrT := []syntax.Expr{lhsMark, nameOf(tpos, errSpelling)}
	text := renderText([]syntax.Stmt{blockOf(stmtPos, []syntax.Stmt{
		errorVarDecl(stmtPos, nameOf(tpos, errSpelling)),
		assignOf(stmtPos, withErrT, []syntax.Expr{callMark}),
		errCheckOf(stmtPos, nameOf(tpos, errSpelling), exitText),
	})}, tw.subs)
	return native, text, spanStart, spanEnd
}

// lowerTryExpr lowers f()? as an expression statement.
func (fc *funcLower) lowerTryExpr(s *syntax.ExprStmt, t *syntax.TryExpr, call *syntax.CallExpr) []syntax.Stmt {
	fl := fc.fl
	native, text, spanStart, spanEnd := fc.tryExprCore(s, t, call)
	fl.add(spanStart, spanEnd, text)
	return native
}

// tryExprCore builds the native and text forms of an expression try and
// returns them with the replaced span.
func (fc *funcLower) tryExprCore(s *syntax.ExprStmt, t *syntax.TryExpr, call *syntax.CallExpr) ([]syntax.Stmt, string, int, int) {
	fl := fc.fl
	tpos := t.Pos()
	stmtPos := leftmostOf(call.Fun)
	spanStart := spanStartBack(fl.src, fl.off(stmtPos))
	qOff := fl.off(tpos)
	spanEnd := skipCloseParens(fl.src, qOff+1)
	callText := fl.textOf(span{start: fl.off(stmtPos), end: qOff})
	errSpelling := fc.nextErr()
	siteSpelling := fl.newSite(tpos, tpos.Line())
	exitNative, exitText := fc.exitPair(tpos, errSpelling, siteSpelling)
	tw := &twin{}
	callMark := tw.mark(tpos, callText)
	n := fc.callResults(call)
	var native []syntax.Stmt
	var text string
	if n == 0 {
		native = []syntax.Stmt{errCheckInitOf(stmtPos, nil, call, nameOf(tpos, errSpelling), exitNative)}
		text = renderText([]syntax.Stmt{errCheckInitOf(stmtPos, nil, callMark, nameOf(tpos, errSpelling), exitText)}, tw.subs)
	} else {
		native = []syntax.Stmt{
			errDefineOf(stmtPos, blankNames(stmtPos, n), call, nameOf(tpos, errSpelling)),
			errCheckOf(stmtPos, nameOf(tpos, errSpelling), exitNative),
		}
		text = renderText([]syntax.Stmt{
			errDefineOf(stmtPos, blankNames(stmtPos, n), callMark, nameOf(tpos, errSpelling)),
			errCheckOf(stmtPos, nameOf(tpos, errSpelling), exitText),
		}, tw.subs)
	}
	return native, text, spanStart, spanEnd
}

// trimSpanBytes trims ASCII white space around src.
func trimSpanBytes(src []byte) []byte {
	start := 0
	for start < len(src) && isSpace(src[start]) {
		start++
	}
	end := len(src)
	for end > start && isSpace(src[end-1]) {
		end--
	}
	return src[start:end]
}
