// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"cmp"
	"fmt"
	"go/token"
	"slices"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// This file lowers defer f()?, errdefer calls, and the return rewrite of
// errdefer functions. Deferred callees and arguments are bound at the
// statement into temporaries, matching the Go defer evaluation rule.

// capturePiece is one callee or argument of a deferred call.
type capturePiece struct {
	expr    syntax.Expr
	text    string
	start   int
	end     int
	capture bool
	single  string
	multi   []string
}

// callPlan binds every callee and argument of a deferred call.
type callPlan struct {
	fun  capturePiece
	args []capturePiece
	dots bool
	end  int
}

// asTuple matches a tuple type.
func asTuple(typ types2.Type) (*types2.Tuple, bool) {
	tup, ok := typ.(*types2.Tuple)
	return tup, ok
}

// typeFlagsOf returns the recorded type flags of e.
func typeFlagsOf(fc *funcLower, e syntax.Expr) (builtin, isType bool) {
	info := fc.fl.pkg.info
	node := syntax.Unparen(e)
	if info.Types != nil {
		if tv, ok := info.Types[node]; ok {
			return tv.IsBuiltin(), tv.IsType()
		}
		if tv, ok := info.Types[e]; ok {
			return tv.IsBuiltin(), tv.IsType()
		}
		return false, false
	}
	flags := node.GetTypeInfo()
	return flags.IsBuiltin(), flags.IsType()
}

// isBuiltinExpr reports whether e denotes a builtin function.
func isBuiltinExpr(fc *funcLower, e syntax.Expr) bool {
	builtin, _ := typeFlagsOf(fc, e)
	return builtin
}

// isTypeExpr reports whether e denotes a type.
func isTypeExpr(fc *funcLower, e syntax.Expr) bool {
	_, isType := typeFlagsOf(fc, e)
	return isType
}

// planCall maps a deferred call to its capture plan. It reports false for
// conversions, which cannot be bound to a variable.
func (fc *funcLower) planCall(call *syntax.CallExpr) (*callPlan, bool) {
	fl := fc.fl
	src := fl.src
	lparen := fl.off(call.Pos())
	split, ok := splitCall(src, lparen)
	if !ok {
		return nil, false
	}
	plan := &callPlan{dots: call.HasDots, end: split.end}
	fun := capturePiece{expr: call.Fun}
	fun.start = fl.off(leftmostOf(call.Fun))
	fun.end = lastTokenEnd(src, fun.start, lparen)
	if isTypeExpr(fc, call.Fun) {
		return nil, false
	}
	fun.capture = !isBuiltinExpr(fc, call.Fun) && !genericTarget(fc, call.Fun)
	fun.text = fl.textOf(span{start: fun.start, end: fun.end})
	if fun.capture {
		fun.single = fc.nextFun()
	}
	plan.fun = fun
	if len(split.args) != len(call.ArgList) {
		return nil, false
	}
	for i, raw := range call.ArgList {
		arg := capturePiece{expr: raw}
		trimmed := trimSpan(src, span{start: split.args[i].start, end: split.args[i].end})
		arg.start = trimmed.start
		arg.end = trimmed.end
		arg.text = fl.textOf(trimmed)
		switch argKind(fc, raw) {
		case argInline:
			arg.capture = false
		case argMulti:
			arg.capture = true
			arg.multi = fc.multiNames(raw)
		default:
			arg.capture = true
			arg.single = fc.nextArg()
		}
		plan.args = append(plan.args, arg)
	}
	return plan, true
}

// genericTarget reports whether fun is a reference to a generic function
// or carries explicit type arguments. Such callees are spelled in the
// closure so inference reruns on the captured arguments; binding one to a
// variable would freeze or lose its type arguments.
func genericTarget(fc *funcLower, fun syntax.Expr) bool {
	node := syntax.Unparen(fun)
	if _, ok := node.(*syntax.IndexExpr); ok {
		return true
	}
	var name *syntax.Name
	switch n := node.(type) {
	case *syntax.Name:
		name = n
	case *syntax.SelectorExpr:
		name = n.Sel
	default:
		return false
	}
	obj := fc.fl.pkg.info.ObjectOf(name)
	fn, ok := obj.(*types2.Func)
	if !ok {
		return false
	}
	sig, ok := fn.Type().(*types2.Signature)
	if !ok || sig.TypeParams() == nil {
		return false
	}
	return sig.TypeParams().Len() > 0
}

// argClass selects inline or captured binding for one argument.
type argClass int

const (
	argCapture argClass = iota
	argInline
	argMulti
)

// argKind binds constants and nil inline and captures the rest. Constants
// keep their literal text so default types never shift; every other value
// is already typed, so short declarations preserve it exactly.
func argKind(fc *funcLower, raw syntax.Expr) argClass {
	node := syntax.Unparen(raw)
	if name, ok := node.(*syntax.Name); ok && name.Value == "nil" {
		return argInline
	}
	if _, ok := typeConst(fc.fl.pkg.info, node); ok {
		return argInline
	}
	if _, ok := typeConst(fc.fl.pkg.info, raw); ok {
		return argInline
	}
	typ := cmp.Or(fc.fl.pkg.info.TypeOf(raw), fc.fl.pkg.info.TypeOf(node))
	if tup, ok := asTuple(typ); ok && tup.Len() > 1 {
		return argMulti
	}
	return argCapture
}

// multiNames returns one fresh spelling per value of a tuple argument.
func (fc *funcLower) multiNames(raw syntax.Expr) []string {
	typ := fc.fl.pkg.info.TypeOf(raw)
	tup, _ := asTuple(typ)
	names := make([]string, tup.Len())
	for i := range names {
		names[i] = fc.nextArg()
	}
	return names
}

// checkPair builds the capture statement and the guarded call check of a
// defer? or errdefer? body, twice each for the native tree and the twin.
func (fc *funcLower) checkPair(plan *callPlan, cpos syntax.Pos) ([]syntax.Stmt, []syntax.Stmt, *twin) {
	fc.fl.needRT = true // the join below calls _logos_rt.Join
	tw := &twin{}
	funN, argN, capNamesN, capValsN := fc.planUses(plan, nil)
	funT, argT, capNamesT, capValsT := fc.planUses(plan, tw)
	inner := "_logos_e"
	joinN := joinAssign(cpos, fc.errName, inner)
	callN := callOf(cpos, funN, argN)
	callN.HasDots = plan.dots
	checkN := ifOf(cpos, defineOf(cpos, []syntax.Expr{nameOf(cpos, inner)}, []syntax.Expr{callN}), notNil(cpos, nameOf(cpos, inner)), joinN)
	joinT := joinAssign(cpos, fc.errName, inner)
	callT := callOf(cpos, funT, argT)
	callT.HasDots = plan.dots
	checkT := ifOf(cpos, defineOf(cpos, []syntax.Expr{nameOf(cpos, inner)}, []syntax.Expr{callT}), notNil(cpos, nameOf(cpos, inner)), joinT)
	var native, text []syntax.Stmt
	if len(capNamesN) > 0 {
		native = append(native, defineOf(cpos, capNamesN, capValsN))
		text = append(text, defineOf(cpos, capNamesT, capValsT))
	}
	return append(native, checkN), append(text, checkT), tw
}

// planUses resolves one side of a capture plan: user nodes with nil twin,
// markers with a twin.
func (fc *funcLower) planUses(plan *callPlan, tw *twin) (syntax.Expr, []syntax.Expr, []syntax.Expr, []syntax.Expr) {
	use := func(piece capturePiece, names []string) syntax.Expr {
		if !piece.capture {
			return useOne(piece, tw)
		}
		if len(names) > 0 {
			out := make([]syntax.Expr, len(names))
			for i, spelling := range names {
				out[i] = nameOf(piece.expr.Pos(), spelling)
			}
			return listOf(piece.expr.Pos(), out)
		}
		return nameOf(piece.expr.Pos(), piece.single)
	}
	funNames := funNameList(plan.fun)
	fun := use(plan.fun, funNames)
	var args, names, vals []syntax.Expr
	names = append(names, nameList(plan.fun)...)
	vals = append(vals, valList(plan.fun, tw)...)
	for _, piece := range plan.args {
		var argNames []string
		if piece.capture {
			argNames = pieceNames(piece)
		}
		args = append(args, use(piece, argNames))
		names = append(names, nameList(piece)...)
		vals = append(vals, valList(piece, tw)...)
	}
	return fun, args, names, vals
}

// useOne resolves one inline piece to its node or its marker.
func useOne(piece capturePiece, tw *twin) syntax.Expr {
	if tw == nil {
		return piece.expr
	}
	return tw.mark(piece.expr.Pos(), piece.text)
}

// funNameList returns the capture names of the callee piece.
func funNameList(piece capturePiece) []string {
	if !piece.capture {
		return nil
	}
	return []string{piece.single}
}

// pieceNames returns the capture names of an argument piece.
func pieceNames(piece capturePiece) []string {
	if len(piece.multi) > 0 {
		return piece.multi
	}
	return []string{piece.single}
}

// nameList returns fresh name nodes for the capture names of a piece.
func nameList(piece capturePiece) []syntax.Expr {
	if !piece.capture {
		return nil
	}
	names := pieceNames(piece)
	out := make([]syntax.Expr, len(names))
	for i, spelling := range names {
		out[i] = nameOf(piece.expr.Pos(), spelling)
	}
	return out
}

// valList returns the captured value side: the user node, or its marker.
func valList(piece capturePiece, tw *twin) []syntax.Expr {
	if !piece.capture {
		return nil
	}
	if tw == nil {
		return []syntax.Expr{piece.expr}
	}
	return []syntax.Expr{tw.mark(piece.expr.Pos(), piece.text)}
}

// lowerDeferTry lowers defer f()? in a statement list.
func (fc *funcLower) lowerDeferTry(s *syntax.CallStmt, t *syntax.TryExpr, call *syntax.CallExpr) []syntax.Stmt {
	fl := fc.fl
	cpos := syntax.StartPos(s)
	stmtOff := fl.off(cpos)
	spanEnd := skipCloseParens(fl.src, fl.off(t.Pos())+1)
	plan, ok := fc.planCall(call)
	if !ok {
		return []syntax.Stmt{s}
	}
	native, textStmts, tw := fc.checkPair(plan, cpos)
	closureN := deferClosure(cpos, []syntax.Stmt{native[len(native)-1]})
	headN := append(slices.Clone(native[:len(native)-1]), deferOf(cpos, callOf(cpos, closureN, nil)))
	closureT := deferClosure(cpos, []syntax.Stmt{textStmts[len(textStmts)-1]})
	headT := append(slices.Clone(textStmts[:len(textStmts)-1]), deferOf(cpos, callOf(cpos, closureT, nil)))
	prefix := leftoverComments(fl, span{start: stmtOff, end: spanEnd}, planSpans(plan))
	fl.add(stmtOff, spanEnd, prefix+renderText(headT, tw.subs))
	return headN
}

// lowerErrdefer lowers errdefer f() and errdefer f()?.
func (fc *funcLower) lowerErrdefer(s *syntax.ErrdeferStmt) []syntax.Stmt {
	fl := fc.fl
	cpos := syntax.StartPos(s)
	stmtOff := fl.off(cpos)
	inner := syntax.Unparen(s.Call)
	if t, ok := inner.(*syntax.TryExpr); ok {
		return fc.lowerErrdeferTry(s, t, cpos, stmtOff)
	}
	call, ok := inner.(*syntax.CallExpr)
	if !ok {
		return []syntax.Stmt{s}
	}
	plan, planned := fc.planCall(call)
	if !planned {
		return []syntax.Stmt{s}
	}
	tw := &twin{}
	funN, argN, capNamesN, capValsN := fc.planUses(plan, nil)
	funT, argT, capNamesT, capValsT := fc.planUses(plan, tw)
	callN := callOf(cpos, funN, argN)
	callN.HasDots = plan.dots
	callT := callOf(cpos, funT, argT)
	callT.HasDots = plan.dots
	bodyN := []syntax.Stmt{exprStmtOf(cpos, callN)}
	bodyT := []syntax.Stmt{exprStmtOf(cpos, callT)}
	native := fc.guardedDefer(cpos, bodyN, capNamesN, capValsN)
	text := fc.guardedDefer(cpos, bodyT, capNamesT, capValsT)
	spanEnd := skipCloseParens(fl.src, plan.end)
	prefix := leftoverComments(fl, span{start: stmtOff, end: spanEnd}, planSpans(plan))
	fl.add(stmtOff, spanEnd, prefix+renderText(text, tw.subs))
	return native
}

// lowerErrdeferTry lowers errdefer f()? with a joining guarded body.
func (fc *funcLower) lowerErrdeferTry(s *syntax.ErrdeferStmt, t *syntax.TryExpr, cpos syntax.Pos, stmtOff int) []syntax.Stmt {
	fl := fc.fl
	call, ok := syntax.Unparen(t.X).(*syntax.CallExpr)
	if !ok {
		return []syntax.Stmt{s}
	}
	plan, planned := fc.planCall(call)
	if !planned {
		return []syntax.Stmt{s}
	}
	spanEnd := skipCloseParens(fl.src, fl.off(t.Pos())+1)
	native, textStmts, tw := fc.checkPair(plan, cpos)
	bodyN := []syntax.Stmt{native[len(native)-1]}
	bodyT := []syntax.Stmt{textStmts[len(textStmts)-1]}
	headN := append(slices.Clone(native[:len(native)-1]), fc.guardedDefer(cpos, bodyN, nil, nil)[0])
	headT := append(slices.Clone(textStmts[:len(textStmts)-1]), fc.guardedDefer(cpos, bodyT, nil, nil)[0])
	prefix := leftoverComments(fl, span{start: stmtOff, end: spanEnd}, planSpans(plan))
	fl.add(stmtOff, spanEnd, prefix+renderText(headT, tw.subs))
	return headN
}

// guardedDefer wraps a body in the errdefer guard and a defer statement,
// preceded by an optional capture statement.
func (fc *funcLower) guardedDefer(cpos syntax.Pos, body []syntax.Stmt, capNames, capVals []syntax.Expr) []syntax.Stmt {
	guarded := guardedJoin(cpos, fc.okName, fc.errName, body)
	closure := deferClosure(cpos, []syntax.Stmt{guarded})
	deferStmt := deferOf(cpos, callOf(cpos, closure, nil))
	if len(capNames) == 0 {
		return []syntax.Stmt{deferStmt}
	}
	return []syntax.Stmt{defineOf(cpos, capNames, capVals), deferStmt}
}

// planSpans returns the source spans owned by a capture plan.
func planSpans(plan *callPlan) []span {
	spans := []span{{start: plan.fun.start, end: plan.fun.end}}
	for _, arg := range plan.args {
		spans = append(spans, span{start: arg.start, end: arg.end})
	}
	return spans
}

// leftoverComments lifts the comments of a replaced span that fall outside
// every piece, so interior comments survive in the eject text.
func leftoverComments(fl *fileLower, whole span, pieces []span) string {
	var prefix string
	for _, c := range commentsIn(fl.src, whole.start, whole.end) {
		if !insideAny(c, pieces) {
			prefix += string(fl.src[c.start:c.end]) + "\n"
		}
	}
	return prefix
}

// insideAny reports whether c lies inside one of pieces.
func insideAny(c span, pieces []span) bool {
	for _, p := range pieces {
		if c.start >= p.start && c.end <= p.end {
			return true
		}
	}
	return false
}

// lowerReturn rewrites the returns of an errdefer function so a panic
// while evaluating results counts as failure. Results go through fresh
// temporaries, so a shadowing declaration at the return site cannot steal
// the assignment the way a named target would.
func (fc *funcLower) lowerReturn(s *syntax.ReturnStmt) []syntax.Stmt {
	fl := fc.fl
	scanForLiterals(fl, s.Results)
	retPos := s.Pos()
	retOff := fl.off(retPos)
	if s.Results == nil {
		fl.add(retOff, retOff, fc.okName+" = true; ")
		return []syntax.Stmt{assignOf(retPos, []syntax.Expr{nameOf(retPos, fc.okName)}, []syntax.Expr{nameOf(retPos, "true")}), s}
	}
	if len(fc.specs) == 0 {
		return []syntax.Stmt{s}
	}
	end := stmtEnd(fl.src, retOff)
	trimmed := trimSpan(fl.src, span{start: retOff + len("return"), end: end})
	resText := fl.textOf(trimmed)
	results := syntax.UnpackListExpr(s.Results)
	temps := fc.nextTemps(len(fc.specs))
	// Each temporary is declared with its result type and then assigned.
	// A short declaration would type the temporary from the value alone:
	// untyped nil has no type, and an untyped constant would take its
	// default type instead of the result type. Assignment converts exactly
	// as a return statement does.
	build := func(rhs []syntax.Expr) []syntax.Stmt {
		out := make([]syntax.Stmt, 0, len(temps)+3)
		for i, spelling := range temps {
			out = append(out, typedVarDecl(retPos, nameOf(retPos, spelling), mustCloneType(fc.specs[i].typeExpr, retPos)))
		}
		return append(out,
			assignOf(retPos, tempNodes(retPos, temps), rhs),
			assignOf(retPos, []syntax.Expr{nameOf(retPos, fc.okName)}, []syntax.Expr{nameOf(retPos, "true")}),
			returnOf(retPos, tempNodes(retPos, temps)),
		)
	}
	native := build(slices.Clone(results))
	tw := &twin{}
	resMark := tw.mark(retPos, resText)
	text := renderText(build([]syntax.Expr{resMark}), tw.subs)
	prefix := leftoverComments(fl, span{start: retOff, end: end}, []span{trimmed})
	fl.add(retOff, end, prefix+text)
	return native
}

// nextTemps returns n fresh per-function temporary spellings.
func (fc *funcLower) nextTemps(n int) []string {
	temps := make([]string, n)
	for i := range temps {
		temps[i] = fmt.Sprintf("_logos_v%d", fc.tmpSeq)
		fc.tmpSeq++
	}
	return temps
}

// tempNodes builds fresh name nodes for spellings at pos.
func tempNodes(pos syntax.Pos, spellings []string) []syntax.Expr {
	nodes := make([]syntax.Expr, len(spellings))
	for i, spelling := range spellings {
		nodes[i] = nameOf(pos, spelling)
	}
	return nodes
}

// skipCloseParens extends a span past parentheses the parser dropped.
func skipCloseParens(src []byte, from int) int {
	end := from
	scanSegment(src, from, len(src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		if t.kind == token.RPAREN {
			end = t.end
			return true
		}
		return false
	})
	return end
}
