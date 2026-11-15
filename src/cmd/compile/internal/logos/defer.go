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

type capturePiece struct {
	expr      syntax.Expr
	text      otext
	start     int
	end       int
	capture   bool
	single    string
	multi     []string
	takeAddr  bool
	takeDeref bool
	nilCheck  bool
}

type callPlan struct {
	fun      capturePiece
	args     []capturePiece
	dots     bool
	end      int
	isMethod bool
	recv     capturePiece
	method   string
}

func typeFlagsOf(fl *fileLower, e syntax.Expr) (builtin, isType bool) {
	info := fl.pkg.info
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

func isBuiltinExpr(fc *funcLower, e syntax.Expr) bool {
	builtin, _ := typeFlagsOf(fc.fl, e)
	return builtin
}

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
	if _, isType := typeFlagsOf(fl, call.Fun); isType {
		return nil, false
	}
	if len(split.args) != len(call.ArgList) {
		return nil, false
	}
	fun.text = fl.lazyText(span{start: fun.start, end: fun.end})
	if recv, method, ok := fc.methodRecv(call); ok {
		plan.isMethod = true
		plan.recv = recv
		plan.method = method
	} else {
		fun.capture = !isBuiltinExpr(fc, call.Fun) && !genericTarget(fc, call.Fun)
		if fun.capture {
			fun.single = fc.nextFun()
		}
	}
	plan.fun = fun
	for i, raw := range call.ArgList {
		arg := capturePiece{expr: raw}
		trimmed := trimSpan(src, span{start: split.args[i].start, end: split.args[i].end})
		arg.start = trimmed.start
		arg.end = trimmed.end
		arg.text = fl.lazyText(trimmed)
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

func (fc *funcLower) calleeFunc(fun syntax.Expr) (*types2.Func, bool) {
	var name *syntax.Name
	switch n := syntax.Unparen(fun).(type) {
	case *syntax.Name:
		name = n
	case *syntax.SelectorExpr:
		name = n.Sel
	default:
		return nil, false
	}
	fn, ok := fc.fl.pkg.info.ObjectOf(name).(*types2.Func)
	return fn, ok
}

func genericTarget(fc *funcLower, fun syntax.Expr) bool {
	if _, ok := syntax.Unparen(fun).(*syntax.IndexExpr); ok {
		return true
	}
	fn, ok := fc.calleeFunc(fun)
	if !ok {
		return false
	}
	sig, ok := fn.Type().(*types2.Signature)
	if !ok || sig.TypeParams() == nil {
		return false
	}
	return sig.TypeParams().Len() > 0
}

func (fc *funcLower) planScopeCall(call *syntax.CallExpr) (*callPlan, bool) {
	plan, ok := fc.planCall(call)
	if ok && plan.fun.capture {
		_, static := fc.calleeFunc(plan.fun.expr)
		plan.fun.capture = !static
	}
	return plan, ok
}

func (fc *funcLower) methodRecv(call *syntax.CallExpr) (capturePiece, string, bool) {
	fl := fc.fl
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	if !ok || sel.Sel == nil {
		return capturePiece{}, "", false
	}
	info := fl.pkg.info
	if info == nil {
		return capturePiece{}, "", false
	}
	s, ok := info.Selections[sel]
	if !ok || s.Kind() != types2.MethodVal {
		return capturePiece{}, "", false
	}
	rspan := fl.receiverSpan(call)
	recv := capturePiece{expr: sel.X, capture: true}
	recv.start = rspan.start
	recv.end = rspan.end
	recv.text = fl.lazyText(rspan)
	recv.single = fc.nextFun()
	fc.bindRecv(&recv, s, sel.X)
	if recv.takeAddr {
		recv.text = cat(gen("&"), recv.text)
	}
	if recv.takeDeref {
		recv.text = cat(gen("*"), recv.text)
	}
	return recv, sel.Sel.Value, true
}

func (fc *funcLower) bindRecv(recv *capturePiece, s *types2.Selection, x syntax.Expr) {
	effX, effText, effT := fc.recvTarget(s, x, recv.text)
	if effT == nil {
		return
	}
	recv.expr = effX
	recv.text = effText
	if _, isParam := types2.Unalias(effT).(*types2.TypeParam); !isParam {
		if _, ok := effT.Underlying().(*types2.Interface); ok {
			recv.nilCheck = true
			return
		}
	}
	meth, ok := s.Obj().(*types2.Func)
	if !ok {
		return
	}
	sig, ok := meth.Type().(*types2.Signature)
	if !ok || sig.Recv() == nil {
		return
	}
	_, wantPtr := types2.Unalias(sig.Recv().Type()).(*types2.Pointer)
	_, havePtr := types2.Unalias(effT).(*types2.Pointer)
	switch {
	case wantPtr && !havePtr:
		recv.takeAddr = true
	case !wantPtr && havePtr:
		recv.takeDeref = true
	}
}

func (fc *funcLower) recvTarget(s *types2.Selection, x syntax.Expr, text otext) (syntax.Expr, otext, types2.Type) {
	typ := s.Recv()
	if typ == nil {
		return x, text, nil
	}
	idx := s.Index()
	if len(idx) == 0 {
		return x, text, typ
	}
	expr := x
	for _, fi := range idx[:len(idx)-1] {
		st := types2.Unalias(typ)
		if ptr, ok := st.(*types2.Pointer); ok {
			st = types2.Unalias(ptr.Elem())
		}
		strct, ok := st.(*types2.Struct)
		if !ok || fi < 0 || fi >= strct.NumFields() {
			return x, text, nil
		}
		fld := strct.Field(fi)
		expr = selOf(x.Pos(), expr, fld.Name())
		text = cat(text, gen("."+fld.Name()))
		typ = fld.Type()
	}
	return expr, text, typ
}

type argClass int

const (
	argCapture argClass = iota
	argInline
	argMulti
)

func argKind(fc *funcLower, raw syntax.Expr) argClass {
	node := syntax.Unparen(raw)
	if _, expanded := node.(*syntax.ListExpr); expanded {
		return argInline
	}
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
	if tup, ok := typ.(*types2.Tuple); ok && tup.Len() > 1 {
		return argMulti
	}
	return argCapture
}

func (fc *funcLower) multiNames(raw syntax.Expr) []string {
	typ := fc.fl.pkg.info.TypeOf(raw)
	tup, _ := typ.(*types2.Tuple)
	names := make([]string, tup.Len())
	for i := range names {
		names[i] = fc.nextArg()
	}
	return names
}

func (fc *funcLower) checkPair(plan *callPlan, cpos syntax.Pos) ([]syntax.Stmt, []syntax.Stmt, *twin) {
	fc.fl.needRT = true
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

func (fc *funcLower) planUses(plan *callPlan, tw *twin) (syntax.Expr, []syntax.Expr, []syntax.Expr, []syntax.Expr) {
	use := func(piece capturePiece, names []string) syntax.Expr {
		if !piece.capture {
			if tw == nil {
				return piece.expr
			}
			return tw.mark(piece.expr.Pos(), piece.text)
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
	var fun syntax.Expr
	var args, names, vals []syntax.Expr
	if plan.isMethod {
		fun = selOf(plan.fun.expr.Pos(), use(plan.recv, pieceNames(plan.recv)), plan.method)
		names = append(names, nameList(plan.recv)...)
		vals = append(vals, valList(plan.recv, tw)...)
	} else {
		fun = use(plan.fun, pieceNames(plan.fun))
		names = append(names, nameList(plan.fun)...)
		vals = append(vals, valList(plan.fun, tw)...)
	}
	for _, piece := range plan.args {
		var argNames []string
		if piece.capture {
			argNames = pieceNames(piece)
		}
		args = append(args, syntax.UnpackListExpr(use(piece, argNames))...)
		names = append(names, nameList(piece)...)
		vals = append(vals, valList(piece, tw)...)
	}
	return fun, args, names, vals
}

func wrapRecv(piece capturePiece) syntax.Expr {
	switch {
	case piece.takeAddr:
		return at(&syntax.Operation{Op: syntax.And, X: piece.expr}, piece.expr.Pos())
	case piece.takeDeref:
		return at(&syntax.Operation{Op: syntax.Mul, X: piece.expr}, piece.expr.Pos())
	default:
		return piece.expr
	}
}

func ifaceNilCheckStmt(pos syntax.Pos, recvTemp, method string) syntax.Stmt {
	cond := at(&syntax.Operation{Op: syntax.Eql, X: nameOf(pos, recvTemp), Y: nilOf(pos)}, pos)
	discard := assignOf(pos, []syntax.Expr{nameOf(pos, "_")}, []syntax.Expr{selOf(pos, nameOf(pos, recvTemp), method)})
	return ifOf(pos, nil, cond, []syntax.Stmt{discard})
}

func spliceRecvCheck(stmts []syntax.Stmt, chk syntax.Stmt) []syntax.Stmt {
	if chk == nil || len(stmts) == 0 {
		return stmts
	}
	out := make([]syntax.Stmt, 0, len(stmts)+1)
	out = append(out, stmts[:len(stmts)-1]...)
	out = append(out, chk)
	return append(out, stmts[len(stmts)-1])
}

func pieceNames(piece capturePiece) []string {
	if len(piece.multi) > 0 {
		return piece.multi
	}
	return []string{piece.single}
}

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

func valList(piece capturePiece, tw *twin) []syntax.Expr {
	if !piece.capture {
		return nil
	}
	if tw == nil {
		return []syntax.Expr{wrapRecv(piece)}
	}
	return []syntax.Expr{tw.mark(piece.expr.Pos(), piece.text)}
}

func (fc *funcLower) lowerDeferTry(s *syntax.CallStmt, t *syntax.TryExpr, call *syntax.CallExpr) []syntax.Stmt {
	fl := fc.fl
	fc.fl.needRT = true
	cpos := syntax.StartPos(s)
	stmtOff := fl.off(cpos)
	spanEnd := skipCloseParens(fl.src, fl.off(t.Pos())+1)
	plan, ok := fc.planCall(call)
	if !ok {
		return []syntax.Stmt{s}
	}
	_, textStmts, tw := fc.checkPair(plan, cpos)
	headN := []syntax.Stmt{deferOf(cpos, callOf(cpos, rtSel(cpos, "DeferTry"), []syntax.Expr{addrOf(cpos, nameOf(cpos, fc.errName)), call}))}
	closureT := deferClosure(cpos, []syntax.Stmt{textStmts[len(textStmts)-1]})
	headT := append(slices.Clone(textStmts[:len(textStmts)-1]), deferOf(cpos, callOf(cpos, closureT, nil)))
	if plan.isMethod && plan.recv.nilCheck {
		headT = spliceRecvCheck(headT, ifaceNilCheckStmt(cpos, plan.recv.single, plan.method))
	}
	fc.spliceCall(stmtOff, spanEnd, plan, scoped(cpos, headT), tw)
	return scoped(cpos, headN)
}

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
	if plan.isMethod && plan.recv.nilCheck {
		native = spliceRecvCheck(native, ifaceNilCheckStmt(cpos, plan.recv.single, plan.method))
	}
	text := fc.guardedDefer(cpos, bodyT, capNamesT, capValsT)
	if plan.isMethod && plan.recv.nilCheck {
		text = spliceRecvCheck(text, ifaceNilCheckStmt(cpos, plan.recv.single, plan.method))
	}
	spanEnd := skipCloseParens(fl.src, plan.end)
	fc.spliceCall(stmtOff, spanEnd, plan, scoped(cpos, text), tw)
	return scoped(cpos, native)
}

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
	if plan.isMethod && plan.recv.nilCheck {
		headN = spliceRecvCheck(headN, ifaceNilCheckStmt(cpos, plan.recv.single, plan.method))
		headT = spliceRecvCheck(headT, ifaceNilCheckStmt(cpos, plan.recv.single, plan.method))
	}
	fc.spliceCall(stmtOff, spanEnd, plan, scoped(cpos, headT), tw)
	return scoped(cpos, headN)
}

func (fc *funcLower) guardedDefer(cpos syntax.Pos, body []syntax.Stmt, capNames, capVals []syntax.Expr) []syntax.Stmt {
	guarded := guardedJoin(cpos, fc.okName, fc.errName, body)
	closure := deferClosure(cpos, []syntax.Stmt{guarded})
	deferStmt := deferOf(cpos, callOf(cpos, closure, nil))
	if len(capNames) == 0 {
		return []syntax.Stmt{deferStmt}
	}
	return []syntax.Stmt{defineOf(cpos, capNames, capVals), deferStmt}
}

func scoped(pos syntax.Pos, stmts []syntax.Stmt) []syntax.Stmt {
	return []syntax.Stmt{blockOf(pos, stmts)}
}

func planSpans(plan *callPlan) []span {
	spans := []span{{start: plan.fun.start, end: plan.fun.end}}
	for _, arg := range plan.args {
		spans = append(spans, span{start: arg.start, end: arg.end})
	}
	return spans
}

func leftoverComments(fl *fileLower, whole span, pieces []span) otext {
	var parts []otext
	for _, c := range commentsIn(fl.src, whole.start, whole.end) {
		if !insideAny(c, pieces) {
			parts = append(parts, fl.srcUser(c.start, c.end), gen("\n"))
		}
	}
	return cat(parts...)
}

func (fc *funcLower) spliceCall(stmtOff, spanEnd int, plan *callPlan, text []syntax.Stmt, tw *twin) {
	prefix := leftoverComments(fc.fl, span{start: stmtOff, end: spanEnd}, planSpans(plan))
	fc.fl.addT(stmtOff, spanEnd, cat(prefix, renderText(text, tw.subs)))
}

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
	for _, sp := range fl.splices {
		if retOff < sp.end && sp.start < end {
			return []syntax.Stmt{s}
		}
	}
	trimmed := trimSpan(fl.src, span{start: retOff + len("return"), end: end})
	resText := fl.lazyText(trimmed)
	results := syntax.UnpackListExpr(s.Results)
	temps := fc.nextTemps(len(fc.specs))
	build := func(rhs []syntax.Expr) []syntax.Stmt {
		return fc.returnValues(retPos, temps, rhs, nil)
	}
	native := build(slices.Clone(results))
	tw := &twin{}
	resMark := tw.mark(retPos, resText)
	text := renderText(build([]syntax.Expr{resMark}), tw.subs)
	prefix := leftoverComments(fl, span{start: retOff, end: end}, []span{trimmed})
	fl.addT(retOff, end, cat(prefix, text))
	return native
}

func (fc *funcLower) returnValues(pos syntax.Pos, temps []string, rhs []syntax.Expr, mid []syntax.Stmt) []syntax.Stmt {
	out := make([]syntax.Stmt, 0, len(temps)+len(mid)+3)
	for i, spelling := range temps {
		out = append(out, typedVarDecl(pos, nameOf(pos, spelling), mustCloneType(fc.specs[i].typeExpr, pos)))
	}
	out = append(out, assignOf(pos, tempNodes(pos, temps), rhs))
	out = append(out, mid...)
	if fc.hasErrdefer {
		out = append(out, assignOf(pos, []syntax.Expr{nameOf(pos, fc.okName)}, []syntax.Expr{nameOf(pos, "true")}))
	}
	return append(out, returnOf(pos, tempNodes(pos, temps)))
}

func (fc *funcLower) nextTemps(n int) []string {
	temps := make([]string, n)
	for i := range temps {
		temps[i] = fmt.Sprintf("_logos_v%d", fc.tmpSeq)
		fc.tmpSeq++
	}
	return temps
}

func tempNodes(pos syntax.Pos, spellings []string) []syntax.Expr {
	nodes := make([]syntax.Expr, len(spellings))
	for i, spelling := range spellings {
		nodes[i] = nameOf(pos, spelling)
	}
	return nodes
}

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
