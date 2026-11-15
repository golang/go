// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"go/token"
	"slices"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

const parCtxName = "_logos_pc"

const (
	parResultName = "_logos_lr0"
	parErrorName  = "_logos_lr1"
)

type parSource int

const (
	parSlice parSource = iota
	parSeq
	parChan
)

type parStage struct {
	unordered bool
	source    parSource
	elemT     types2.Type
	resT      types2.Type
	recv      syntax.Expr
	recvSpan  span
	ctx       syntax.Expr
	ctxSpan   span
	limit     syntax.Expr
	limSpan   span
	seg       []chainOp
	segT      []types2.Type
	post      []chainOp
	term      chainOp
	owned     []bool
	ownedOut  bool
}

func (fl *fileLower) parHead(e syntax.Expr) (group *syntax.CallExpr, unordered bool, elem types2.Type, ok bool) {
	call, isCall := syntax.Unparen(e).(*syntax.CallExpr)
	if !isCall {
		return nil, false, nil, false
	}
	elem, ok = types2.LogosGroupStage(fl.pkg.tpkg, fl.pkg.info, call)
	if !ok {
		return nil, false, nil, false
	}
	group = call
	if chainMethodOf(call) == "unordered" {
		unordered = true
		group = syntax.Unparen(syntax.Unparen(call.Fun).(*syntax.SelectorExpr).X).(*syntax.CallExpr)
	}
	return group, unordered, elem, true
}

func (fl *fileLower) isParStage(base syntax.Expr) bool {
	_, _, _, ok := fl.parHead(base)
	return ok
}

func parCalls(call *syntax.CallExpr, base syntax.Expr) []*syntax.CallExpr {
	var out []*syntax.CallExpr
	for cur := syntax.Expr(call); cur != base; {
		c := syntax.Unparen(cur).(*syntax.CallExpr)
		out = append(out, c)
		cur = syntax.Unparen(c.Fun).(*syntax.SelectorExpr).X
	}
	return out
}

func parElem(t types2.Type) types2.Type {
	if t == nil {
		return nil
	}
	if s, ok := types2.Unalias(parResult(t)).(*types2.Slice); ok {
		return s.Elem()
	}
	return nil
}

func (fl *fileLower) parSourceOf(recv syntax.Expr) parSource {
	t := fl.pkg.info.TypeOf(recv)
	if c := constAsView(t); c != nil {
		t = c.Elem
	}
	switch types2.Unalias(t).Underlying().(type) {
	case *types2.Chan:
		return parChan
	case *types2.Signature:
		return parSeq
	}
	return parSlice
}

func (fl *fileLower) parArgSpans(group *syntax.CallExpr) (ctx, limit span) {
	split, ok := splitCall(fl.src, fl.off(group.Pos()))
	if !ok || len(split.args) != 2 {
		base.Fatalf("logos: cannot locate the group stage arguments at %s", group.Pos().String())
	}
	ctx = trimSpan(fl.src, split.args[0])
	arg := split.args[1]
	eq := -1
	scanSegment(fl.src, arg.start, arg.end, func(t tok) bool {
		if t.kind == token.ASSIGN {
			eq = t.end
		}
		return eq < 0
	})
	if eq < 0 {
		base.Fatalf("logos: cannot locate the group stage limit at %s", group.Pos().String())
	}
	return ctx, trimSpan(fl.src, span{start: eq, end: arg.end})
}

func (fl *fileLower) parStageOf(call *syntax.CallExpr, ops []chainOp, base syntax.Expr, term chainOp) *parStage {
	group, unordered, elem, _ := fl.parHead(base)
	sel := syntax.Unparen(group.Fun).(*syntax.SelectorExpr)
	st := &parStage{unordered: unordered, elemT: elem, term: term, recv: sel.X}
	st.source = fl.parSourceOf(st.recv)
	st.recvSpan = fl.receiverSpan(group)
	st.ctx = group.ArgList[0]
	st.limit = group.ArgList[1].(*syntax.KeyValueExpr).Value
	st.ctxSpan, st.limSpan = fl.parArgSpans(group)
	calls := parCalls(call, base)
	if term.method != "" {
		calls = calls[1:]
	}
	inner := slices.Clone(ops)
	slices.Reverse(inner)
	slices.Reverse(calls)
	n := 0
	for n < len(inner) && (inner[n].method == "map" || inner[n].method == "filter") {
		n++
	}
	st.seg, st.post = inner[:n], inner[n:]
	for _, c := range calls[:n] {
		st.segT = append(st.segT, parElem(fl.pkg.info.TypeOf(c)))
	}
	return st
}

func (cg *chainGen) parCapture(op *chainOp) {
	if op.lam != nil {
		cg.fl.preLowerCaptureBody(op.lam)
		op.savedBody = op.lam.Body
	}
	cg.lowerArgChains(op.arg)
}

func (cg *chainGen) parPrepare(st *parStage) {
	cg.fl.needRT = true
	for i := range st.seg {
		cg.parCapture(&st.seg[i])
	}
	for i := range st.post {
		cg.parCapture(&st.post[i])
	}
	cg.parCapture(&st.term)
	cg.lowerArgChains(st.ctx)
	cg.lowerArgChains(st.limit)
	cg.lowerArgChains(st.recv)
	in := st.elemT
	for i, op := range st.seg {
		out := st.segT[i]
		if op.method == "filter" {
			out = in
		} else if out == nil {
			out = cg.prodType(op)
		}
		if out == nil {
			out = in
		}
		if isUntypedBasic(out) {
			out = defaultBasicType(out)
		}
		st.segT[i] = out
		in = out
	}
	st.resT = in
	cg.elemT = in
	st.ownership()
}

func (st *parStage) kept() []span {
	out := []span{st.recvSpan, st.ctxSpan, st.limSpan, st.term.argSpan}
	for _, op := range st.seg {
		out = append(out, op.argSpan)
	}
	for _, op := range st.post {
		out = append(out, op.argSpan)
	}
	return out
}

func (cg *chainGen) parSeqText(st *parStage) ([]otext, otext) {
	fl := cg.fl
	p := cg.prefix
	s, r := cg.spellType(st.elemT), cg.spellType(st.resT)
	verb := "GroupSeq"
	if st.source == parChan {
		verb = "GroupChan"
	}
	if st.unordered {
		verb += "Unordered"
	}
	srcN, ctxN, limN := p+"_src", p+"_ctx", p+"_lim"
	binds := []otext{
		cat(gen(srcN+" := "), fl.textOf(st.recvSpan)),
		cat(gen(ctxN+" := "), fl.textOf(st.ctxSpan)),
		cat(gen(limN+" := "), fl.textOf(st.limSpan)),
	}
	for i, op := range st.seg {
		if op.lam == nil {
			binds = append(binds, cat(gen(cg.parOpName(i)+" := "), fl.textOf(op.argSpan)))
		}
	}
	child := cg.parChild(st)
	var src otext
	switch st.source {
	case parSlice:
		y, e := p+"_y", p+"_s"
		src = gen("func(" + y + " func(" + s + ") bool) { for _, " + e + " := range " + srcN + " { if !" + y + "(" + e + ") { return } } }")
	case parSeq:
		src = gen("(func(func(" + s + ") bool))(" + srcN + ")")
	default:
		src = gen(srcN)
	}
	return binds, cat(gen(rtAlias+"."+verb+"["+s+", "+r+"]("+ctxN+", "+limN+", "), src, gen(", "), child, gen(")"))
}

func (cg *chainGen) parOpName(i int) string {
	return fmt.Sprintf("%s_c%d", cg.prefix, i)
}

func (cg *chainGen) parChild(st *parStage) otext {
	p := cg.prefix
	ps := p + "_ps"
	rs := cg.spellType(st.resT)
	parts := []otext{gen("func(" + parCtxName + " " + rtAlias + ".Context, " + ps + " " + cg.spellType(st.elemT) + ") (" + rs + ", bool, error) { ")}
	if name, ok := plainContextName(st.ctx); ok {
		parts = append(parts, gen(name+" := "+parCtxName+"; _ = "+name+"; "))
	}
	in := st.elemT
	for i, op := range st.seg {
		if op.lam != nil {
			parts = append(parts, cg.parClosure(cg.parOpName(i), op, in, st.segT[i]), gen("; "))
		}
		in = st.segT[i]
	}
	cur := ps
	for i := range st.seg {
		var line string
		line, cur = cg.parStep(st, i, cur)
		parts = append(parts, gen(line+"; "))
	}
	ret := "return " + cur + ", true, nil }"
	if logosMayBeCloser(st.resT) && !st.ownedOut {
		ret = "return " + cur + ", true, " + rtAlias + ".ErrNotOwned }"
	}
	parts = append(parts, gen(ret))
	return cat(parts...)
}

func parMapAfter(seg []chainOp, i int) bool {
	for _, op := range seg[i+1:] {
		if op.method == "map" {
			return true
		}
	}
	return false
}

func (st *parStage) inType(i int) types2.Type {
	if i == 0 {
		return st.elemT
	}
	return st.segT[i-1]
}

func (st *parStage) ownership() {
	st.owned = make([]bool, len(st.seg))
	st.ownedOut = false
	for i, op := range st.seg {
		st.owned[i] = st.ownedOut
		if op.method == "map" {
			in, out := st.inType(i), st.segT[i]
			st.ownedOut = logosMayBeCloser(out) && formsOwn(op.forms, in != nil && out != nil && types2.Identical(in, out), st.ownedOut)
		}
	}
}

func (cg *chainGen) parCloseOne(i int, v string) string {
	p := cg.prefix
	return fmt.Sprintf("if %[1]s_cc%[2]d, %[1]s_ok%[2]d := any(%[3]s).(interface{ Close() error }); %[1]s_ok%[2]d { %[1]s_cc%[2]d.Close() }", p, i, v)
}

func (cg *chainGen) parNotOwned(st *parStage) string {
	return cg.zeroText(st.resT) + ", false, " + rtAlias + ".ErrNotOwned"
}

func (cg *chainGen) parDrop(st *parStage, i int, cur string) string {
	closes := st.owned[i] && logosMayBeCloser(st.inType(i))
	switch {
	case closes && !parMapAfter(st.seg, i):
		return "return " + cur + ", false, nil"
	case closes:
		return cg.parCloseOne(i, cur) + "; return " + cg.parNotOwned(st)
	case logosMayBeCloser(st.resT):
		return "return " + cg.parNotOwned(st)
	case parMapAfter(st.seg, i):
		return "return " + cg.zeroText(st.resT) + ", false, nil"
	}
	return "return " + cur + ", false, nil"
}

func (cg *chainGen) parStep(st *parStage, i int, cur string) (line, next string) {
	op := st.seg[i]
	p := cg.prefix
	fn := cg.parOpName(i)
	errN := p + "_e"
	zero := cg.zeroText(st.resT)
	fail := "return " + zero + ", false, " + errN
	if op.method == "filter" {
		drop := cg.parDrop(st, i, cur)
		if op.lam == nil {
			return fmt.Sprintf("if !%s(%s) { %s }", fn, cur, drop), cur
		}
		keep := fmt.Sprintf("%s_k%d", p, i)
		examined := fail
		if st.owned[i] && logosMayBeCloser(st.inType(i)) {
			examined = cg.parCloseOne(i, cur) + "; " + fail
		}
		return fmt.Sprintf("%s, %s := %s(%s); if %s != nil { %s }; if !%s { %s }", keep, errN, fn, cur, errN, examined, keep, drop), cur
	}
	next = fmt.Sprintf("%s_v%d", p, i)
	if op.lam == nil {
		return next + " := " + fn + "(" + cur + ")", next
	}
	return fmt.Sprintf("%s, %s := %s(%s); if %s != nil { %s }", next, errN, fn, cur, errN, fail), next
}

func (cg *chainGen) parClosure(name string, op chainOp, in, out types2.Type) otext {
	fl := cg.fl
	result, zero, resT := "bool", "false", types2.Type(types2.Typ[types2.Bool])
	if op.method == "map" {
		result, zero, resT = cg.spellType(out), cg.zeroText(out), out
	}
	param := op.lam.Params[0].Value + " " + cg.spellType(in)
	if op.isBlk {
		return cg.parBlockClosure(name, op, in, resT)
	}
	head := name + " := func(" + param + ") (" + result + ", error) {"
	fc := fl.parFuncLower(resT)
	if op.isTry {
		return cg.parTryClosure(fc, head, op, zero)
	}
	body, ok := op.savedBody.(syntax.Expr)
	if !ok {
		base.Fatalf("logos: group stage lambda at %s has no expression body", op.lam.Pos().String())
	}
	pre := fc.parHoist(body)
	return cat(gen(head+" "), pre, gen("return "), fl.drainBodyWrapper(body, fl.textOf(op.bodySpan)), gen(", nil }"))
}

func (fl *fileLower) parFuncLower(res types2.Type) *funcLower {
	return &funcLower{fl: fl, groupStack: parGroupStack(res)}
}

func parGroupStack(res types2.Type) []groupFrame {
	return []groupFrame{{gctx: parCtxName, child: true, segment: true, segmentRes: res}}
}

func (fc *funcLower) parHoist(e syntax.Expr) otext {
	defer fc.groupClosureOwner()()
	scanForLiterals(fc.fl, e)
	top, _ := fc.groupTop()
	mark := fc.hoistSubMark()
	h := fc.newGroupHoist(top, fc.fl.exprStart(e))
	h.expr(e)
	fc.hoistSubEmit(fc.hoistSubTake(mark))
	return h.render()
}

func (cg *chainGen) parTryClosure(fc *funcLower, head string, op chainOp, zero string) otext {
	fl := cg.fl
	body, _ := op.savedBody.(syntax.Expr)
	try, _ := syntax.Unparen(body).(*syntax.TryExpr)
	var inner *syntax.CallExpr
	if try != nil {
		inner, _ = syntax.Unparen(try.X).(*syntax.CallExpr)
	}
	if inner == nil {
		base.Fatalf("logos: group stage lambda at %s needs a call before ?", op.lam.Pos().String())
	}
	pre := fc.parHoist(inner)
	call := fl.textOf(span{start: fl.exprStart(inner), end: fl.chainCallEnd(inner)})
	v, e := cg.prefix+"_tv", cg.prefix+"_te"
	return cat(gen(head+" "), pre, gen(v+", "+e+" := "), call, gen("; if "+e+" != nil { return "+zero+", "+e+" }; return "+v+", nil }"))
}

func (cg *chainGen) parBlockClosure(name string, op chainOp, in, res types2.Type) otext {
	fl := cg.fl
	block, ok := op.savedBody.(*syntax.BlockStmt)
	if !ok {
		base.Fatalf("logos: group stage lambda at %s has no block body", op.lam.Pos().String())
	}
	sig := "func(" + op.lam.Params[0].Value + " " + cg.spellType(in) + ") (" + parResultName + " " + cg.spellType(res) + ", " + parErrorName + " error)"
	ft, ok := typeExprAt(sig, op.lam.Pos())
	if !ok {
		base.Fatalf("logos: cannot build the group stage lambda at %s", op.lam.Pos().String())
	}
	fc := newFuncLower(fl, ft.(*syntax.FuncType), fl.parSignature(op.lam.Params[0].Value, in, res), block)
	fc.groupStack = parGroupStack(res)
	fc.elementFrame = true
	fc.run()
	text := fl.textOf(span{start: fl.off(syntax.StartPos(block)), end: fl.off(syntax.EndPos(block))})
	if strings.HasPrefix(text.s, "{") {
		text = text.slice(1, len(text.s))
	}
	return cat(gen(name+" := "+sig+" {"), text, gen(" }"))
}

func (fl *fileLower) parSignature(param string, in, res types2.Type) *types2.Signature {
	pkg := fl.pkg.tpkg
	params := types2.NewTuple(types2.NewParam(syntax.Pos{}, pkg, param, in))
	results := types2.NewTuple(
		types2.NewParam(syntax.Pos{}, pkg, parResultName, res),
		types2.NewParam(syntax.Pos{}, pkg, parErrorName, types2.Universe.Lookup("error").Type()))
	return types2.NewSignatureType(nil, nil, nil, params, results, false)
}

func (cg *chainGen) parPost(st *parStage, call *syntax.CallExpr) *chainEmit {
	ce := &chainEmit{cg: cg, ops: st.post, term: st.term, elemT: st.resT, resultT: parResult(cg.fl.pkg.info.TypeOf(call)), isTry: true, ownedIn: st.ownedOut}
	ce.prepare()
	ce.outCap, ce.hasCap = "", false
	return ce
}

func parResult(t types2.Type) types2.Type {
	if tup, ok := t.(*types2.Tuple); ok && tup.Len() == 2 {
		return tup.At(0).Type()
	}
	return t
}

func (fl *fileLower) isParChain(e syntax.Expr) bool {
	for {
		call, ok := syntax.Unparen(e).(*syntax.CallExpr)
		if !ok {
			return false
		}
		if fl.isParStage(call) {
			return true
		}
		if !fl.isChainCall(call) {
			return false
		}
		e = syntax.Unparen(call.Fun).(*syntax.SelectorExpr).X
	}
}

func (cg *chainGen) runPar(call *syntax.CallExpr, ops []chainOp, base syntax.Expr, term chainOp, carrier *syntax.CallExpr) {
	if cg.fl.isParStage(base) {
		cg.runParStage(call, ops, base, term)
		return
	}
	cg.runParTail(call, ops, base, term, carrier)
}

func (cg *chainGen) runParStage(call *syntax.CallExpr, ops []chainOp, base syntax.Expr, term chainOp) {
	fl := cg.fl
	st := fl.parStageOf(call, ops, base, term)
	cg.whole.start, cg.whole.end = fl.chainSpan(call)
	cg.kept = st.kept()
	cg.parPrepare(st)
	ce := cg.parPost(st, call)
	binds, seq := cg.parSeqText(st)
	cg.parReplace(call, ce, cg.parLoopText(ce, binds, seq))
}

func (cg *chainGen) runParTail(call *syntax.CallExpr, ops []chainOp, base syntax.Expr, term chainOp, carrier *syntax.CallExpr) {
	fl := cg.fl
	slices.Reverse(ops)
	recv := fl.receiverSpan(carrier)
	cg.whole.start, cg.whole.end = fl.chainSpan(call)
	cg.kept = keptSpans(recv, ops, term)
	for i := range ops {
		cg.parCapture(&ops[i])
	}
	cg.parCapture(&term)
	cg.lowerArgChains(base)
	cg.elemT = cg.baseElem(base)
	p := cg.prefix
	in, inErr := p+"_in", p+"_ierr"
	ce := &chainEmit{cg: cg, ops: ops, term: term, baseText: gen(in), elemT: cg.elemT, resultT: parResult(fl.pkg.info.TypeOf(call)), isTry: true}
	ce.prepare()
	prelude, body, epilogue := ce.runLoopO()
	parts := []otext{cat(gen(in+", "+inErr+" := "), fl.textOf(recv), gen("; if "+inErr+" != nil { "+ce.failReturn(p, inErr)+" }; "))}
	for _, line := range prelude {
		parts = append(parts, line, gen("; "))
	}
	for _, line := range append(body, epilogue...) {
		parts = append(parts, gen(line), gen("; "))
	}
	cg.parReplace(call, ce, cat(parts...).trimRight(" "))
}

func (cg *chainGen) parReplace(call *syntax.CallExpr, ce *chainEmit, loop otext) {
	fl := cg.fl
	text := cat(gen("func() ("+ce.resultSpelling()+", error) { "), cat(leftoverComments(fl, cg.whole, cg.kept), loop), gen(" }()"))
	fl.stageReplacement(call, parseStageFrag(fl, text.s, leftmostOf(call)))
	fl.addT(cg.whole.start, cg.whole.end, text)
}

func (cg *chainGen) parLoopText(ce *chainEmit, binds []otext, seq otext) otext {
	p := cg.prefix
	prelude, body, epilogue := ce.loopO()
	var parts []otext
	for _, bind := range binds {
		parts = append(parts, bind, gen("; "))
	}
	for _, line := range prelude[1:] {
		parts = append(parts, line, gen("; "))
	}
	head := fmt.Sprintf("for %[1]s_e, %[1]s_err := range ", p)
	if !ce.needElem() {
		head = fmt.Sprintf("for _, %[1]s_err := range ", p)
	}
	parts = append(parts, gen(head), seq, gen(" { "))
	var breaks []string
	for i, op := range ce.ops {
		if op.method == "take" {
			breaks = append(breaks, fmt.Sprintf("if %[1]s_tc%[2]d >= %[1]s_tn%[2]d { break }", p, i))
		}
	}
	check := fmt.Sprintf("if %s_err != nil { %s }", p, ce.failStage(p, p+"_err"))
	lines := slices.Insert(slices.Clone(body[1:len(body)-1]), len(breaks), check)
	lines = append(append(lines, breaks...), "}")
	for _, line := range lines {
		parts = append(parts, gen(line), gen("; "))
	}
	for _, line := range epilogue {
		parts = append(parts, gen(line), gen("; "))
	}
	return cat(parts...).trimRight(" ")
}

func (fl *fileLower) parRangeParts(s syntax.Stmt) (*syntax.ForStmt, *syntax.RangeClause, *syntax.CallExpr, bool) {
	for {
		lab, ok := s.(*syntax.LabeledStmt)
		if !ok {
			break
		}
		s = lab.Stmt
	}
	fs, ok := s.(*syntax.ForStmt)
	if !ok {
		return nil, nil, nil, false
	}
	rc, ok := fs.Init.(*syntax.RangeClause)
	if !ok {
		return nil, nil, nil, false
	}
	call, ok := syntax.Unparen(rc.X).(*syntax.CallExpr)
	if !ok || !fl.isChainCall(call) {
		return nil, nil, nil, false
	}
	return fs, rc, call, true
}

func (fl *fileLower) lowerParRange(s syntax.Stmt, enc []resultSpec) ([]syntax.Stmt, bool) {
	fs, rc, call, ok := fl.parRangeParts(s)
	if !ok {
		return nil, false
	}
	ops, base, term, _ := fl.decomposeChain(call)
	if term.method != "" || !fl.isParStage(base) {
		return nil, false
	}
	fl.lowerChainHeadLits(rc)
	fl.loweredChains[call] = true
	cg := &chainGen{fl: fl, pos: call.Pos(), prefix: chainPrefix(call.Pos()), enc: enc}
	st := fl.parStageOf(call, ops, base, term)
	cg.parPrepare(st)
	binds, seq := cg.parSeqText(st)
	var out []syntax.Stmt
	start := fl.stmtStartOf(s)
	for _, h := range cg.hoists {
		out = append(out, h.stmts...)
		fl.addT(start, start, cat(h.text, gen("; ")))
	}
	for _, bind := range binds {
		out = append(out, parseChainStmtFrag(fl, bind.s, call.Pos())...)
		fl.addT(start, start, cat(bind, gen("; ")))
	}
	errName := chainErrName(fs.Pos())
	if !rc.Def && rc.Lhs != nil {
		out = append(out, fl.declareChainErr(s, errName)...)
	}
	check, checkText := fl.chainErrCheck(enc, fs.Body.Pos(), errName)
	fl.parRangeHead(fs, rc, seq, errName)
	open := fl.off(fs.Body.Pos()) + 1
	fl.add(open, open, " "+checkText+"; ")
	fl.lowerChainChildren(s, enc)
	pos := fs.Pos()
	names := []syntax.Expr{nameOf(pos, "_")}
	if rc.Lhs != nil {
		names = syntax.UnpackListExpr(rc.Lhs)
	} else {
		rc.Def = true
	}
	rc.Lhs = listOf(pos, append(slices.Clone(names), nameOf(pos, errName)))
	rc.X = parseStageFrag(fl, seq.s, pos)
	fs.Body.List = append(check, fs.Body.List...)
	return append(out, s), true
}

func (fl *fileLower) parRangeHead(fs *syntax.ForStmt, rc *syntax.RangeClause, seq otext, errName string) {
	lhs, assign := gen("_"), gen(" := ")
	if rc.Lhs != nil {
		start := fl.off(leftmostOf(rc.Lhs))
		rangeOff := fl.off(rc.Pos())
		split := rangeOff
		depth := 0
		scanSegment(fl.src, start, rangeOff, func(t tok) bool {
			if t.comment || t.auto {
				return true
			}
			switch t.kind {
			case token.LPAREN, token.LBRACK, token.LBRACE:
				depth++
			case token.RPAREN, token.RBRACK, token.RBRACE:
				depth--
			case token.DEFINE, token.ASSIGN:
				if depth == 0 {
					split = t.start
					return false
				}
			}
			return true
		})
		lhs = fl.srcUser(start, split)
		assign = fl.srcUser(split, rangeOff)
	}
	fl.addT(fl.off(fs.Pos()), fl.off(fs.Body.Pos()), cat(gen("for "), lhs, gen(", "+errName), assign, gen("range "), seq, gen(" ")))
}

func (fl *fileLower) parReturn(s *syntax.ReturnStmt) {
	if s.Results == nil {
		return
	}
	end := stmtEnd(fl.src, fl.off(s.Pos()))
	fl.add(end, end, ", nil")
}

func parseStageFrag(fl *fileLower, text string, pos syntax.Pos) syntax.Expr {
	text = fl.settleMarks(gen(text)).s
	src := "package logoschain\nvar _ = " + text + "\n"
	f, err := syntax.Parse(syntax.NewFileBase("logos-chain"), strings.NewReader(src), nil, nil, syntax.CheckBranches)
	if err != nil || len(f.DeclList) != 1 {
		base.Fatalf("logos: cannot parse chain fragment %q: %v", text, err)
	}
	decl, ok := f.DeclList[0].(*syntax.VarDecl)
	if !ok {
		base.Fatalf("logos: cannot parse chain fragment %q", text)
	}
	vals := syntax.UnpackListExpr(decl.Values)
	if len(vals) != 1 {
		base.Fatalf("logos: cannot parse chain fragment %q", text)
	}
	normalizeChainPos(vals[0], pos)
	return vals[0]
}
