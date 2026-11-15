// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"go/token"
	"strconv"

	"cmd/compile/internal/syntax"
)

func (h *groupHoist) child() *groupHoist {
	return &groupHoist{fc: h.fc, top: h.top, from: h.from, pos: h.pos}
}

func (h *groupHoist) scopePos(p syntax.Pos) syntax.Pos {
	if h.pos.IsKnown() {
		return h.pos
	}
	return p
}

func (h *groupHoist) shortCircuit(op *syntax.Operation) syntax.Expr {
	fc := h.fc
	fl := fc.fl
	pos := h.scopePos(op.Pos())
	whole, leftSpan, rightSpan := h.span(op), h.span(op.X), h.span(op.Y)
	mark := fc.hoistSubMark()
	left := h.expr(op.X)
	right := h.child()
	right.pos = pos
	rightExpr := right.expr(op.Y)
	fc.hoistSubEmit(fc.hoistSubTake(mark))
	leftText, rightText := fl.textOf(leftSpan), fl.textOf(rightSpan)
	t := fc.groupTempName("t")
	declN, declT := h.valueTemp(pos, op, t, left, leftText)
	var guard syntax.Expr = nameOf(pos, t)
	negate := ""
	if op.Op == syntax.OrOr {
		guard = at(&syntax.Operation{Op: syntax.Not, X: guard}, pos)
		negate = "!"
	}
	set := assignOf(pos, []syntax.Expr{nameOf(pos, t)}, []syntax.Expr{rightExpr})
	body := append(right.native, set)
	bodyText := append(right.text, cat(gen(t+" = "), rightText))
	h.add(
		append(declN, ifOf(pos, nil, guard, body)),
		append(declT, cat(gen("if "+negate+t+" {\n"), joinO(bodyText, "\n"), gen("\n}")))...,
	)
	fc.hoistSubs = append(fc.hoistSubs, hoistSub{start: whole.start, end: whole.end, text: "(" + t + " == true)"})
	return at(&syntax.ParenExpr{X: at(&syntax.Operation{Op: syntax.Eql, X: nameOf(pos, t), Y: nameOf(pos, "true")}, pos)}, syntax.StartPos(op))
}

func (fl *fileLower) groupSimpleStart(s syntax.Stmt) int {
	switch n := s.(type) {
	case *syntax.ExprStmt:
		return fl.exprStart(n.X)
	case *syntax.SendStmt:
		return fl.exprStart(n.Chan)
	case *syntax.AssignStmt:
		return fl.exprStart(n.Lhs)
	}
	return fl.off(syntax.StartPos(s))
}

func (fl *fileLower) groupHeaderSemis(from, to int) []span {
	var out []span
	depth := 0
	scanSegment(fl.src, from, to, func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			depth--
		case token.SEMICOLON:
			if depth == 0 {
				out = append(out, span{start: t.start, end: t.end})
			}
		}
		return true
	})
	return out
}

func (h *groupHoist) send(s *syntax.SendStmt) (syntax.Stmt, otext, span) {
	fc := h.fc
	fl := fc.fl
	pos := h.scopePos(s.Pos())
	chSpan, valSpan := h.span(s.Chan), h.span(s.Value)
	mark := fc.hoistSubMark()
	h.slots([]*syntax.Expr{&s.Chan, &s.Value})
	fc.hoistSubEmit(fc.hoistSubTake(mark))
	chText, valText := fl.textOf(chSpan), fl.textOf(valSpan)
	send := at(&syntax.SendStmt{Chan: s.Chan, Value: s.Value}, pos)
	sel := at(&syntax.SelectStmt{
		Body: []*syntax.CommClause{
			at(&syntax.CommClause{Comm: send}, pos),
			fc.groupDoneClause(pos, h.top),
		},
	}, pos)
	text := cat(gen("select { case "), chText, gen(" <- "), valText, gen(": "+fc.groupDoneText(h.top)+" }"))
	return sel, text, span{start: chSpan.start, end: valSpan.end}
}

func (h *groupHoist) recvStmt(s *syntax.ExprStmt, op *syntax.Operation) (syntax.Stmt, otext, span) {
	fc := h.fc
	whole := h.span(op)
	nch, chText, _ := h.channel(op)
	pos := h.scopePos(s.Pos())
	recv := at(&syntax.Operation{Op: syntax.Recv, X: nch}, pos)
	sel := at(&syntax.SelectStmt{
		Body: []*syntax.CommClause{
			at(&syntax.CommClause{Comm: exprStmtOf(pos, recv)}, pos),
			fc.groupDoneClause(pos, h.top),
		},
	}, pos)
	return sel, cat(gen("select { case <-"), chText, gen(": "+fc.groupDoneText(h.top)+" }")), whole
}

func (h *groupHoist) stmt(s syntax.Stmt, sp span) {
	fc := h.fc
	fl := fc.fl
	mark := fc.hoistSubMark()
	switch n := s.(type) {
	case *syntax.SendStmt:
		sel, text, _ := h.send(n)
		h.add([]syntax.Stmt{sel}, text)
		return
	case *syntax.ExprStmt:
		if op, ok := groupRecvOp(n.X); ok {
			sel, text, _ := h.recvStmt(n, op)
			h.add([]syntax.Stmt{sel}, text)
			return
		}
		n.X = h.statement(n.X)
	case *syntax.AssignStmt:
		h.assign(n)
	}
	fc.hoistSubEmit(fc.hoistSubTake(mark))
	if es, ok := s.(*syntax.ExprStmt); ok && es.X == nil {
		_ = fl.textOf(sp)
		return
	}
	if h.tryInit(s) {
		return
	}
	h.add([]syntax.Stmt{s}, fl.textOf(sp))
}

func (h *groupHoist) tryInit(s syntax.Stmt) bool {
	init, ok := s.(syntax.SimpleStmt)
	if !ok {
		return false
	}
	core, isTry := h.fc.initCoreOf(init)
	if !isTry {
		return false
	}
	native, text, _, _ := core()
	anchorScopes(native, h.scopePos(s.Pos()))
	h.add(native, text)
	return true
}

func (h *groupHoist) moveInit(init syntax.SimpleStmt, limit int) (span, span) {
	fl := h.fc.fl
	start := fl.groupSimpleStart(init)
	semi := fl.groupHeaderSemis(start, limit)[0]
	sp := trimSpan(fl.src, span{start: start, end: semi.start})
	h.stmt(init, sp)
	return sp, semi
}

func (fl *fileLower) groupInitClosure(init syntax.SimpleStmt) bool {
	stmt, ok := init.(*syntax.ExprStmt)
	if !ok {
		return false
	}
	closure := fl.groupClosureCall(stmt.X)
	return closure != nil && !closure.lifted
}

func (fc *funcLower) groupHoistIf(s *syntax.IfStmt, top groupFrame) []syntax.Stmt {
	fl := fc.fl
	scanForLiterals(fl, s.Init)
	scanForLiterals(fl, s.Cond)
	start := fc.hostStart(s)
	mark := fc.hoistSubMark()
	h := fc.newGroupHoist(top, start)
	h.atStatement(s)
	h.pos = s.Pos()
	condRecv := s.Cond != nil && fl.groupNodesHaveRecv(s.Cond)
	split := s.Init != nil && (condRecv || fl.groupInitClosure(s.Init) || fc.groupTryInit(s.Init))
	switch {
	case split:
		moved, semi := h.moveInit(s.Init, fl.off(s.Then.Pos()))
		fl.add(moved.start, semi.end, "")
		s.Init = nil
	default:
		if as, ok := s.Init.(*syntax.AssignStmt); ok {
			h.assign(as)
		}
	}
	if condRecv {
		s.Cond = h.expr(s.Cond)
	}
	h.before(mark, start)
	out := h.scoped(s)
	s.Then.List = fc.lowerList(s.Then.List)
	fc.groupHoistElse(s, top)
	return out
}

func (fc *funcLower) groupHoistElse(s *syntax.IfStmt, top groupFrame) {
	if s.Else == nil {
		return
	}
	switch e := s.Else.(type) {
	case *syntax.BlockStmt:
		e.List = fc.lowerList(e.List)
	case *syntax.IfStmt:
		if !fc.fl.groupStmtHasRecv(e) {
			s.Else = fc.lowerIf(e)[0]
			return
		}
		s.Else = fc.groupHoistIf(e, top)[0]
	}
}

func (fl *fileLower) groupSwitchHeaderEnd(s *syntax.SwitchStmt) int {
	if len(s.Body) > 0 {
		return fl.off(s.Body[0].Pos())
	}
	return fl.off(s.Rbrace)
}

func (fc *funcLower) groupHoistSwitch(s *syntax.SwitchStmt, top groupFrame) []syntax.Stmt {
	fl := fc.fl
	scanForLiterals(fl, s.Init)
	scanForLiterals(fl, s.Tag)
	for _, c := range s.Body {
		scanForLiterals(fl, c.Cases)
	}
	start := fc.hostStart(s)
	mark := fc.hoistSubMark()
	h := fc.newGroupHoist(top, start)
	h.atStatement(s)
	h.pos = s.Pos()
	tagRecv := s.Tag != nil && fl.groupNodesHaveRecv(s.Tag)
	if fl.groupCasesHaveRecv(s) {
		return fc.groupSwitchCases(s, h, mark, start)
	}
	split := s.Init != nil && (tagRecv || fl.groupInitClosure(s.Init) || fc.groupTryInit(s.Init))
	if split {
		moved, semi := h.moveInit(s.Init, fl.groupSwitchHeaderEnd(s))
		fl.add(moved.start, semi.end, "")
		s.Init = nil
	} else if as, ok := s.Init.(*syntax.AssignStmt); ok {
		h.assign(as)
	}
	if tagRecv {
		s.Tag = h.expr(s.Tag)
	}
	h.before(mark, start)
	out := h.scoped(s)
	for _, c := range s.Body {
		c.Body = fc.lowerList(c.Body)
	}
	return out
}

func (fc *funcLower) groupSwitchCases(s *syntax.SwitchStmt, h *groupHoist, mark, start int) []syntax.Stmt {
	fl := fc.fl
	pos := s.Pos()
	headStart := fl.off(pos) + len("switch")
	headEnd := headStart
	if s.Init != nil {
		_, semi := h.moveInit(s.Init, fl.groupSwitchHeaderEnd(s))
		headEnd = semi.end
		s.Init = nil
	}
	tagName := ""
	if s.Tag != nil {
		tagSpan := h.span(s.Tag)
		headEnd = tagSpan.end
		tagName = h.cacheTag(s.Tag, tagSpan)
	}
	sel := fc.groupTempName("sel")
	h.add(
		[]syntax.Stmt{typedVarDecl(pos, nameOf(pos, sel), nameOf(pos, "int"))},
		gen("var "+sel+" int"),
	)
	for k, c := range s.Body {
		if c.Cases == nil {
			continue
		}
		label := strconv.Itoa(k + 1)
		for _, e := range syntax.UnpackListExpr(c.Cases) {
			h.matchCase(sel, label, tagName, e)
		}
		caseStart := fl.off(c.Pos())
		fl.add(caseStart, fl.off(c.Colon), "case "+label)
		c.Cases = litOf(c.Pos(), label, syntax.IntLit)
	}
	fl.add(headStart, headEnd, " "+sel+" ")
	s.Tag = nameOf(pos, sel)
	h.before(mark, start)
	out := h.scoped(s)
	for _, c := range s.Body {
		c.Body = fc.lowerList(c.Body)
	}
	return out
}

func (h *groupHoist) cacheTag(tag syntax.Expr, sp span) string {
	fc := h.fc
	name := fc.groupTempName("tag")
	mark := fc.hoistSubMark()
	value := h.expr(tag)
	fc.hoistSubEmit(fc.hoistSubTake(mark))
	native, texts := h.valueTemp(tag.Pos(), tag, name, value, fc.fl.textOf(sp))
	h.add(native, texts...)
	return name
}

func (h *groupHoist) matchCase(sel, label, tagName string, e syntax.Expr) {
	fc := h.fc
	fl := fc.fl
	pos := h.scopePos(e.Pos())
	sp := h.span(e)
	child := h.child()
	mark := fc.hoistSubMark()
	value := child.expr(e)
	fc.hoistSubEmit(fc.hoistSubTake(mark))
	text := fl.textOf(sp)
	cond := value
	if tagName != "" {
		cond = at(&syntax.Operation{Op: syntax.Eql, X: nameOf(pos, tagName), Y: value}, pos)
		text = cat(gen(tagName+" == ("), text, gen(")"))
	}
	pick := assignOf(pos, []syntax.Expr{nameOf(pos, sel)}, []syntax.Expr{litOf(pos, label, syntax.IntLit)})
	body := append(child.native, ifOf(pos, nil, cond, []syntax.Stmt{pick}))
	bodyText := append(child.text, cat(gen("if "), text, gen(" {\n"+sel+" = "+label+"\n}")))
	unset := at(&syntax.Operation{Op: syntax.Eql, X: nameOf(pos, sel), Y: litOf(pos, "0", syntax.IntLit)}, pos)
	h.add(
		[]syntax.Stmt{ifOf(pos, nil, unset, body)},
		cat(gen("if "+sel+" == 0 {\n"), joinO(bodyText, "\n"), gen("\n}")),
	)
}

func (fc *funcLower) groupHoistFor(s *syntax.ForStmt, top groupFrame) []syntax.Stmt {
	fl := fc.fl
	if rc, ok := s.Init.(*syntax.RangeClause); ok {
		return fc.groupHoistRange(s, rc, top)
	}
	pos := s.Pos()
	keyword := fl.off(pos)
	open := fl.off(s.Body.Pos())
	semis := fl.groupHeaderSemis(keyword, open)
	postRecv := s.Post != nil && fl.groupNodesHaveRecv(s.Post)
	moveCond := s.Cond != nil && (postRecv || fl.groupNodesHaveRecv(s.Cond))
	mark := fc.hoistSubMark()
	h := fc.newGroupHoist(top, keyword)
	h.pos = pos
	bodyPos := s.Body.Pos()
	flag := ""
	if postRecv {
		flag = fc.groupTempName("first")
	}
	fc.groupLoopInit(h, s, groupForHead{semis: semis, open: open, flag: flag})
	var topNative []syntax.Stmt
	var topText []otext
	if postRecv {
		post := trimSpan(fl.src, span{start: semis[1].end, end: open})
		scanForLiterals(fl, s.Post)
		step := h.child()
		step.pos = bodyPos
		step.stmt(s.Post, post)
		fl.clearMoved(post)
		s.Post = nil
		first := ifOf(bodyPos, nil, nameOf(bodyPos, flag), []syntax.Stmt{
			assignOf(bodyPos, []syntax.Expr{nameOf(bodyPos, flag)}, []syntax.Expr{nameOf(bodyPos, "false")}),
		})
		first.Else = blockOf(bodyPos, step.native)
		topNative = append(topNative, first)
		topText = append(topText, cat(
			gen("if "+flag+" {\n"+flag+" = false\n} else {\n"),
			joinO(step.text, "\n"),
			gen("\n}"),
		))
	}
	if moveCond {
		scanForLiterals(fl, s.Cond)
		sp := h.span(s.Cond)
		cond := h.child()
		cond.pos = bodyPos
		cmark := fc.hoistSubMark()
		value := cond.expr(s.Cond)
		fc.hoistSubEmit(fc.hoistSubTake(cmark))
		text := fl.textOf(sp)
		fl.clearMoved(sp)
		s.Cond = nil
		stop := ifOf(bodyPos, nil,
			at(&syntax.Operation{Op: syntax.Not, X: at(&syntax.ParenExpr{X: value}, bodyPos)}, bodyPos),
			[]syntax.Stmt{at(&syntax.BranchStmt{Tok: syntax.Break}, bodyPos)})
		topNative = append(append(topNative, cond.native...), stop)
		topText = append(append(topText, cond.text...), cat(gen("if !("), text, gen(") {\nbreak\n}")))
	}
	fl.addT(open+1, open+1, cat(gen("\n"), joinO(topText, "\n"), gen("\n")))
	h.before(mark, fc.hostStart(s))
	out := fc.lowerFor(s, h.native)
	s.Body.List = append(topNative, s.Body.List...)
	return out
}

type groupForHead struct {
	semis []span
	open  int
	flag  string
}

func (fc *funcLower) groupTryInit(init syntax.SimpleStmt) bool {
	_, ok := fc.initCoreOf(init)
	return ok
}

func (fc *funcLower) groupLoopInit(h *groupHoist, s *syntax.ForStmt, head groupForHead) {
	fl := fc.fl
	pos := s.Pos()
	tryInit := fc.groupTryInit(s.Init)
	def, isDef := s.Init.(*syntax.AssignStmt)
	kept := isDef && def.Op == syntax.Def && !tryInit
	switch {
	case kept:
		h.assign(def)
	case s.Init != nil && (head.flag != "" || tryInit || fl.groupNodesHaveRecv(s.Init)):
		moved, _ := h.moveInit(s.Init, head.open)
		fl.clearMoved(moved)
		s.Init = nil
	}
	if head.flag == "" {
		return
	}
	semi := head.semis[0].start
	if !kept {
		s.Init = defineOf(pos, []syntax.Expr{nameOf(pos, head.flag)}, []syntax.Expr{nameOf(pos, "true")})
		fl.add(semi, semi, head.flag+" := true")
		return
	}
	def.Lhs = listOf(pos, append(syntax.UnpackListExpr(def.Lhs), nameOf(pos, head.flag)))
	def.Rhs = listOf(pos, append(syntax.UnpackListExpr(def.Rhs), nameOf(pos, "true")))
	op := fl.off(def.Pos())
	fl.add(op, op, ", "+head.flag+" ")
	fl.add(semi, semi, ", true")
}

func (fc *funcLower) groupHoistRange(s *syntax.ForStmt, rc *syntax.RangeClause, top groupFrame) []syntax.Stmt {
	start := fc.hostStart(s)
	mark := fc.hoistSubMark()
	h := fc.newGroupHoist(top, start)
	h.pos = s.Pos()
	rc.X = h.expr(rc.X)
	h.before(mark, start)
	if fc.groupRangeOverChan(rc) {
		return fc.lowerGroupRange(s, rc, top, h.native)
	}
	return fc.lowerFor(s, h.native)
}

func (fc *funcLower) groupHoistCall(s *syntax.CallStmt, top groupFrame) ([]syntax.Stmt, bool) {
	t, call, isTry := tryCallOf(s)
	if !isTry {
		var ok bool
		call, ok = syntax.Unparen(s.Call).(*syntax.CallExpr)
		if !ok {
			return nil, false
		}
	}
	if !fc.fl.groupNodesHaveRecv(call) {
		return nil, false
	}
	if s.Tok == syntax.Defer && !isTry {
		scanForLiterals(fc.fl, s)
	}
	start := fc.groupStmtStart(s)
	mark := fc.hoistSubMark()
	h := fc.newGroupHoist(top, start)
	h.pos = s.Pos()
	h.slots(groupOperandSlots(call))
	h.before(mark, start)
	switch {
	case s.Tok == syntax.Go:
		return append(h.native, fc.lowerGroupGo(s)...), true
	case isTry:
		return append(h.native, fc.lowerDeferTry(s, t, call)...), true
	}
	groupFlattenArgs(call)
	return append(h.native, s), true
}

func groupDeferredCall(deferred syntax.Expr) (*syntax.CallExpr, bool) {
	e := syntax.Unparen(deferred)
	if t, ok := e.(*syntax.TryExpr); ok {
		e = syntax.Unparen(t.X)
	}
	call, ok := e.(*syntax.CallExpr)
	return call, ok
}

func (fc *funcLower) groupHoistDeferredArgs(s syntax.Stmt, deferred syntax.Expr, top groupFrame) ([]syntax.Stmt, bool) {
	call, ok := groupDeferredCall(deferred)
	if !ok || !fc.fl.groupNodesHaveRecv(call) {
		return nil, false
	}
	start := fc.fl.off(s.Pos())
	mark := fc.hoistSubMark()
	h := fc.newGroupHoist(top, start)
	h.pos = s.Pos()
	h.slots(groupOperandSlots(call))
	h.before(mark, start)
	return h.native, true
}

func (fc *funcLower) groupScopeArgs(s *syntax.ScopeDeferStmt, call *syntax.CallExpr) *groupHoist {
	top, ok := fc.groupChanTop()
	if !ok || !fc.fl.groupNodesHaveRecv(call) {
		return nil
	}
	start := fc.fl.off(s.Pos())
	mark := fc.hoistSubMark()
	h := fc.newGroupHoist(top, start)
	h.pos = s.Pos()
	h.slots(groupOperandSlots(call))
	fc.hoistSubEmit(fc.hoistSubTake(mark))
	return h
}
