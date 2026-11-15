// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func (fc *funcLower) groupChanTop() (groupFrame, bool) {
	if top, ok := fc.groupTop(); ok && top.child {
		return top, true
	}
	return groupFrame{}, false
}

func (fc *funcLower) groupDoneResults(pos syntax.Pos, top groupFrame) []syntax.Expr {
	var res []syntax.Expr
	if top.closure != nil {
		res = append(res, tempNodes(pos, top.closure.values)...)
	}
	if top.segmentRes != nil {
		res = append(res, fc.fl.zeroValue(top.segmentRes, pos))
	}
	return append(res, callOf(pos, selOf(pos, nameOf(pos, top.gctx), "Err"), nil))
}

func (fc *funcLower) groupDoneText(top groupFrame) string {
	var res []string
	for _, r := range fc.groupDoneResults(syntax.Pos{}, top) {
		res = append(res, renderExpr(r))
	}
	return "case <-" + top.gctx + ".Done(): return " + strings.Join(res, ", ")
}

func (fc *funcLower) groupDoneClause(pos syntax.Pos, top groupFrame) *syntax.CommClause {
	done := callOf(pos, selOf(pos, nameOf(pos, top.gctx), "Done"), nil)
	recv := at(&syntax.Operation{Op: syntax.Recv, X: done}, pos)
	ret := returnOf(pos, fc.groupDoneResults(pos, top))
	return at(&syntax.CommClause{Comm: exprStmtOf(pos, recv), Body: []syntax.Stmt{ret}}, pos)
}

func groupRecvOp(e syntax.Expr) (*syntax.Operation, bool) {
	op, ok := syntax.Unparen(e).(*syntax.Operation)
	if !ok || op.Op != syntax.Recv || op.Y != nil {
		return nil, false
	}
	return op, true
}

func groupChanElem(t types2.Type) types2.Type {
	ch, ok := t.Underlying().(*types2.Chan)
	if !ok {
		base.Fatalf("logos: receive from non-channel type %s", t)
	}
	return ch.Elem()
}

func (fc *funcLower) groupTempName(prefix string) string {
	nm := fmt.Sprintf("_logos_%s%d", prefix, fc.tmpSeq)
	fc.tmpSeq++
	return nm
}

type groupChanOperand struct {
	expr syntax.Expr
	text otext
	elem types2.Type
}

func (fc *funcLower) groupElemTemp(pos syntax.Pos, tmp string, c groupChanOperand) ([]syntax.Stmt, []otext, groupChanOperand) {
	if elemNative, elemSpelled, ok := fc.fl.spellTypeBoth(c.elem, pos); ok {
		return []syntax.Stmt{typedVarDecl(pos, nameOf(pos, tmp), elemNative)},
			[]otext{gen("var " + tmp + " " + elemSpelled)}, c
	}
	chName := fc.groupTempName("c")
	fc.fl.needRT = true
	zero := callOf(pos, selOf(pos, nameOf(pos, rtAlias), "ElemZero"), []syntax.Expr{nameOf(pos, chName)})
	native := []syntax.Stmt{
		defineOf(pos, []syntax.Expr{nameOf(pos, chName)}, []syntax.Expr{c.expr}),
		defineOf(pos, []syntax.Expr{nameOf(pos, tmp)}, []syntax.Expr{zero}),
	}
	text := []otext{
		cat(gen(chName+" := "), c.text),
		gen(tmp + " := " + rtAlias + ".ElemZero(" + chName + ")"),
	}
	return native, text, groupChanOperand{expr: nameOf(pos, chName), text: gen(chName), elem: c.elem}
}

func (fc *funcLower) groupRecvSelect(pos syntax.Pos, top groupFrame, c groupChanOperand, vals int) ([]syntax.Stmt, []otext, []string) {
	tmp := fc.groupTempName("t")
	names := []string{tmp}
	native, text, c := fc.groupElemTemp(pos, tmp, c)
	caseLHS := tmp
	if vals == 2 {
		okName := fc.groupTempName("o")
		names = append(names, okName)
		native = append(native, typedVarDecl(pos, nameOf(pos, okName), nameOf(pos, "bool")))
		text = append(text, gen("var "+okName+" bool"))
		caseLHS += ", " + okName
	}
	recvN := assignOf(pos, tempNodes(pos, names), []syntax.Expr{at(&syntax.Operation{Op: syntax.Recv, X: c.expr}, pos)})
	recvSelect := at(&syntax.SelectStmt{
		Body: []*syntax.CommClause{at(&syntax.CommClause{Comm: recvN}, pos), fc.groupDoneClause(pos, top)},
	}, pos)
	if fc.fl.groupRecvSelects == nil {
		fc.fl.groupRecvSelects = make(map[*syntax.SelectStmt]bool)
	}
	fc.fl.groupRecvSelects[recvSelect] = true
	native = append(native, recvSelect)
	text = append(text, cat(gen("select { case "+caseLHS+" = <-"), c.text, gen(": "+fc.groupDoneText(top)+" }")))
	return native, text, names
}

func (fc *funcLower) groupChanStmt(s syntax.Stmt) ([]syntax.Stmt, bool) {
	top, ok := fc.groupChanTop()
	if !ok {
		return nil, false
	}
	before := base.Errors()
	out, handled := fc.groupChanDispatch(s, top)
	if handled && base.Errors() == before {
		last := out[len(out)-1]
		for {
			if lab, ok := last.(*syntax.LabeledStmt); ok {
				last = lab.Stmt
			} else if b, ok := last.(*syntax.BlockStmt); ok && len(b.List) > 0 {
				last = b.List[len(b.List)-1]
			} else {
				break
			}
		}
		if fc.fl.groupStmtHasRecv(last) {
			fc.groupReject(last.Pos(), "in an unsupported position")
		}
	}
	return out, handled
}

func (fc *funcLower) groupChanDispatch(s syntax.Stmt, top groupFrame) ([]syntax.Stmt, bool) {
	switch s := s.(type) {
	case *syntax.AssignStmt:
		if fc.fl.groupStmtHasRecv(s) {
			return fc.groupHoistAssign(s, top), true
		}
	case *syntax.DeclStmt:
		if fc.fl.groupStmtHasRecv(s) {
			return fc.groupHoistDecl(s, top), true
		}
	case *syntax.ExprStmt:
		if fc.fl.groupStmtHasRecv(s) {
			return fc.groupHoistExprStmt(s, top), true
		}
	case *syntax.ReturnStmt:
		if fc.fl.groupStmtHasRecv(s) {
			return fc.groupHoistReturn(s, top), true
		}
	case *syntax.SendStmt:
		return fc.groupHoistSend(s, top), true
	case *syntax.ForStmt:
		if fc.fl.groupStmtHasRecv(s) {
			return fc.groupHoistFor(s, top), true
		}
		return fc.groupChanFor(s, top)
	case *syntax.CallStmt:
		return fc.groupHoistCall(s, top)
	case *syntax.ErrdeferStmt:
		pre, ok := fc.groupHoistDeferredArgs(s, s.Call, top)
		if !ok {
			return nil, false
		}
		return append(pre, fc.lowerErrdefer(s)...), true
	case *syntax.IfStmt:
		if fc.fl.groupStmtHasRecv(s) {
			return fc.groupHoistIf(s, top), true
		}
	case *syntax.SwitchStmt:
		if fc.fl.groupStmtHasRecv(s) {
			return fc.groupHoistSwitch(s, top), true
		}
	case *syntax.SelectStmt:
		if fc.fl.groupSelectHasRecv(s) {
			return fc.groupHoistSelect(s, top), true
		}
	}
	return nil, false
}

func (fc *funcLower) groupHoistSelect(s *syntax.SelectStmt, top groupFrame) []syntax.Stmt {
	fl := fc.fl
	start := fc.groupStmtStart(s)
	mark := fc.hoistSubMark()
	h := fc.newGroupHoist(top, start)
	h.atStatement(s)
	for _, clause := range s.Body {
		scanForLiterals(fl, clause.Comm)
	}
	h.selectEntries(s)
	h.before(mark, start)
	for _, clause := range s.Body {
		clause.Body = fc.lowerList(clause.Body)
	}
	fc.groupAugmentSelect(s)
	return append(h.native, s)
}

type groupSelectEntry struct {
	clause *syntax.CommClause
	comm   *syntax.AssignStmt
	op     *syntax.Operation
	lhs    []syntax.Expr
	chSpan span
	vals   int
	elem   types2.Type
}

func (h *groupHoist) selectEntries(s *syntax.SelectStmt) {
	var slots []*syntax.Expr
	var entries []groupSelectEntry
	for _, clause := range s.Body {
		comm := clause.Comm
		if comm == nil {
			continue
		}
		switch comm := comm.(type) {
		case *syntax.SendStmt:
			slots = append(slots, &comm.Chan, &comm.Value)
		case *syntax.ExprStmt:
			if op, ok := groupRecvOp(comm.X); ok {
				slots = append(slots, &op.X)
			}
		case *syntax.AssignStmt:
			op, ok := h.recvComm(comm)
			if !ok {
				break
			}
			lhs := syntax.UnpackListExpr(comm.Lhs)
			if comm.Op != syntax.Def && h.fc.fl.groupNodesHaveRecv(comm.Lhs) {
				entries = append(entries, groupSelectEntry{clause: clause, comm: comm, op: op, lhs: lhs, chSpan: h.span(op.X), vals: len(lhs), elem: groupChanElem(h.fc.fl.pkg.info.TypeOf(op.X))})
			}
			slots = append(slots, &op.X)
		}
	}
	h.slots(slots)
	h.unwrapSendValues(s)
	h.splitRecvComms(entries)
}

func (h *groupHoist) unwrapSendValues(s *syntax.SelectStmt) {
	for _, clause := range s.Body {
		send, ok := clause.Comm.(*syntax.SendStmt)
		if !ok {
			continue
		}
		if list, ok := send.Value.(*syntax.ListExpr); ok && len(list.ElemList) == 1 {
			send.Value = list.ElemList[0]
		}
	}
}

func (h *groupHoist) recvComm(comm *syntax.AssignStmt) (*syntax.Operation, bool) {
	rhs := syntax.UnpackListExpr(comm.Rhs)
	if len(rhs) != 1 {
		return nil, false
	}
	return groupRecvOp(rhs[0])
}

func (h *groupHoist) splitRecvComms(entries []groupSelectEntry) {
	for _, e := range entries {
		fl := h.fc.fl
		lhsSpan := span{start: h.span(e.lhs[0]).start, end: h.span(e.lhs[len(e.lhs)-1]).end}
		pos := e.comm.Pos()
		chText := h.fc.fl.textOf(e.chSpan)
		elemT := e.elem
		tmp := h.fc.groupTempName("r")
		native, texts, c := h.fc.groupElemTemp(pos, tmp, groupChanOperand{expr: e.op.X, text: chText, elem: elemT})
		h.add(native, texts...)
		e.op.X = c.expr
		fl.addT(e.chSpan.start, e.chSpan.end, c.text)
		names := []syntax.Expr{nameOf(pos, tmp)}
		if e.vals == 2 {
			okName := h.fc.groupTempName("o")
			h.add([]syntax.Stmt{typedVarDecl(pos, nameOf(pos, okName), nameOf(pos, "bool"))}, gen("var "+okName+" bool"))
			names = append(names, nameOf(pos, okName))
		}
		e.comm.Lhs = listOf(pos, names)
		pro := assignOf(pos, e.lhs, names)
		target := h.fc.newGroupHoist(h.top, lhsSpan.start)
		mark := h.fc.hoistSubMark()
		target.assign(pro)
		h.fc.hoistSubEmit(h.fc.hoistSubTake(mark))
		var tempNames []string
		for _, name := range names {
			tempNames = append(tempNames, name.(*syntax.Name).Value)
		}
		list := strings.Join(tempNames, ", ")
		lhsText := fl.moveRegion(lhsSpan, list)
		bodyStart := fl.off(e.clause.Colon) + 1
		fl.addT(bodyStart, bodyStart, cat(gen("\n"), target.render(), gen(lhsText+" = "+list+";\n")))
		prefix := append(target.native, pro)
		e.clause.Body = append(prefix, e.clause.Body...)
	}
}

func (fc *funcLower) groupChanFor(s *syntax.ForStmt, top groupFrame) ([]syntax.Stmt, bool) {
	rc, ok := s.Init.(*syntax.RangeClause)
	if !ok || !fc.groupRangeOverChan(rc) {
		return nil, false
	}
	return fc.lowerGroupRange(s, rc, top, nil), true
}

func (fc *funcLower) groupRangeOverChan(rc *syntax.RangeClause) bool {
	t := fc.fl.pkg.info.TypeOf(syntax.Unparen(rc.X))
	if t == nil {
		return false
	}
	_, ok := t.Underlying().(*types2.Chan)
	return ok
}

func (fc *funcLower) lowerGroupRange(s *syntax.ForStmt, rc *syntax.RangeClause, top groupFrame, prologue []syntax.Stmt) []syntax.Stmt {
	fl := fc.fl
	pos := s.Pos()
	out := fc.scoped(s, prologue)
	scanForLiterals(fl, rc.X)
	chStart := fl.off(leftmostOf(rc.X))
	chText := fl.textOf(trimSpan(fl.src, span{start: chStart, end: fl.off(s.Body.Pos())}))
	fl.needRT = true
	adapter := callOf(pos, selOf(pos, nameOf(pos, rtAlias), "Chan"), []syntax.Expr{
		nameOf(pos, top.gctx), rc.X,
	})
	rc.X = adapter
	scanForLiterals(fl, s.Cond)
	scanForLiterals(fl, s.Post)
	s.Body.List = fc.lowerList(s.Body.List)
	var lhsPart otext
	if rc.Lhs != nil {
		lhsPart = fl.srcUser(fl.off(leftmostOf(rc.Lhs)), fl.off(rc.Pos()))
	}
	headStart := fl.off(pos)
	braceOff := fl.off(s.Body.Pos())
	fl.addT(headStart, braceOff, cat(gen("for "), lhsPart, gen("range "+rtAlias+".Chan("+top.gctx+", "), chText, gen(") ")))
	checkPos := s.Body.Rbrace
	checkN := ifOf(checkPos, nil,
		at(&syntax.Operation{
			Op: syntax.Neq,
			X:  callOf(checkPos, selOf(checkPos, nameOf(checkPos, top.gctx), "Err"), nil),
			Y:  nilOf(checkPos),
		}, checkPos),
		[]syntax.Stmt{
			returnOf(checkPos, fc.groupDoneResults(checkPos, top)),
		})
	end := fl.off(s.Body.Rbrace) + 1
	fl.addT(end, end, cat(gen("\n"), renderText([]syntax.Stmt{checkN}, nil)))
	return append(out, checkN)
}

func groupSelectHasDefault(s *syntax.SelectStmt) bool {
	for _, c := range s.Body {
		if c.Comm == nil {
			return true
		}
	}
	return false
}

func (fc *funcLower) groupAugmentSelect(s *syntax.SelectStmt) {
	if fc.fl.groupRecvSelects[s] {
		return
	}
	top, ok := fc.groupChanTop()
	if !ok || groupSelectHasDefault(s) {
		return
	}
	s.Body = append(s.Body, fc.groupDoneClause(s.Rbrace, top))
	rbraceOff := fc.fl.off(s.Rbrace)
	fc.fl.add(rbraceOff, rbraceOff, "\n"+fc.groupDoneText(top)+"\n")
}
