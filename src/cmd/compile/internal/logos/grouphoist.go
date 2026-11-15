// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"go/token"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

type hoistSub struct {
	start int
	end   int
	text  string
}

func (fc *funcLower) hoistSubMark() int {
	return len(fc.hoistSubs)
}

func (fc *funcLower) hoistSubTake(mark int) []hoistSub {
	subs := fc.hoistSubs[mark:]
	fc.hoistSubs = fc.hoistSubs[:mark]
	return subs
}

func (fc *funcLower) hoistSubEmit(subs []hoistSub) {
	for _, sub := range subs {
		fc.fl.add(sub.start, sub.end, sub.text)
	}
}

func (fc *funcLower) groupHasCall(e syntax.Expr) bool {
	found := false
	syntax.Inspect(e, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.FuncLit:
			return false
		case *syntax.CallExpr:
			if tv, ok := fc.fl.pkg.info.Types[n.Fun]; !ok || !tv.IsType() {
				found = true
			}
		}
		return !found
	})
	return found
}

func (fc *funcLower) groupReject(pos syntax.Pos, where string) {
	host := "a go group child"
	if top, ok := fc.groupTop(); ok && top.segment {
		host = "a group stage lambda"
	}
	base.Errorf("%s: logos: channel receive %s cannot lower in %s", pos.String(), where, host)
}

type groupHoist struct {
	fc      *funcLower
	top     groupFrame
	from    int
	pos     syntax.Pos
	native  []syntax.Stmt
	text    []otext
	prelude *groupClosurePrelude
}

func (fc *funcLower) newGroupHoist(top groupFrame, from int) *groupHoist {
	return &groupHoist{fc: fc, top: top, from: from}
}

func (h *groupHoist) add(native []syntax.Stmt, text ...otext) {
	h.native = append(h.native, native...)
	h.text = append(h.text, text...)
}

func (h *groupHoist) render() otext {
	return cat(joinO(h.text, "\n"), gen("\n"))
}

func (h *groupHoist) before(mark, start int) bool {
	if len(h.native) == 0 {
		h.fc.hoistSubTake(mark)
		return false
	}
	h.fc.hoistSubEmit(h.fc.hoistSubTake(mark))
	if h.prelude != nil {
		h.prelude.text = cat(h.prelude.text, h.render())
	} else {
		h.fc.fl.addT(start, start, h.render())
	}
	return true
}

func (h *groupHoist) span(e syntax.Expr) span {
	fl := h.fc.fl
	return fl.groupBalanced(h.from, fl.exprStart(e), fl.groupOperandEnd(e))
}

func (h *groupHoist) slots(slots []*syntax.Expr) {
	last := -1
	for i, s := range slots {
		if h.fc.fl.groupNodesHaveRecv(*s) {
			last = i
		}
	}
	h.slotsThrough(slots, last)
}

func (h *groupHoist) slotsBeforeRecv(slots []*syntax.Expr) {
	h.slotsThrough(slots, len(slots))
}

func (h *groupHoist) slotsThrough(slots []*syntax.Expr, last int) {
	fc := h.fc
	for i, s := range slots {
		if i > last {
			return
		}
		mark := fc.hoistSubMark()
		if fc.fl.groupNodesHaveRecv(*s) {
			*s = h.expr(*s)
		}
		if i < last && fc.groupHasCall(*s) {
			if lit, ok := (*s).(*syntax.CompositeLit); ok && lit.Type == nil {
				inner := groupOperandSlots(lit)
				h.slotsThrough(inner, len(inner))
				continue
			}
			fc.hoistSubEmit(fc.hoistSubTake(mark))
			*s = h.plain(*s)
		}
	}
}

func (h *groupHoist) expr(e syntax.Expr) syntax.Expr {
	if closure := h.fc.fl.groupClosureCall(e); closure != nil {
		return h.closure(closure, groupClosureValue)
	}
	if op, ok := groupRecvOp(e); ok {
		return nameOf(syntax.StartPos(op), h.recv(op, 1)[0])
	}
	if op, ok := e.(*syntax.Operation); ok && op.Y != nil && (op.Op == syntax.AndAnd || op.Op == syntax.OrOr) && h.fc.fl.groupNodesHaveRecv(op.Y) {
		return h.shortCircuit(op)
	}
	h.slots(groupOperandSlots(e))
	if call, ok := e.(*syntax.CallExpr); ok {
		groupFlattenArgs(call)
	}
	return e
}

func (h *groupHoist) valueTemp(pos syntax.Pos, like syntax.Expr, tmp string, value syntax.Expr, valueText otext) ([]syntax.Stmt, []otext) {
	fl := h.fc.fl
	if typ := fl.pkg.info.TypeOf(like); typ != nil {
		if native, spelled, ok := fl.spellTypeBoth(types2.Default(typ), pos); ok {
			return []syntax.Stmt{
					typedVarDecl(pos, nameOf(pos, tmp), native),
					assignOf(pos, []syntax.Expr{nameOf(pos, tmp)}, []syntax.Expr{value}),
				},
				[]otext{gen("var " + tmp + " " + spelled), cat(gen(tmp+" = "), valueText)}
		}
	}
	return []syntax.Stmt{defineOf(pos, []syntax.Expr{nameOf(pos, tmp)}, []syntax.Expr{value})},
		[]otext{cat(gen(tmp+" := "), valueText)}
}

func (h *groupHoist) plain(e syntax.Expr) syntax.Expr {
	fc := h.fc
	sp := h.span(e)
	tmp := fc.groupTempName("t")
	text := gen(fc.fl.moveRegion(sp, tmp))
	native, texts := h.valueTemp(e.Pos(), e, tmp, e, text)
	h.add(native, texts...)
	return nameOf(syntax.StartPos(e), tmp)
}

func (h *groupHoist) channel(op *syntax.Operation) (syntax.Expr, otext, types2.Type) {
	fc := h.fc
	fl := fc.fl
	ch := op.X
	elemT := groupChanElem(fl.pkg.info.TypeOf(ch))
	chSpan := h.span(ch)
	nch := ch
	if fl.groupNodesHaveRecv(ch) {
		mark := fc.hoistSubMark()
		nch = h.expr(ch)
		fc.hoistSubEmit(fc.hoistSubTake(mark))
	}
	chText := fl.textOf(chSpan)
	return nch, chText, elemT
}

func (h *groupHoist) recv(op *syntax.Operation, vals int) []string {
	fc := h.fc
	whole := h.span(op)
	nch, chText, elemT := h.channel(op)
	native, text, names := fc.groupRecvSelect(h.scopePos(op.Pos()), h.top, groupChanOperand{expr: nch, text: chText, elem: elemT}, vals)
	h.add(native, text...)
	fc.hoistSubs = append(fc.hoistSubs, hoistSub{start: whole.start, end: whole.end, text: strings.Join(names, ", ")})
	return names
}

func groupOperandSlots(e syntax.Expr) []*syntax.Expr {
	switch n := e.(type) {
	case *syntax.ParenExpr:
		return []*syntax.Expr{&n.X}
	case *syntax.Operation:
		if n.Y == nil {
			return []*syntax.Expr{&n.X}
		}
		return []*syntax.Expr{&n.X, &n.Y}
	case *syntax.CallExpr:
		var out []*syntax.Expr
		if sel, ok := n.Fun.(*syntax.SelectorExpr); ok {
			out = append(out, &sel.X)
		} else {
			out = append(out, &n.Fun)
		}
		for i := range n.ArgList {
			out = append(out, &n.ArgList[i])
		}
		return out
	case *syntax.CompositeLit:
		var out []*syntax.Expr
		for i := range n.ElemList {
			if kv, ok := n.ElemList[i].(*syntax.KeyValueExpr); ok {
				out = append(out, &kv.Key, &kv.Value)
			} else {
				out = append(out, &n.ElemList[i])
			}
		}
		return out
	case *syntax.KeyValueExpr:
		return []*syntax.Expr{&n.Key, &n.Value}
	case *syntax.IndexExpr:
		return []*syntax.Expr{&n.X, &n.Index}
	case *syntax.SliceExpr:
		out := []*syntax.Expr{&n.X}
		for i := range n.Index {
			if n.Index[i] != nil {
				out = append(out, &n.Index[i])
			}
		}
		return out
	case *syntax.SelectorExpr:
		return []*syntax.Expr{&n.X}
	case *syntax.AssertExpr:
		return []*syntax.Expr{&n.X}
	case *syntax.TypeSwitchGuard:
		return []*syntax.Expr{&n.X}
	case *syntax.TryExpr:
		return []*syntax.Expr{&n.X}
	case *syntax.InterpExpr:
		out := make([]*syntax.Expr, len(n.Exprs))
		for i := range n.Exprs {
			out[i] = &n.Exprs[i]
		}
		return out
	case *syntax.TupleLit:
		out := make([]*syntax.Expr, len(n.Elems))
		for i := range n.Elems {
			out[i] = &n.Elems[i]
		}
		return out
	}
	return nil
}

func (h *groupHoist) lvalueSlots(e syntax.Expr) []*syntax.Expr {
	info := h.fc.fl.pkg.info
	switch n := e.(type) {
	case *syntax.IndexExpr:
		if t := info.TypeOf(n.X); t != nil {
			if _, array := t.Underlying().(*types2.Array); array {
				return append(h.lvalueSlots(n.X), &n.Index)
			}
		}
		return []*syntax.Expr{&n.X, &n.Index}
	case *syntax.SelectorExpr:
		if t := info.TypeOf(n.X); t != nil {
			if _, value := t.Underlying().(*types2.Struct); value {
				return h.lvalueSlots(n.X)
			}
		}
		return []*syntax.Expr{&n.X}
	case *syntax.Operation:
		if n.Y == nil && n.Op == syntax.Mul {
			return []*syntax.Expr{&n.X}
		}
	}
	return nil
}

func (h *groupHoist) assign(s *syntax.AssignStmt) {
	pos := s.Pos()
	lhs := syntax.UnpackListExpr(s.Lhs)
	rhs := syntax.UnpackListExpr(s.Rhs)
	var slots []*syntax.Expr
	if s.Op != syntax.Def {
		for _, l := range lhs {
			slots = append(slots, h.lvalueSlots(l)...)
		}
	}
	if len(lhs) == 2 && len(rhs) == 1 {
		if op, ok := groupRecvOp(rhs[0]); ok {
			h.slotsBeforeRecv(slots)
			names := h.recv(op, 2)
			s.Rhs = listOf(pos, []syntax.Expr{nameOf(pos, names[0]), nameOf(pos, names[1])})
			return
		}
	}
	for i := range rhs {
		slots = append(slots, &rhs[i])
	}
	h.slots(slots)
	if s.Rhs != nil {
		s.Rhs = listOf(pos, rhs)
	}
}

func (h *groupHoist) decl(s *syntax.DeclStmt) {
	pos := s.Pos()
	var specs []*syntax.VarDecl
	var values [][]syntax.Expr
	var slots []*syntax.Expr
	for _, d := range s.DeclList {
		spec, ok := d.(*syntax.VarDecl)
		if !ok || spec.Values == nil {
			continue
		}
		vals := syntax.UnpackListExpr(spec.Values)
		if len(spec.NameList) == 2 && len(vals) == 1 {
			if op, ok := groupRecvOp(vals[0]); ok {
				h.slotsBeforeRecv(slots)
				slots = nil
				names := h.recv(op, 2)
				spec.Values = listOf(pos, []syntax.Expr{nameOf(pos, names[0]), nameOf(pos, names[1])})
				continue
			}
		}
		specs = append(specs, spec)
		values = append(values, vals)
		for i := range vals {
			slots = append(slots, &vals[i])
		}
	}
	h.slots(slots)
	for i, spec := range specs {
		spec.Values = listOf(pos, values[i])
	}
}

func (fc *funcLower) groupStmtStart(s syntax.Stmt) int {
	fl := fc.fl
	if es, ok := s.(*syntax.ExprStmt); ok {
		return spanStartBack(fl.src, fl.exprStart(es.X))
	}
	return spanStartBack(fl.src, fl.off(syntax.StartPos(s)))
}

func (fc *funcLower) groupHoistAssign(s *syntax.AssignStmt, top groupFrame) []syntax.Stmt {
	_, _, isTry := tryAssignOf(s)
	if !isTry {
		scanForLiterals(fc.fl, s)
	}
	start := fc.groupStmtStart(s)
	mark := fc.hoistSubMark()
	h := fc.newGroupHoist(top, start)
	h.atStatement(s)
	h.assign(s)
	if !h.before(mark, start) {
		return []syntax.Stmt{s}
	}
	if t, call, ok := tryAssignOf(s); ok {
		return append(h.native, fc.lowerTryAssign(s, t, call)...)
	}
	return append(h.native, s)
}

func (fc *funcLower) groupHoistDecl(s *syntax.DeclStmt, top groupFrame) []syntax.Stmt {
	scanForLiterals(fc.fl, s)
	start := fc.groupStmtStart(s)
	mark := fc.hoistSubMark()
	h := fc.newGroupHoist(top, start)
	h.decl(s)
	if !h.before(mark, start) {
		return []syntax.Stmt{s}
	}
	return append(h.native, s)
}

func (fc *funcLower) groupHoistExprStmt(s *syntax.ExprStmt, top groupFrame) []syntax.Stmt {
	if op, ok := groupRecvOp(s.X); ok {
		return fc.groupHoistRecvStmt(s, op, top)
	}
	_, _, isTry := tryExprOf(s)
	if !isTry {
		scanForLiterals(fc.fl, s)
	}
	start := fc.groupStmtStart(s)
	mark := fc.hoistSubMark()
	h := fc.newGroupHoist(top, start)
	h.atStatement(s)
	s.X = h.statement(s.X)
	if !h.before(mark, start) {
		return []syntax.Stmt{s}
	}
	if s.X == nil {
		return h.native
	}
	if t, call, ok := tryExprOf(s); ok {
		return append(h.native, fc.lowerTryExpr(s, t, call)...)
	}
	return append(h.native, s)
}

func (fc *funcLower) groupHoistRecvStmt(s *syntax.ExprStmt, op *syntax.Operation, top groupFrame) []syntax.Stmt {
	fl := fc.fl
	scanForLiterals(fl, s)
	h := fc.newGroupHoist(top, fc.groupStmtStart(s))
	sel, text, whole := h.recvStmt(s, op)
	h.add([]syntax.Stmt{sel}, text)
	fl.addT(whole.start, whole.end, joinO(h.text, "\n"))
	return h.native
}

func (fc *funcLower) groupHoistReturn(s *syntax.ReturnStmt, top groupFrame) []syntax.Stmt {
	scanForLiterals(fc.fl, s)
	start := fc.groupStmtStart(s)
	mark := fc.hoistSubMark()
	h := fc.newGroupHoist(top, start)
	h.atStatement(s)
	results := syntax.UnpackListExpr(s.Results)
	slots := make([]*syntax.Expr, len(results))
	for i := range results {
		slots[i] = &results[i]
	}
	h.slots(slots)
	if len(h.native) != 0 {
		s.Results = listOf(s.Pos(), results)
	}
	if !h.before(mark, start) {
		return []syntax.Stmt{s}
	}
	return append(h.native, s)
}

func (fc *funcLower) groupHoistSend(s *syntax.SendStmt, top groupFrame) []syntax.Stmt {
	fl := fc.fl
	scanForLiterals(fl, s)
	h := fc.newGroupHoist(top, fc.groupStmtStart(s))
	sel, text, sp := h.send(s)
	h.add([]syntax.Stmt{sel}, text)
	fl.addT(sp.start, sp.end, joinO(h.text, "\n"))
	return h.native
}

func (fl *fileLower) groupNodesHaveRecv(nodes ...syntax.Node) bool {
	found := false
	for _, n := range nodes {
		if n == nil || found {
			continue
		}
		syntax.Inspect(n, func(n syntax.Node) bool {
			switch n := n.(type) {
			case *syntax.FuncLit:
				if closure := fl.groupClosureFor(n); closure != nil && !closure.lifted {
					found = true
				}
				return false
			case *syntax.SelectStmt:
				return false
			case syntax.Expr:
				if _, ok := groupRecvOp(n); ok {
					found = true
				}
			}
			return !found
		})
	}
	return found
}

func (fl *fileLower) groupStmtHasRecv(s syntax.Stmt) bool {
	switch s := s.(type) {
	case *syntax.IfStmt:
		return fl.groupNodesHaveRecv(s.Init, s.Cond)
	case *syntax.SwitchStmt:
		return fl.groupNodesHaveRecv(s.Init, s.Tag) || fl.groupCasesHaveRecv(s)
	case *syntax.ForStmt:
		return fl.groupNodesHaveRecv(s.Init, s.Cond, s.Post)
	}
	return fl.groupNodesHaveRecv(s)
}

func (fl *fileLower) groupCasesHaveRecv(s *syntax.SwitchStmt) bool {
	for _, c := range s.Body {
		if c.Cases != nil && fl.groupNodesHaveRecv(c.Cases) {
			return true
		}
	}
	return false
}

func (fl *fileLower) groupSelectHasRecv(s *syntax.SelectStmt) bool {
	for _, c := range s.Body {
		if c.Comm != nil && fl.groupNodesHaveRecv(c.Comm) {
			return true
		}
	}
	return false
}

func (fl *fileLower) groupTokenEnd(off int) int {
	end := off
	scanSegment(fl.src, off, len(fl.src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		end = t.end
		return false
	})
	return end
}

func (fl *fileLower) groupClosingEnd(open int) int {
	end := open
	depth := 0
	scanSegment(fl.src, open, len(fl.src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			depth--
			if depth == 0 {
				end = t.end
				return false
			}
		}
		return true
	})
	return end
}

func (fl *fileLower) groupOperandEnd(e syntax.Expr) int {
	if closure := fl.groupClosureCall(e); closure != nil {
		return closure.source.end
	}
	switch n := e.(type) {
	case *syntax.CompositeLit:
		return fl.off(n.Rbrace) + 1
	case *syntax.FuncLit:
		return fl.off(n.Body.Rbrace) + 1
	case *syntax.CallExpr:
		return fl.callBodySpan(n).end
	case *syntax.ParenExpr, *syntax.IndexExpr, *syntax.SliceExpr, *syntax.AssertExpr, *syntax.TypeSwitchGuard:
		return fl.groupClosingEnd(fl.off(n.Pos()))
	case *syntax.SelectorExpr:
		return fl.groupTokenEnd(fl.off(n.Sel.Pos()))
	case *syntax.KeyValueExpr:
		return fl.groupOperandEnd(n.Value)
	case *syntax.TupleLit:
		return fl.groupClosingEnd(fl.off(n.Pos()))
	case *syntax.InterpExpr:
		return fl.off(n.Pos()) + len(n.Raw)
	case *syntax.Operation:
		if n.Y != nil {
			return fl.groupOperandEnd(n.Y)
		}
		return fl.groupOperandEnd(n.X)
	}
	return fl.groupTokenEnd(fl.off(e.Pos()))
}

func (fl *fileLower) groupBalanced(from, start, end int) span {
	var stack []int
	base := -1
	scanSegment(fl.src, from, len(fl.src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		if base < 0 && t.start >= start {
			base = len(stack)
		}
		if base >= 0 && t.start >= end && len(stack) <= base {
			return false
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			stack = append(stack, t.start)
		case token.RPAREN, token.RBRACK, token.RBRACE:
			n := len(stack)
			if n == 0 {
				return true
			}
			open := stack[n-1]
			stack = stack[:n-1]
			switch {
			case base < 0:
			case n-1 < base:
				base = n - 1
				start = min(start, open)
			case t.start >= end:
				end = t.end
			}
		}
		return true
	})
	return span{start: start, end: end}
}
