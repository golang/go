// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"slices"
	"strconv"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

type scopeItem struct {
	plan    *callPlan
	armed   string
	pos     syntax.Pos
	hoisted bool
	join    *groupFrame
}

type scopeDeferCtx struct {
	owner    syntax.Node
	startOff int
	endOff   int
	items    []scopeItem
	depth    int
	join     *groupFrame
}

type scopeFrame struct {
	label string
	loop  bool
}

func ownScopeItems(ctx *scopeDeferCtx) []scopeItem {
	items := ctx.items
	out := make([]scopeItem, 0, len(items)+1)
	if ctx.join != nil {
		out = append(out, scopeItem{join: ctx.join})
	}
	for i := len(items) - 1; i >= 0; i-- {
		out = append(out, items[i])
	}
	return out
}

func (ctx *scopeDeferCtx) guardBody(pos syntax.Pos, armed string, call syntax.Stmt) []syntax.Stmt {
	body := []syntax.Stmt{assignOf(pos, []syntax.Expr{nameOf(pos, armed)}, []syntax.Expr{nameOf(pos, "false")})}
	if ctx.join != nil {
		body = append(body, groupWaitOnce(pos, ctx.join.g, ctx.join.j))
	}
	return append(body, call)
}

func (fc *funcLower) nextScope() string {
	spelling := "_logos_sd" + strconv.Itoa(fc.scopeSeq)
	fc.scopeSeq++
	return spelling
}

func (fc *funcLower) lowerScopeDefers() {
	if !fc.hasScopeDefer {
		return
	}
	fc.collectScopeInfo()
	fl := fc.fl
	fc.body.List = fc.scopeBlockList(fc.body.List, fc.body, fl.off(fc.body.Rbrace), nil, nil)
	if len(fc.scopeHoists) != 0 {
		base.Fatalf("logos: scope defer guard left inside loop")
	}
}

func (fc *funcLower) collectScopeInfo() {
	fc.scopeParents = make(map[syntax.Node]syntax.Node)
	fc.scopeLabels = make(map[string]*syntax.LabeledStmt)
	var rec func(n syntax.Node, parent syntax.Node)
	rec = func(n, parent syntax.Node) {
		if n == nil {
			return
		}
		if _, ok := n.(*syntax.FuncLit); ok {
			return
		}
		fc.scopeParents[n] = parent
		if l, ok := n.(*syntax.LabeledStmt); ok {
			fc.scopeLabels[l.Label.Value] = l
		}
		switch n := n.(type) {
		case *syntax.BlockStmt:
			for _, s := range n.List {
				rec(s, n)
			}
		case *syntax.IfStmt:
			rec(n.Then, n)
			if n.Else != nil {
				rec(n.Else, n)
			}
		case *syntax.ForStmt:
			if n.Init != nil {
				rec(n.Init, n)
			}
			rec(n.Body, n)
		case *syntax.SwitchStmt:
			for _, c := range n.Body {
				rec(c, n)
			}
		case *syntax.SelectStmt:
			for _, c := range n.Body {
				rec(c, n)
			}
		case *syntax.CaseClause:
			for _, s := range n.Body {
				rec(s, n)
			}
		case *syntax.CommClause:
			if n.Comm != nil {
				rec(n.Comm, n)
			}
			for _, s := range n.Body {
				rec(s, n)
			}
		case *syntax.LabeledStmt:
			rec(n.Stmt, n)
		}
	}
	rec(fc.body, nil)
}

func (fc *funcLower) scopeContains(block, target syntax.Node) bool {
	for n := target; n != nil; n = fc.scopeParents[n] {
		if n == block {
			return true
		}
	}
	return false
}

func (fc *funcLower) scopeBlockList(list []syntax.Stmt, owner syntax.Node, endOff int, active []*scopeDeferCtx, stack []scopeFrame) []syntax.Stmt {
	out := make([]syntax.Stmt, 0, len(list))
	cur := active
	ctx := fc.groupCtxs[owner]
	if ctx != nil {
		cur = append(slices.Clone(active), ctx)
	}
	for _, s := range list {
		target, labels := unwrapScopeLabels(s)
		if sd, ok := target.(*syntax.ScopeDeferStmt); ok {
			if ctx == nil {
				ctx = &scopeDeferCtx{owner: owner, startOff: fc.fl.off(sd.Pos()), endOff: endOff, depth: len(stack)}
				fc.scopeCtxs = append(fc.scopeCtxs, ctx)
				next := make([]*scopeDeferCtx, len(cur)+1)
				copy(next, cur)
				next[len(cur)] = ctx
				cur = next
			}
			if len(labels) > 0 {
				if fc.scopeDeferLabels == nil {
					fc.scopeDeferLabels = make(map[string]int)
				}
				for _, lab := range labels {
					fc.scopeDeferLabels[lab.Value] = len(stack)
				}
			}
			if scopeInLoop(stack) {
				out = append(out, fc.scopeLoopPrelude(sd, labels, ctx)...)
				continue
			}
			out = append(out, fc.scopeLabeledPrelude(sd, labels, ctx)...)
			continue
		}
		if br, ok := s.(*syntax.BranchStmt); ok && br.Tok == syntax.Fallthrough && br.Label == nil {
			out = append(out, fc.scopeFallthrough(br, ctx)...)
			continue
		}
		out = append(out, fc.scopeStmt(s, cur, stack)...)
	}
	if ctx != nil && !scopeEndsExiting(out) {
		native, text, tw := fc.scopeStanza(syntax.EndPos(list[len(list)-1]), ownScopeItems(ctx))
		out = append(out, native...)
		fl := fc.fl
		gapStart := max(stmtEnd(fl.src, fl.off(syntax.StartPos(list[len(list)-1]))), scopeStmtEnd(fl, list[len(list)-1]))
		prefix := leftoverComments(fl, span{start: gapStart, end: endOff}, nil)
		gap := cat(prefix, renderText(text, tw.subs))
		if prefix.s == "" {
			gap = cat(gen("\n"), gap)
		}
		fl.addT(gapStart, endOff, gap)
	}
	return out
}

func scopeStmtEnd(fl *fileLower, s syntax.Stmt) int {
	switch s.(type) {
	case *syntax.ForStmt, *syntax.SwitchStmt, *syntax.SelectStmt, *syntax.IfStmt, *syntax.BlockStmt, *syntax.LabeledStmt, *syntax.GroupStmt, *syntax.GroupChild:
		return fl.off(syntax.EndPos(s)) + 1
	}
	return fl.off(syntax.EndPos(s))
}

func (fc *funcLower) scopeFallthrough(br *syntax.BranchStmt, ctx *scopeDeferCtx) []syntax.Stmt {
	if ctx == nil {
		return []syntax.Stmt{br}
	}
	native, text, tw := fc.scopeStanza(br.Pos(), ownScopeItems(ctx))
	fc.fl.addT(fc.fl.off(br.Pos()), fc.fl.off(br.Pos()), renderText(text, tw.subs))
	return append(native, br)
}

func scopeEndsExiting(list []syntax.Stmt) bool {
	if len(list) == 0 {
		return false
	}
	switch list[len(list)-1].(type) {
	case *syntax.ReturnStmt, *syntax.BranchStmt:
		return true
	}
	return false
}

func unwrapScopeLabels(s syntax.Stmt) (syntax.Stmt, []*syntax.Name) {
	var labels []*syntax.Name
	for {
		lab, ok := s.(*syntax.LabeledStmt)
		if !ok {
			return s, labels
		}
		labels = append(labels, lab.Label)
		s = lab.Stmt
	}
}
func (fc *funcLower) scopeLabeledPrelude(sd *syntax.ScopeDeferStmt, labels []*syntax.Name, ctx *scopeDeferCtx) []syntax.Stmt {
	stmtOff, prefix := fc.fl.off(sd.Pos()), gen("")
	if len(labels) > 0 {
		stmtOff = fc.fl.off(labels[0].Pos())
		prefix = fc.fl.srcUser(stmtOff, fc.fl.off(sd.Pos()))
	}
	native := fc.scopePrelude(sd, ctx, stmtOff, prefix)
	for i := len(labels) - 1; i >= 0; i-- {
		inner := native[0]
		lab := &syntax.LabeledStmt{Label: labels[i], Stmt: inner}
		lab.SetPos(labels[i].Pos())
		native[0] = lab
	}
	return native
}

func (fc *funcLower) scopePrelude(sd *syntax.ScopeDeferStmt, ctx *scopeDeferCtx, stmtOff int, prefix otext) []syntax.Stmt {
	fl := fc.fl
	pos := sd.Pos()
	call, ok := syntax.Unparen(sd.Call).(*syntax.CallExpr)
	if !ok {
		return []syntax.Stmt{sd}
	}
	recv := fc.groupScopeArgs(sd, call)
	plan, planned := fc.planScopeCall(call)
	if !planned {
		return []syntax.Stmt{sd}
	}
	item := scopeItem{plan: plan, armed: fc.nextScope(), pos: pos}
	ctx.items = append(ctx.items, item)
	tw := &twin{}
	funN, argN, capNamesN, capValsN := fc.planUses(plan, nil)
	funT, argT, capNamesT, capValsT := fc.planUses(plan, tw)
	callN := callOf(pos, funN, argN)
	callN.HasDots = plan.dots
	callT := callOf(pos, funT, argT)
	callT.HasDots = plan.dots
	armedN := nameOf(pos, item.armed)
	armedT := nameOf(pos, item.armed)
	guardN := ifOf(pos, nil, armedN, ctx.guardBody(pos, item.armed, exprStmtOf(pos, callN)))
	guardT := ifOf(pos, nil, armedT, ctx.guardBody(pos, item.armed, exprStmtOf(pos, callT)))
	var native []syntax.Stmt
	var text []syntax.Stmt
	if recv != nil {
		native = append(native, recv.native...)
		prefix = cat(prefix, recv.render())
	}
	if len(capNamesN) > 0 {
		native = append(native, fc.bindCaptures(pos, plan, capNamesN, capValsN))
		text = append(text, fc.bindCaptures(pos, plan, capNamesT, capValsT))
	}
	fc.hoistTemp(item.armed, nameOf(pos, "bool"), "bool")
	native = append(native,
		assignOf(pos, []syntax.Expr{nameOf(pos, item.armed)}, []syntax.Expr{nameOf(pos, "true")}),
		guardDefer(pos, capNamesN, guardN),
	)
	text = append(text,
		assignOf(pos, []syntax.Expr{nameOf(pos, item.armed)}, []syntax.Expr{nameOf(pos, "true")}),
		guardDefer(pos, capNamesT, guardT),
	)
	spanEnd := skipCloseParens(fl.src, plan.end)
	kept := leftoverComments(fl, span{start: stmtOff, end: spanEnd}, planSpans(plan))
	fl.addT(stmtOff, spanEnd, cat(prefix, kept, renderText(text, tw.subs)))
	return native
}

func guardDefer(pos syntax.Pos, caps []syntax.Expr, guard syntax.Stmt) syntax.Stmt {
	deferStmt := deferOf(pos, callOf(pos, deferClosure(pos, []syntax.Stmt{guard}), nil))
	if len(caps) == 0 {
		return deferStmt
	}
	return blockOf(pos, []syntax.Stmt{defineOf(pos, nameExprs(pos, caps), nameExprs(pos, caps)), deferStmt})
}

func nameExprs(pos syntax.Pos, names []syntax.Expr) []syntax.Expr {
	out := make([]syntax.Expr, len(names))
	for i, name := range names {
		out[i] = nameOf(pos, name.(*syntax.Name).Value)
	}
	return out
}

func (fc *funcLower) scopeLoopPrelude(sd *syntax.ScopeDeferStmt, labels []*syntax.Name, ctx *scopeDeferCtx) []syntax.Stmt {
	stmtOff, prefix := fc.fl.off(sd.Pos()), gen("")
	if len(labels) > 0 {
		stmtOff = fc.fl.off(labels[0].Pos())
		prefix = fc.fl.srcUser(stmtOff, fc.fl.off(sd.Pos()))
	}
	native := fc.scopeHoistedPrelude(sd, ctx, stmtOff, prefix)
	for i := len(labels) - 1; i >= 0; i-- {
		inner := native[0]
		lab := &syntax.LabeledStmt{Label: labels[i], Stmt: inner}
		lab.SetPos(labels[i].Pos())
		native[0] = lab
	}
	return native
}

func (fc *funcLower) scopeHoistedPrelude(sd *syntax.ScopeDeferStmt, ctx *scopeDeferCtx, stmtOff int, prefix otext) []syntax.Stmt {
	fl := fc.fl
	pos := sd.Pos()
	call, ok := syntax.Unparen(sd.Call).(*syntax.CallExpr)
	if !ok {
		return []syntax.Stmt{sd}
	}
	recv := fc.groupScopeArgs(sd, call)
	plan, planned := fc.planScopeCall(call)
	if !planned {
		return []syntax.Stmt{sd}
	}
	item := scopeItem{plan: plan, armed: fc.nextScope(), pos: pos, hoisted: true}
	ctx.items = append(ctx.items, item)
	fc.scopeHoists = append(fc.scopeHoists, item.armed)
	tw := &twin{}
	funN, argN, capNamesN, capValsN := fc.planUses(plan, nil)
	funT, argT, capNamesT, capValsT := fc.planUses(plan, tw)
	callN := callOf(pos, funN, argN)
	callN.HasDots = plan.dots
	callT := callOf(pos, funT, argT)
	callT.HasDots = plan.dots
	var native []syntax.Stmt
	var text []syntax.Stmt
	if recv != nil {
		native = append(native, recv.native...)
		prefix = cat(prefix, recv.render())
	}
	if len(capNamesN) > 0 {
		native = append(native, fc.bindCaptures(pos, plan, capNamesN, capValsN))
		text = append(text, fc.bindCaptures(pos, plan, capNamesT, capValsT))
	}
	native = append(native, assignOf(pos, []syntax.Expr{nameOf(pos, item.armed)}, []syntax.Expr{deferClosure(pos, []syntax.Stmt{exprStmtOf(pos, callN)})}))
	text = append(text, assignOf(pos, []syntax.Expr{nameOf(pos, item.armed)}, []syntax.Expr{deferClosure(pos, []syntax.Stmt{exprStmtOf(pos, callT)})}))
	spanEnd := skipCloseParens(fl.src, plan.end)
	kept := leftoverComments(fl, span{start: stmtOff, end: spanEnd}, planSpans(plan))
	fl.addT(stmtOff, spanEnd, cat(prefix, kept, renderText(text, tw.subs)))
	return native
}

type hoistSlot struct {
	name string
	typ  types2.Type
}

func (fc *funcLower) bindCaptures(pos syntax.Pos, plan *callPlan, names, vals []syntax.Expr) syntax.Stmt {
	if fc.hoistPlan(pos, plan) {
		return assignOf(pos, names, vals)
	}
	return defineOf(pos, names, vals)
}

func (fc *funcLower) hoistPlan(pos syntax.Pos, plan *callPlan) bool {
	var vars []hoistedVar
	for _, slot := range fc.planSlots(plan) {
		if slot.typ == nil || !hoistableType(slot.typ, fc.fl.pkg.tpkg) {
			return false
		}
		native, eject, ok := fc.fl.spellTypeBoth(slot.typ, pos)
		if !ok {
			return false
		}
		vars = append(vars, hoistedVar{name: slot.name, native: native, eject: eject})
	}
	for _, v := range vars {
		fc.hoistTemp(v.name, v.native, v.eject)
	}
	return true
}

func (fc *funcLower) planSlots(plan *callPlan) []hoistSlot {
	pieces := append([]capturePiece{plan.fun}, plan.args...)
	if plan.isMethod {
		pieces[0] = plan.recv
	}
	var slots []hoistSlot
	for _, piece := range pieces {
		slots = append(slots, fc.pieceSlots(piece)...)
	}
	return slots
}

func (fc *funcLower) pieceSlots(piece capturePiece) []hoistSlot {
	if !piece.capture {
		return nil
	}
	typ := fc.fl.pkg.info.TypeOf(piece.expr)
	if len(piece.multi) > 0 {
		tup, _ := typ.(*types2.Tuple)
		slots := make([]hoistSlot, len(piece.multi))
		for i, name := range piece.multi {
			slots[i] = hoistSlot{name: name, typ: tup.At(i).Type()}
		}
		return slots
	}
	return []hoistSlot{{name: piece.single, typ: capturedType(piece, typ)}}
}

func capturedType(piece capturePiece, typ types2.Type) types2.Type {
	if typ == nil {
		return nil
	}
	if piece.takeAddr {
		return types2.NewPointer(typ)
	}
	if ptr, ok := types2.Unalias(typ).(*types2.Pointer); ok && piece.takeDeref {
		return ptr.Elem()
	}
	return typ
}

func scopeInLoop(stack []scopeFrame) bool {
	for _, fr := range stack {
		if fr.loop {
			return true
		}
	}
	return false
}

func scopeGuardDefer(pos syntax.Pos, armed, temp string) *syntax.CallStmt {
	body := []syntax.Stmt{
		defineOf(pos, []syntax.Expr{nameOf(pos, temp)}, []syntax.Expr{nameOf(pos, armed)}),
		assignOf(pos, []syntax.Expr{nameOf(pos, armed)}, []syntax.Expr{nilOf(pos)}),
		exprStmtOf(pos, callOf(pos, nameOf(pos, temp), nil)),
	}
	return deferOf(pos, callOf(pos, deferClosure(pos, []syntax.Stmt{ifOf(pos, nil, notNil(pos, nameOf(pos, armed)), body)}), nil))
}

func scopeHolderStanza(pos syntax.Pos, armed, temp string) *syntax.IfStmt {
	body := []syntax.Stmt{
		defineOf(pos, []syntax.Expr{nameOf(pos, temp)}, []syntax.Expr{nameOf(pos, armed)}),
		assignOf(pos, []syntax.Expr{nameOf(pos, armed)}, []syntax.Expr{nilOf(pos)}),
		exprStmtOf(pos, callOf(pos, nameOf(pos, temp), nil)),
	}
	return ifOf(pos, nil, notNil(pos, nameOf(pos, armed)), body)
}

func (fc *funcLower) drainScopeHoists(pos syntax.Pos) []syntax.Stmt {
	if len(fc.scopeHoists) == 0 {
		return nil
	}
	var native []syntax.Stmt
	var text []syntax.Stmt
	for _, armed := range fc.scopeHoists {
		temp := fc.nextScope()
		fc.hoistTemp(armed, at(&syntax.FuncType{ParamList: []*syntax.Field{}}, pos), "func()")
		native = append(native, scopeGuardDefer(pos, armed, temp))
		text = append(text, scopeGuardDefer(pos, armed, temp))
	}
	fc.scopeHoists = nil
	fc.fl.addT(fc.fl.off(pos), fc.fl.off(pos), renderText(text, nil))
	return native
}

func (fc *funcLower) scopeStanza(pos syntax.Pos, items []scopeItem) (native, text []syntax.Stmt, tw *twin) {
	tw = &twin{}
	for _, it := range items {
		if it.join != nil {
			joinNative, joinText := fc.groupJoinWait(pos, *it.join)
			native = append(native, joinNative...)
			text = append(text, joinText...)
			continue
		}
		if it.hoisted {
			temp := fc.nextScope()
			native = append(native, scopeHolderStanza(pos, it.armed, temp))
			text = append(text, scopeHolderStanza(pos, it.armed, temp))
			continue
		}
		funN, argN, _, _ := fc.planUses(it.plan, nil)
		funT, argT, _, _ := fc.planUses(it.plan, tw)
		callN := callOf(pos, funN, argN)
		callN.HasDots = it.plan.dots
		callT := callOf(pos, funT, argT)
		callT.HasDots = it.plan.dots
		native = append(native, ifOf(pos, nil, nameOf(pos, it.armed), []syntax.Stmt{
			assignOf(pos, []syntax.Expr{nameOf(pos, it.armed)}, []syntax.Expr{nameOf(pos, "false")}),
			exprStmtOf(pos, callN),
		}))
		text = append(text, ifOf(pos, nil, nameOf(pos, it.armed), []syntax.Stmt{
			assignOf(pos, []syntax.Expr{nameOf(pos, it.armed)}, []syntax.Expr{nameOf(pos, "false")}),
			exprStmtOf(pos, callT),
		}))
	}
	return native, text, tw
}

func (fc *funcLower) scopeStmt(s syntax.Stmt, active []*scopeDeferCtx, stack []scopeFrame) []syntax.Stmt {
	switch s := s.(type) {
	case *syntax.ReturnStmt:
		return fc.scopeReturn(s, active)
	case *syntax.BranchStmt:
		return fc.scopeBranch(s, active, stack)
	case *syntax.BlockStmt:
		s.List = fc.scopeBlockList(s.List, s, fc.fl.off(s.Rbrace), active, stack)
		return []syntax.Stmt{s}
	case *syntax.IfStmt:
		s.Then.List = fc.scopeBlockList(s.Then.List, s.Then, fc.fl.off(s.Then.Rbrace), active, stack)
		switch els := s.Else.(type) {
		case *syntax.BlockStmt:
			els.List = fc.scopeBlockList(els.List, els, fc.fl.off(els.Rbrace), active, stack)
		case *syntax.IfStmt:
			res := fc.scopeStmt(els, active, stack)
			s.Else = res[0]
		}
		return []syntax.Stmt{s}
	case *syntax.ForStmt:
		outer := !scopeInLoop(stack)
		stack = append(stack, scopeFrame{loop: true})
		s.Body.List = fc.scopeBlockList(s.Body.List, s.Body, fc.fl.off(s.Body.Rbrace), active, stack)
		if outer {
			if pre := fc.drainScopeHoists(s.Pos()); pre != nil {
				return append(pre, s)
			}
		}
		return []syntax.Stmt{s}
	case *syntax.SwitchStmt:
		stack = append(stack, scopeFrame{})
		for i, c := range s.Body {
			c.Body = fc.scopeBlockList(c.Body, c, fc.caseEndOff(s, i), active, stack)
		}
		return []syntax.Stmt{s}
	case *syntax.SelectStmt:
		stack = append(stack, scopeFrame{})
		for i, c := range s.Body {
			c.Body = fc.scopeBlockList(c.Body, c, fc.commEndOff(s, i), active, stack)
		}
		return []syntax.Stmt{s}
	case *syntax.LabeledStmt:
		loop := false
		if _, ok := s.Stmt.(*syntax.ForStmt); ok {
			loop = true
		}
		outer := !scopeInLoop(stack)
		stack = append(stack, scopeFrame{label: s.Label.Value, loop: loop})
		res := fc.scopeStmt(s.Stmt, active, stack)
		if loop && outer {
			if pre := fc.drainScopeHoists(s.Label.Pos()); pre != nil {
				if len(res) == 1 {
					s.Stmt = res[0]
					return append(pre, s)
				}
				return append(pre, blockOf(s.Pos(), res))
			}
		}
		if len(res) == 1 {
			s.Stmt = res[0]
			return []syntax.Stmt{s}
		}
		s.Stmt = blockOf(s.Pos(), res)
		return []syntax.Stmt{s}
	case *syntax.GroupStmt:
		return fc.scopeGroupStmt(s, active, stack)
	default:
		return []syntax.Stmt{s}
	}
}

func (fc *funcLower) caseEndOff(s *syntax.SwitchStmt, i int) int {
	if i+1 < len(s.Body) {
		return fc.fl.off(s.Body[i+1].Pos())
	}
	return fc.fl.off(s.Rbrace)
}

func (fc *funcLower) commEndOff(s *syntax.SelectStmt, i int) int {
	if i+1 < len(s.Body) {
		return fc.fl.off(s.Body[i+1].Pos())
	}
	return fc.fl.off(s.Rbrace)
}

func scopeItemsFor(active []*scopeDeferCtx, d int) []scopeItem {
	var out []scopeItem
	for i := len(active) - 1; i >= 0; i-- {
		if active[i].depth <= d {
			continue
		}
		out = append(out, ownScopeItems(active[i])...)
	}
	return out
}

func (fc *funcLower) scopeReturn(s *syntax.ReturnStmt, active []*scopeDeferCtx) []syntax.Stmt {
	if len(active) == 0 {
		return []syntax.Stmt{s}
	}
	pos := s.Pos()
	items := scopeItemsFor(active, -1)
	results := syntax.UnpackListExpr(s.Results)
	if len(results) == 0 || (len(results) != len(fc.specs) && !(len(results) == 1 && len(fc.specs) > 1)) {
		native, text, tw := fc.scopeStanza(pos, items)
		fc.fl.addT(fc.fl.off(pos), fc.fl.off(pos), renderText(text, tw.subs))
		return append(native, s)
	}
	names := make([]syntax.Expr, len(fc.specs))
	namesT := make([]syntax.Expr, len(fc.specs))
	for i := range fc.specs {
		names[i] = nameOf(pos, fc.specs[i].name.Value)
		namesT[i] = nameOf(pos, fc.specs[i].name.Value)
	}
	tw := &twin{}
	stanzaN, stanzaT, stw := fc.scopeStanza(pos, items)
	tw.absorb(stw.subs)
	marks := make([]syntax.Expr, len(results))
	for i, v := range results {
		marks[i] = tw.mark(v.Pos(), fc.fl.lazyText(span{start: fc.fl.off(syntax.StartPos(v)), end: fc.fl.exprEndOf(v)}))
	}
	if fc.elementFrame && len(results) == 1 {
		results = append(slices.Clone(results), nilOf(pos))
		marks = append(marks, nilOf(pos))
	}
	native := append([]syntax.Stmt{assignOf(pos, names, results)}, stanzaN...)
	native = append(native, returnOf(pos, nil))
	text := append([]syntax.Stmt{assignOf(pos, namesT, marks)}, stanzaT...)
	text = append(text, returnOf(pos, nil))
	fl := fc.fl
	fl.addT(fl.off(pos), stmtEnd(fl.src, fl.off(pos)), renderText(text, tw.subs))
	return native
}

func (fc *funcLower) scopeBranch(s *syntax.BranchStmt, active []*scopeDeferCtx, stack []scopeFrame) []syntax.Stmt {
	if len(active) == 0 {
		return []syntax.Stmt{s}
	}
	pos := s.Pos()
	var items []scopeItem
	switch s.Tok {
	case syntax.Break, syntax.Continue:
		if s.Label != nil {
			items = fc.scopeLabeledItems(s.Label.Value, active, stack)
			if items == nil {
				return []syntax.Stmt{s}
			}
		} else {
			d := scopeTargetDepth(s.Tok, stack)
			if d < 0 {
				return []syntax.Stmt{s}
			}
			items = scopeItemsFor(active, d)
		}
	case syntax.Goto:
		if s.Label == nil {
			return []syntax.Stmt{s}
		}
		target, ok := fc.scopeLabels[s.Label.Value]
		if !ok {
			return []syntax.Stmt{s}
		}
		items = reverseScopeItems(active, fc, target)
	case syntax.Fallthrough:
		items = scopeItemsFor(active, -1)
	default:
		return []syntax.Stmt{s}
	}
	if len(items) == 0 {
		return []syntax.Stmt{s}
	}
	native, text, tw := fc.scopeStanza(pos, items)
	fc.fl.addT(fc.fl.off(pos), fc.fl.off(pos), renderText(text, tw.subs))
	return append(native, s)
}

func reverseScopeItems(active []*scopeDeferCtx, fc *funcLower, target syntax.Node) []scopeItem {
	var out []scopeItem
	for i := len(active) - 1; i >= 0; i-- {
		if fc.scopeContains(active[i].owner, target) {
			continue
		}
		out = append(out, ownScopeItems(active[i])...)
	}
	return out
}

func (fc *funcLower) scopeLabeledItems(label string, active []*scopeDeferCtx, stack []scopeFrame) []scopeItem {
	for k := len(stack) - 1; k >= 0; k-- {
		if stack[k].label == label {
			return scopeItemsFor(active, k)
		}
	}
	if d, ok := fc.scopeDeferLabels[label]; ok {
		return scopeItemsFor(active, d)
	}
	return nil
}

func scopeTargetDepth(tok syntax.Token, stack []scopeFrame) int {
	if tok == syntax.Continue {
		for k := len(stack) - 1; k >= 0; k-- {
			if stack[k].loop {
				return k
			}
		}
		return -1
	}
	return len(stack) - 1
}

func (fc *funcLower) scopeExitStanza(pos syntax.Pos) (native, text []syntax.Stmt, subs []textSub) {
	if len(fc.scopeCtxs) == 0 {
		return nil, nil, nil
	}
	off := fc.fl.off(pos)
	var items []scopeItem
	for i := len(fc.scopeCtxs) - 1; i >= 0; i-- {
		ctx := fc.scopeCtxs[i]
		if ctx.startOff <= off && off < ctx.endOff {
			items = append(items, ownScopeItems(ctx)...)
		}
	}
	if len(items) == 0 {
		return nil, nil, nil
	}
	native, text, tw := fc.scopeStanza(pos, items)
	return native, text, tw.subs
}
