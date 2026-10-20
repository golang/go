// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"strconv"

	"cmd/compile/internal/syntax"
)

// This file lowers M6-S10 `scope defer` statements (T-104.7). A scope
// defer runs its cleanup when control leaves the enclosing block, for
// every exit: fallthrough, return, break, continue, labeled forms, goto
// out of the block, and panic.
//
// There is no wrapper closure and no unwind table. Each scope defer
// binds its callee and arguments at the statement (Go defer evaluation
// order) behind an armed flag, registers an open-coded guard `defer`
// for the panic path, and every exit in the rest of the block runs the
// armed cleanups inline, innermost block first and last-in-first-out
// within a block:
//
//	{
//		_logos_f0, _logos_a0 := mu.Lock, ...
//		_logos_sd0 := true
//		defer func() {
//			if _logos_sd0 {
//				_logos_sd0 = false
//				_logos_f0(_logos_a0)
//			}
//		}()
//		work()
//		if _logos_sd0 {
//			_logos_sd0 = false
//			_logos_f0(_logos_a0)
//		}
//	}
//
// The armed flag covers jumps over the statement: a `goto` past a scope
// defer leaves it disarmed, so later exits skip it exactly as if the
// `defer` had never registered. Plain `defer` statements in the same
// tail are untouched and keep firing at function exit. A `?` error
// return generated later inside the tail picks up the stanza through
// exitPair, which consults the recorded tail spans.

// scopeItem is one scope defer of one block: its capture plan, its armed
// flag spelling, and its statement position.
type scopeItem struct {
	plan  *callPlan
	armed string
	pos   syntax.Pos
}

// scopeDeferCtx is the scope-defer state of one block: the tail region
// it covers, its defers in source order, and the break-target depth at
// which it was created.
type scopeDeferCtx struct {
	owner    syntax.Node
	startOff int
	endOff   int
	items    []scopeItem
	depth    int
}

// scopeFrame is one break/continue/goto target enclosing the current
// point: a loop, switch, select, or labeled statement.
type scopeFrame struct {
	node  syntax.Node
	label string
	loop  bool
	jump  bool
}

// scopeActiveItems returns the runnable items of active innermost first.
func scopeActiveItems(active []*scopeDeferCtx) []scopeItem {
	var out []scopeItem
	for i := len(active) - 1; i >= 0; i-- {
		out = append(out, ownScopeItems(active[i])...)
	}
	return out
}

// ownScopeItems returns one block's items last-in-first-out.
func ownScopeItems(ctx *scopeDeferCtx) []scopeItem {
	items := ctx.items
	out := make([]scopeItem, len(items))
	for i, it := range items {
		out[len(items)-1-i] = it
	}
	return out
}

// nextScope returns the next per-function scope-defer temporary spelling.
func (fc *funcLower) nextScope() string {
	spelling := "_logos_sd" + strconv.Itoa(fc.scopeSeq)
	fc.scopeSeq++
	return spelling
}

// lowerScopeDefers runs the scope-defer pre-pass over the function body:
// it emits preludes, rewrites block exits, and records tail spans for
// exitPair. Statement lowering afterwards sees only plain nodes.
func (fc *funcLower) lowerScopeDefers() {
	if !fc.hasScopeDefer {
		return
	}
	fc.collectScopeInfo()
	fl := fc.fl
	fc.body.List = fc.scopeBlockList(fc.body.List, fc.body, fl.off(fc.body.Rbrace), nil, nil)
}

// collectScopeInfo records the parent of every statement-level node and
// every label of the function body, skipping nested function literals
// (labels there belong to the literal's own funcLower).
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
		default:
			syntax.Inspect(n, func(x syntax.Node) bool {
				if _, ok := x.(*syntax.FuncLit); ok {
					return false
				}
				return true
			})
		}
	}
	rec(fc.body, nil)
}

// scopeContains reports whether target lies inside block.
func (fc *funcLower) scopeContains(block, target syntax.Node) bool {
	for n := target; n != nil; n = fc.scopeParents[n] {
		if n == block {
			return true
		}
	}
	return false
}

// scopeBlockList lowers one statement list that forms a scope block,
// owned by owner and ending at endOff. Scope defers among the direct
// members open one context for the rest of the list.
func (fc *funcLower) scopeBlockList(list []syntax.Stmt, owner syntax.Node, endOff int, active []*scopeDeferCtx, stack []scopeFrame) []syntax.Stmt {
	out := make([]syntax.Stmt, 0, len(list))
	cur := active
	var ctx *scopeDeferCtx
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
		native, text, tw := fc.scopeStanza(ctx.items[0].pos, ownScopeItems(ctx))
		out = append(out, native...)
		fl := fc.fl
		gapStart := stmtEnd(fl.src, fl.off(syntax.StartPos(list[len(list)-1])))
		prefix := leftoverComments(fl, span{start: gapStart, end: endOff}, nil)
		gap := prefix + renderText(text, tw.subs)
		if prefix == "" {
			gap = "\n" + gap
		}
		fl.add(gapStart, endOff, gap)
	}
	return out
}

// scopeFallthrough prefixes a case-body fallthrough with its own block's
// stanza only: control stays inside every enclosing block, so outer
// defers must not run.
func (fc *funcLower) scopeFallthrough(br *syntax.BranchStmt, ctx *scopeDeferCtx) []syntax.Stmt {
	if ctx == nil {
		return []syntax.Stmt{br}
	}
	native, text, tw := fc.scopeStanza(br.Pos(), ownScopeItems(ctx))
	fc.fl.add(fc.fl.off(br.Pos()), fc.fl.off(br.Pos()), renderText(text, tw.subs))
	return append(native, br)
}

// scopeEndsExiting reports whether list ends with a statement that leaves
// unconditionally, making a fallthrough stanza dead code.
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

// unwrapScopeLabels peels labeled statements, reporting the innermost
// statement and the labels outside-in.
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

// scopeLabeledPrelude emits the prelude of a scope defer carrying labels
// (`L: scope defer f()`): the labels stay on the first prelude statement
// in source and tree, so a `goto L` re-registers the defer.
func (fc *funcLower) scopeLabeledPrelude(sd *syntax.ScopeDeferStmt, labels []*syntax.Name, ctx *scopeDeferCtx) []syntax.Stmt {
	stmtOff, prefix := fc.fl.off(sd.Pos()), ""
	if len(labels) > 0 {
		stmtOff, prefix = fc.fl.off(labels[0].Pos()), fc.labelPrefixText(sd, labels)
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

// labelPrefixText spells the source from the first label to the scope
// statement, so the prelude replacement keeps `L1: L2:` verbatim.
func (fc *funcLower) labelPrefixText(sd *syntax.ScopeDeferStmt, labels []*syntax.Name) string {
	return string(fc.fl.src[fc.fl.off(labels[0].Pos()):fc.fl.off(sd.Pos())])
}

// scopePrelude emits the capture bindings, armed flag, and guard defer
// of one scope defer, replacing its statement span.
func (fc *funcLower) scopePrelude(sd *syntax.ScopeDeferStmt, ctx *scopeDeferCtx, stmtOff int, prefix string) []syntax.Stmt {
	fl := fc.fl
	pos := sd.Pos()
	call, ok := syntax.Unparen(sd.Call).(*syntax.CallExpr)
	if !ok {
		return []syntax.Stmt{sd}
	}
	plan, planned := fc.planCall(call)
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
	guardN := ifOf(pos, nil, armedN, []syntax.Stmt{
		assignOf(pos, []syntax.Expr{nameOf(pos, item.armed)}, []syntax.Expr{nameOf(pos, "false")}),
		exprStmtOf(pos, callN),
	})
	guardT := ifOf(pos, nil, armedT, []syntax.Stmt{
		assignOf(pos, []syntax.Expr{nameOf(pos, item.armed)}, []syntax.Expr{nameOf(pos, "false")}),
		exprStmtOf(pos, callT),
	})
	var native []syntax.Stmt
	var text []syntax.Stmt
	if len(capNamesN) > 0 {
		native = append(native, defineOf(pos, capNamesN, capValsN))
		text = append(text, defineOf(pos, capNamesT, capValsT))
	}
	native = append(native,
		defineOf(pos, []syntax.Expr{nameOf(pos, item.armed)}, []syntax.Expr{nameOf(pos, "true")}),
		deferOf(pos, callOf(pos, deferClosure(pos, []syntax.Stmt{guardN}), nil)),
	)
	text = append(text,
		defineOf(pos, []syntax.Expr{nameOf(pos, item.armed)}, []syntax.Expr{nameOf(pos, "true")}),
		deferOf(pos, callOf(pos, deferClosure(pos, []syntax.Stmt{guardT}), nil)),
	)
	spanEnd := skipCloseParens(fl.src, plan.end)
	kept := leftoverComments(fl, span{start: stmtOff, end: spanEnd}, planSpans(plan))
	fl.add(stmtOff, spanEnd, prefix+kept+renderText(text, tw.subs))
	return native
}

// scopeStanza builds the runnable cleanup sequence for items twice:
// native with user nodes, text with markers. Items run innermost first,
// last-in-first-out; each fires only while armed.
func (fc *funcLower) scopeStanza(pos syntax.Pos, items []scopeItem) (native, text []syntax.Stmt, tw *twin) {
	tw = &twin{}
	for _, it := range items {
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

// scopeStmt lowers one statement inside a scope region, threading the
// active contexts and break-target stack.
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
		stack = append(stack, scopeFrame{node: s, loop: true, jump: true})
		s.Body.List = fc.scopeBlockList(s.Body.List, s.Body, fc.fl.off(s.Body.Rbrace), active, stack)
		return []syntax.Stmt{s}
	case *syntax.SwitchStmt:
		stack = append(stack, scopeFrame{node: s, jump: true})
		for i, c := range s.Body {
			c.Body = fc.scopeBlockList(c.Body, c, fc.caseEndOff(s, i), active, stack)
		}
		return []syntax.Stmt{s}
	case *syntax.SelectStmt:
		stack = append(stack, scopeFrame{node: s, jump: true})
		for i, c := range s.Body {
			c.Body = fc.scopeBlockList(c.Body, c, fc.commEndOff(s, i), active, stack)
		}
		return []syntax.Stmt{s}
	case *syntax.LabeledStmt:
		loop := false
		if _, ok := s.Stmt.(*syntax.ForStmt); ok {
			loop = true
		}
		stack = append(stack, scopeFrame{node: s, label: s.Label.Value, loop: loop, jump: true})
		res := fc.scopeStmt(s.Stmt, active, stack)
		if len(res) == 1 {
			s.Stmt = res[0]
			return []syntax.Stmt{s}
		}
		s.Stmt = blockOf(s.Pos(), res)
		return []syntax.Stmt{s}
	case *syntax.GroupStmt:
		return fc.scopeGroupStmt(s, active, stack) // M8-S2 hook: scope defers bind inline in group bodies.
	default:
		return []syntax.Stmt{s}
	}
}

// caseEndOff returns the offset at which a case body ends: the next
// clause, or the switch's closing brace.
func (fc *funcLower) caseEndOff(s *syntax.SwitchStmt, i int) int {
	if i+1 < len(s.Body) {
		return fc.fl.off(s.Body[i+1].Pos())
	}
	return fc.fl.off(s.Rbrace)
}

// commEndOff returns the offset at which a comm body ends: the next
// clause, or the select's closing brace.
func (fc *funcLower) commEndOff(s *syntax.SelectStmt, i int) int {
	if i+1 < len(s.Body) {
		return fc.fl.off(s.Body[i+1].Pos())
	}
	return fc.fl.off(s.Rbrace)
}

// scopeItemsFor returns the runnable items of the contexts with depth
// greater than d, innermost first.
func scopeItemsFor(active []*scopeDeferCtx, d int) []scopeItem {
	var out []scopeItem
	for i := len(active) - 1; i >= 0; i-- {
		if active[i].depth <= d {
			continue
		}
		items := active[i].items
		for j := len(items) - 1; j >= 0; j-- {
			out = append(out, items[j])
		}
	}
	return out
}

// scopeReturn rewrites a return inside a scope region: named results take
// the values first (so cleanups see them), then the stanza runs, then a
// bare return exits.
func (fc *funcLower) scopeReturn(s *syntax.ReturnStmt, active []*scopeDeferCtx) []syntax.Stmt {
	if len(active) == 0 {
		return []syntax.Stmt{s}
	}
	pos := s.Pos()
	items := scopeActiveItems(active)
	results := syntax.UnpackListExpr(s.Results)
	if len(results) == 0 || len(results) != len(fc.specs) {
		native, text, tw := fc.scopeStanza(pos, items)
		fc.fl.add(fc.fl.off(pos), fc.fl.off(pos), renderText(text, tw.subs))
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
		marks[i] = tw.mark(v.Pos(), fc.fl.textOf(span{start: fc.fl.off(syntax.StartPos(v)), end: fc.fl.off(syntax.EndPos(v))}))
	}
	native := append([]syntax.Stmt{assignOf(pos, names, results)}, stanzaN...)
	native = append(native, returnOf(pos, nil))
	text := append([]syntax.Stmt{assignOf(pos, namesT, marks)}, stanzaT...)
	text = append(text, returnOf(pos, nil))
	fl := fc.fl
	fl.add(fl.off(pos), stmtEnd(fl.src, fl.off(pos)), renderText(text, tw.subs))
	return native
}

// scopeBranch prefixes a block-leaving branch with the stanza of every
// context it leaves.
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
		items = scopeActiveItems(active)
	default:
		return []syntax.Stmt{s}
	}
	if len(items) == 0 {
		return []syntax.Stmt{s}
	}
	native, text, tw := fc.scopeStanza(pos, items)
	fc.fl.add(fc.fl.off(pos), fc.fl.off(pos), renderText(text, tw.subs))
	return append(native, s)
}

// reverseScopeItems collects runnable items of contexts not containing
// target, innermost first.
func reverseScopeItems(active []*scopeDeferCtx, fc *funcLower, target syntax.Node) []scopeItem {
	var out []scopeItem
	for i := len(active) - 1; i >= 0; i-- {
		if fc.scopeContains(active[i].owner, target) {
			continue
		}
		items := active[i].items
		for j := len(items) - 1; j >= 0; j-- {
			out = append(out, items[j])
		}
	}
	return out
}

// scopeLabeledItems resolves break/continue with a label: contexts
// inside the labeled statement run; a label outside the traversal
// leaves the statement alone.
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

// scopeTargetDepth returns the stack index of the innermost break or
// continue target, or -1 when the traversal holds none.
func scopeTargetDepth(tok syntax.Token, stack []scopeFrame) int {
	for k := len(stack) - 1; k >= 0; k-- {
		if tok == syntax.Continue {
			if stack[k].loop {
				return k
			}
			continue
		}
		if stack[k].jump {
			return k
		}
	}
	return -1
}

// scopeExitStanza builds the stanza for a generated exit (a `?` error
// return) at pos when pos lies inside a recorded scope tail.
func (fc *funcLower) scopeExitStanza(pos syntax.Pos) (native, text []syntax.Stmt, subs []textSub) {
	if len(fc.scopeCtxs) == 0 {
		return nil, nil, nil
	}
	off := fc.fl.off(pos)
	var items []scopeItem
	for i := len(fc.scopeCtxs) - 1; i >= 0; i-- {
		ctx := fc.scopeCtxs[i]
		if ctx.startOff <= off && off < ctx.endOff {
			for j := len(ctx.items) - 1; j >= 0; j-- {
				items = append(items, ctx.items[j])
			}
		}
	}
	if len(items) == 0 {
		return nil, nil, nil
	}
	native, text, tw := fc.scopeStanza(pos, items)
	return native, text, tw.subs
}
