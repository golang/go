// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"bytes"
	"cmp"
	"slices"
	"strconv"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/logos/lint"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

const logosMisplacedTryTag = "[correctness::misplaced_try] "

func logosMisplacedTryPending(terr types2.Error) bool {
	return strings.HasPrefix(terr.Msg, logosMisplacedTryTag)
}

func logosMisplacedTryResolvable(files []*syntax.File, pos syntax.Pos) bool {
	f := logosFileAt(files, pos)
	if f == nil {
		return false
	}
	stashed, _ := nextInputs([]*syntax.File{f})
	_, ok := stashed[f]
	return ok
}

func logosFlushMisplacedTry(m *posMap, files []*syntax.File, info *types2.Info, pending []syntax.Pos) {
	if len(pending) == 0 {
		return
	}
	plans := map[syntax.Stmt]*stmtPlan{}
	for _, pos := range pending {
		f := logosFileAt(files, pos)
		if f == nil {
			continue
		}
		stashed, _ := nextInputs([]*syntax.File{f})
		src, ok := stashed[f]
		if !ok {
			continue
		}
		t, stack := tryAt(f, pos)
		if t == nil {
			continue
		}
		var r lint.Result
		var offered bool
		if stmt, fn, good := enclosingSplitStmt(stack); good {
			plan := planFor(info, stmt, src, plans)
			multi := plan.ok && len(plan.tries) > 1
			first := multi && !plan.emitted
			residual := residualArityOK(info, stmt)
			switch {
			case first && residual && unconditionalAncestors(stack, stmt) && enclosingReturnsError(info, fn) &&
				planOrderOK(stmt, plan.tries, src):
				plan.emitted = true
				r, offered = splitResult(splitRequest{info: info, fn: fn, stmt: stmt, src: src, tries: plan.tries})
			case multi && plan.emitted:
				r, offered = misplacedTryBare(info, t, "")
			case plan.ok && len(plan.tries) == 1:
				if residual && unconditionalAncestors(stack, stmt) && enclosingReturnsError(info, fn) {
					r, offered = misplacedTryFix(info, fn, stmt, t, src, plan.tries[0])
				}
				if !offered {
					r, offered = misplacedTryBare(info, t, plan.note)
				}
			default:
				r, offered = misplacedTryBare(info, t, plan.note)
			}
		} else {
			r, offered = misplacedTryBare(info, t, "")
		}
		if offered {
			base.LogosReport(logosDiag(m, r))
		}
	}
}

func residualArityOK(info *types2.Info, stmt syntax.Stmt) bool {
	as, ok := stmt.(*syntax.AssignStmt)
	if !ok || as.Rhs == nil || (as.Op != 0 && as.Op != syntax.Def) {
		return true
	}
	rhs := syntax.UnpackListExpr(as.Rhs)
	if len(rhs) != 1 {
		return true
	}
	t := info.TypeOf(syntax.Unparen(rhs[0]))
	tup, isTuple := t.(*types2.Tuple)
	if !isTuple {
		return true
	}
	return tup.Len() == len(syntax.UnpackListExpr(as.Lhs))
}

type stmtPlan struct {
	tries   []trySite
	ok      bool
	note    string
	emitted bool
}

func planFor(info *types2.Info, stmt syntax.Stmt, src []byte, plans map[syntax.Stmt]*stmtPlan) *stmtPlan {
	if p, ok := plans[stmt]; ok {
		return p
	}
	p := &stmtPlan{}
	plans[stmt] = p
	type cand struct {
		expr   *syntax.TryExpr
		off    int
		argPos bool
	}
	var cands []cand
	blocked := false
	var stack []syntax.Node
	var scStack []bool
	syntax.Inspect(stmt, func(n syntax.Node) bool {
		if n == nil {
			stack = stack[:len(stack)-1]
			scStack = scStack[:len(scStack)-1]
			return false
		}
		switch n.(type) {
		case *syntax.FuncLit, *syntax.LambdaExpr:
			return false
		}
		sc := false
		if len(stack) > 0 {
			inherited := scStack[len(scStack)-1]
			switch p := stack[len(stack)-1].(type) {
			case *syntax.Operation:
				sc = inherited || (n == syntax.Node(p.Y) && (p.Op == syntax.AndAnd || p.Op == syntax.OrOr))
			case *syntax.CoalesceExpr:
				sc = inherited || n == syntax.Node(p.Y)
			default:
				sc = inherited
			}
		}
		if t, isTry := n.(*syntax.TryExpr); isTry {
			argPos := false
			if len(stack) > 0 {
				_, argPos = stack[len(stack)-1].(*syntax.CallExpr)
			}
			if sc {
				blocked = true
				p.note = shortCircuitNote
			} else if inner, isCall := syntax.Unparen(t.X).(*syntax.CallExpr); isCall {
				if off, has := tryOffset(src, syntax.StartPos(t)); has {
					if _, good := checkedCallResult(info, inner); good {
						cands = append(cands, cand{expr: t, off: off, argPos: argPos})
					} else {
						blocked = true
					}
				} else {
					blocked = true
				}
			} else {
				blocked = true
			}
		}
		stack = append(stack, n)
		scStack = append(scStack, sc)
		return true
	})
	if blocked {
		return p
	}
	if len(cands) == 0 {
		return p
	}
	slices.SortFunc(cands, func(a, b cand) int { return cmp.Compare(a.off, b.off) })
	p.tries = make([]trySite, len(cands))
	for i, c := range cands {
		call := syntax.Unparen(c.expr.X).(*syntax.CallExpr)
		tup, _ := checkedCallResult(info, call)
		p.tries[i] = trySite{expr: c.expr, call: call, tup: tup, argPos: c.argPos}
	}
	if ifInitFeedsTries(info, stmt, p.tries) {
		p.tries = nil
		return p
	}
	p.ok = true
	return p
}

func ifInitFeedsTries(info *types2.Info, stmt syntax.Stmt, tries []trySite) bool {
	ifs, ok := stmt.(*syntax.IfStmt)
	if !ok || ifs.Init == nil {
		return false
	}
	as, ok := ifs.Init.(*syntax.AssignStmt)
	if !ok {
		return true
	}
	written := map[types2.Object]bool{}
	for _, l := range syntax.UnpackListExpr(as.Lhs) {
		name, ok := syntax.Unparen(l).(*syntax.Name)
		if !ok {
			return true
		}
		if obj := cmp.Or(info.Defs[name], info.Uses[name]); obj != nil {
			written[obj] = true
		}
	}
	for _, t := range tries {
		reads := false
		syntax.Inspect(t.call, func(n syntax.Node) bool {
			if name, ok := n.(*syntax.Name); ok && written[info.Uses[name]] {
				reads = true
			}
			return !reads
		})
		if reads {
			return true
		}
	}
	return false
}

func planOrderOK(stmt syntax.Stmt, tries []trySite, src []byte) bool {
	last := tries[len(tries)-1]
	lastStart, ok1 := tryOffset(src, syntax.StartPos(last.expr))
	if !ok1 {
		return false
	}
	ok := true
	syntax.Inspect(stmt, func(n syntax.Node) bool {
		if !ok {
			return false
		}
		switch n := n.(type) {
		case *syntax.CallExpr:
			s, ok1 := tryOffset(src, syntax.StartPos(n))
			e, ok2 := callEnd(src, n)
			if !ok1 || !ok2 {
				ok = false
				return false
			}
			for _, t := range tries {
				ts, _ := tryOffset(src, syntax.StartPos(t.expr))
				te, _ := tryOffset(src, syntax.EndPos(t.expr))
				if s >= ts && e <= te {
					return true
				}
			}
			if e <= lastStart {
				ok = false
				return false
			}
		case *syntax.Operation:
			if n.Op != syntax.Recv {
				return true
			}
			s, ok1 := tryOffset(src, syntax.StartPos(n))
			e, ok2 := tryOffset(src, syntax.EndPos(n))
			if !ok1 || !ok2 {
				ok = false
				return false
			}
			for _, t := range tries {
				ts, _ := tryOffset(src, syntax.StartPos(t.expr))
				te, _ := tryOffset(src, syntax.EndPos(t.expr))
				if s >= ts && e <= te {
					return true
				}
			}
			if e <= lastStart {
				ok = false
				return false
			}
		}
		return true
	})
	return ok
}

func findTryStack(f *syntax.File, pos syntax.Pos) []syntax.Node {
	_, stack := tryAt(f, pos)
	return stack
}

func misplacedTryFix(info *types2.Info, fn syntax.Node, stmt syntax.Stmt, t *syntax.TryExpr, src []byte, site trySite) (lint.Result, bool) {
	call, ok := syntax.Unparen(t.X).(*syntax.CallExpr)
	if !ok || call != site.call {
		return lint.Result{}, false
	}
	if site.tup.Len() > 2 && !site.argPos {
		if _, isRet := stmt.(*syntax.ReturnStmt); !isRet {
			return lint.Result{}, false
		}
	}
	if !splitPreservesOrder(stmt, t, src) {
		return lint.Result{}, false
	}
	r := splitRequest{info: info, fn: fn, stmt: stmt, src: src, tries: []trySite{site}}
	return splitResult(r)
}

func misplacedTryBare(info *types2.Info, t *syntax.TryExpr, note string) (lint.Result, bool) {
	var origin string
	var line int
	if call, ok := syntax.Unparen(t.X).(*syntax.CallExpr); ok {
		origin = syntax.String(call)
		line = int(call.Pos().Line())
	}
	return lint.Result{
		Rule:     "correctness::misplaced_try",
		Severity: lint.Forbid,
		Pos:      syntax.StartPos(t),
		End:      syntax.EndPos(t),
		Message:  "? must end an assignment, expression statement, defer, or errdefer",
		Note:     note,
		Provenance: lint.Provenance{
			OriginExpr: origin,
			OriginLine: line,
		},
	}, true
}

func enclosingSig(info *types2.Info, fn syntax.Node) (*types2.Signature, bool) {
	switch n := fn.(type) {
	case *syntax.FuncDecl:
		obj := info.Defs[n.Name]
		fobj, ok := obj.(*types2.Func)
		if !ok || fobj == nil {
			return nil, false
		}
		return fobj.Signature(), true
	case *syntax.FuncLit:
		s, ok := info.TypeOf(n).(*types2.Signature)
		if !ok || s == nil {
			return nil, false
		}
		return s, true
	default:
		return nil, false
	}
}

func tryAt(f *syntax.File, pos syntax.Pos) (*syntax.TryExpr, []syntax.Node) {
	var stack, found []syntax.Node
	var target *syntax.TryExpr
	syntax.Inspect(f, func(n syntax.Node) bool {
		switch {
		case n == nil:
			stack = stack[:len(stack)-1]
			return false
		case found != nil:
			return false
		}
		stack = append(stack, n)
		if t, ok := n.(*syntax.TryExpr); ok && t.Pos() == pos {
			found = slices.Clone(stack)
			target = t
		}
		return true
	})
	return target, found
}

func checkedCallResult(info *types2.Info, call *syntax.CallExpr) (*types2.Tuple, bool) {
	typ := info.TypeOf(call)
	if typ == nil {
		return nil, false
	}
	tup, ok := typ.(*types2.Tuple)
	if !ok || tup.Len() < 2 {
		return nil, false
	}
	errType := types2.Universe.Lookup("error").Type()
	if !types2.Identical(tup.At(tup.Len()-1).Type(), errType) {
		return nil, false
	}
	nvals := tup.Len() - 1
	for i := range nvals {
		if tup.At(i).Type() == nil {
			return nil, false
		}
	}
	return tup, true
}

func enclosingSplitStmt(stack []syntax.Node) (syntax.Stmt, syntax.Node, bool) {
	tryIdx := len(stack) - 1
	stmtIdx := -1
	var stmt syntax.Stmt
	for i := tryIdx - 1; i >= 0; i-- {
		switch s := stack[i].(type) {
		case *syntax.AssignStmt:
			if s.Rhs == nil || (s.Op != 0 && s.Op != syntax.Def) {
				return nil, nil, false
			}
			if len(syntax.UnpackListExpr(s.Rhs)) != 1 {
				return nil, nil, false
			}
			for _, l := range syntax.UnpackListExpr(s.Lhs) {
				if _, ok := syntax.Unparen(l).(*syntax.Name); !ok {
					return nil, nil, false
				}
			}
			stmt, stmtIdx = s, i
		case *syntax.ExprStmt:
			stmt, stmtIdx = s, i
		case *syntax.IfStmt:
			if s.Cond == nil || stack[i+1] != syntax.Node(s.Cond) {
				return nil, nil, false
			}
			stmt, stmtIdx = s, i
		case *syntax.ReturnStmt:
			stmt, stmtIdx = s, i
		case *syntax.FuncLit, *syntax.LambdaExpr:
			return nil, nil, false
		case syntax.Stmt:
			return nil, nil, false
		default:
			continue
		}
		break
	}
	if stmtIdx < 0 {
		return nil, nil, false
	}
	parent := stack[stmtIdx-1]
	switch p := parent.(type) {
	case *syntax.BlockStmt, *syntax.CaseClause:
	case *syntax.CommClause:
		if p.Comm == stmt {
			return nil, nil, false
		}
	default:
		return nil, nil, false
	}
	for i := stmtIdx - 1; i >= 0; i-- {
		switch n := stack[i].(type) {
		case *syntax.FuncDecl:
			return stmt, n, true
		case *syntax.FuncLit:
			return stmt, n, true
		case *syntax.LambdaExpr:
			return nil, nil, false
		}
	}
	return nil, nil, false
}

func leftmostTry(stmt syntax.Stmt, t *syntax.TryExpr, src []byte) bool {
	want, ok := tryOffset(src, syntax.StartPos(t))
	if !ok {
		return false
	}
	best := -1
	syntax.Inspect(stmt, func(n syntax.Node) bool {
		if u, ok := n.(*syntax.TryExpr); ok {
			if off, ok := tryOffset(src, syntax.StartPos(u)); ok {
				if best < 0 || off < best {
					best = off
				}
			}
		}
		return true
	})
	return best == want
}

func unconditionalAncestors(stack []syntax.Node, stmt syntax.Stmt) bool {
	for i := len(stack) - 2; i >= 0; i-- {
		if stack[i] == stmt {
			return true
		}
		child := stack[i+1]
		switch n := stack[i].(type) {
		case *syntax.CallExpr:
			if _, ok := syntax.Unparen(n.Fun).(*syntax.OptSelectExpr); ok && child != syntax.Node(n.Fun) {
				return false
			}
		case *syntax.OptSelectExpr:
			if child != syntax.Node(n.X) {
				return false
			}
		case *syntax.SwitchExpr, *syntax.ArmClause,
			*syntax.CaseClause, *syntax.CommClause, *syntax.ErrdeferStmt:
			return false
		case *syntax.SelectorExpr, *syntax.IndexExpr,
			*syntax.SliceExpr, *syntax.AssertExpr, *syntax.ParenExpr,
			*syntax.CompositeLit, *syntax.KeyValueExpr, *syntax.ListExpr,
			*syntax.TryExpr, *syntax.Operation, *syntax.CoalesceExpr:
		default:
			return false
		}
	}
	return false
}

func splitPreservesOrder(stmt syntax.Stmt, t *syntax.TryExpr, src []byte) bool {
	start, ok1 := tryOffset(src, syntax.StartPos(t))
	end, ok2 := tryOffset(src, syntax.EndPos(t))
	if !ok1 || !ok2 || end <= start {
		return false
	}
	ok := true
	syntax.Inspect(stmt, func(n syntax.Node) bool {
		if !ok {
			return false
		}
		switch n := n.(type) {
		case *syntax.CallExpr:
			s, ok1 := tryOffset(src, syntax.StartPos(n))
			e, ok2 := callEnd(src, n)
			if !ok1 || !ok2 {
				ok = false
				return false
			}
			if s >= start && e <= end {
				return true
			}
			if e <= start {
				ok = false
				return false
			}
		case *syntax.Operation:
			if n.Op != syntax.Recv {
				return true
			}
			s, ok1 := tryOffset(src, syntax.StartPos(n))
			e, ok2 := tryOffset(src, syntax.EndPos(n))
			if !ok1 || !ok2 {
				ok = false
				return false
			}
			if s >= start && e <= end {
				return true
			}
			if e <= start {
				ok = false
				return false
			}
		}
		return true
	})
	return ok
}

func enclosingReturnsError(info *types2.Info, fn syntax.Node) bool {
	sig, ok := enclosingSig(info, fn)
	if !ok || sig == nil || sig.Results().Len() == 0 {
		return false
	}
	errType := types2.Universe.Lookup("error").Type()
	return types2.Identical(sig.Results().At(sig.Results().Len()-1).Type(), errType)
}

type splitRequest struct {
	info  *types2.Info
	fn    syntax.Node
	stmt  syntax.Stmt
	src   []byte
	tries []trySite
}

type trySite struct {
	expr   *syntax.TryExpr
	call   *syntax.CallExpr
	tup    *types2.Tuple
	argPos bool
}

const shortCircuitNote = "hoisting a `?` out of a short-circuit operand would run it even when the left side decides the result, so the `?` must stay bound inside the branch and no split is offered"

func returnSplitSuffix(r splitRequest) (string, bool) {
	rs, ok := r.stmt.(*syntax.ReturnStmt)
	if !ok {
		return "", true
	}
	if rs.Results == nil {
		return "", false
	}
	exprs := syntax.UnpackListExpr(rs.Results)
	tryCount := 0
	for _, e := range exprs {
		for _, t := range r.tries {
			if syntax.Unparen(e) == syntax.Node(t.expr) {
				tryCount++
				break
			}
		}
	}
	if tryCount == 0 {
		return "", true
	}
	sig, ok := enclosingSig(r.info, r.fn)
	if !ok || sig == nil || sig.Results().Len() == 0 {
		return "", false
	}
	nvals := 0
	for _, t := range r.tries {
		nvals += t.tup.Len() - 1
	}
	arity := len(exprs) - tryCount + nvals
	total := sig.Results().Len()
	if arity != total && arity != total-1 {
		return "", false
	}
	direct := -1
	for i, e := range exprs {
		if syntax.Unparen(e) == syntax.Node(r.tries[0].expr) {
			direct = i
			break
		}
	}
	if direct < 0 {
		return "", false
	}
	pos := direct
	for _, t := range r.tries {
		for i := range t.tup.Len() - 1 {
			if pos+i >= total {
				return "", false
			}
			if !types2.AssignableTo(t.tup.At(i).Type(), sig.Results().At(pos+i).Type()) {
				return "", false
			}
		}
		pos += t.tup.Len() - 1
	}
	if arity == total-1 {
		return ", nil", true
	}
	return "", true
}

func splitResult(r splitRequest) (lint.Result, bool) {
	startPos := splitStmtStart(r.stmt)
	start, ok1 := tryOffset(r.src, startPos)
	stmtEnd, ok6 := splitStmtEnd(r.src, r.stmt)
	if !ok1 || !ok6 {
		return lint.Result{}, false
	}
	if !(start < stmtEnd) {
		return lint.Result{}, false
	}
	suffix, ok := returnSplitSuffix(r)
	if !ok {
		return lint.Result{}, false
	}
	indentStart := start
	for indentStart > 0 && (r.src[indentStart-1] == ' ' || r.src[indentStart-1] == '\t') {
		indentStart--
	}
	indent := string(r.src[indentStart:start])
	stmtSrc := string(r.src[start:stmtEnd])
	type seg struct {
		relStart, relEnd int
		callStart        int
		callEnd          int
		gap              string
		hoist            string
		bind             string
		names            string
		top              bool
	}
	var segs []seg
	sites := slices.Clone(r.tries)
	slices.SortFunc(sites, func(a, b trySite) int {
		ae, _ := tryOffset(r.src, syntax.EndPos(a.expr))
		be, _ := tryOffset(r.src, syntax.EndPos(b.expr))
		return cmp.Compare(ae, be)
	})
	used := usedNames(r.fn)
	for _, site := range sites {
		tryStart, ok2 := tryOffset(r.src, syntax.StartPos(site.expr))
		tryEnd, ok3 := tryOffset(r.src, syntax.EndPos(site.expr))
		callStart, ok4 := tryOffset(r.src, syntax.StartPos(site.call))
		callEndOff, ok5 := callEnd(r.src, site.call)
		if !ok2 || !ok3 || !ok4 || !ok5 {
			return lint.Result{}, false
		}
		if !(start <= tryStart && tryStart < tryEnd && tryEnd <= stmtEnd) {
			return lint.Result{}, false
		}
		if !(start <= callStart && callStart < callEndOff && callEndOff <= tryEnd) {
			return lint.Result{}, false
		}
		gapSrc := string(r.src[callEndOff:tryEnd])
		if len(gapSrc) == 0 || gapSrc[len(gapSrc)-1] != '?' {
			return lint.Result{}, false
		}
		segs = append(segs, seg{
			relStart:  tryStart - start,
			relEnd:    tryEnd - start,
			callStart: callStart - start,
			callEnd:   callEndOff - start,
			gap:       gapSrc,
		})
	}
	for i, site := range sites {
		sg := &segs[i]
		suffixI := ""
		if i == len(sites)-1 {
			suffixI = suffix
		}
		picked := strings.Join(freshNames(used, site.call, site.tup.Len()-1), ", ")
		sg.bind = picked
		sg.names = picked + suffixI
	}
	for i := range segs {
		sg := &segs[i]
		callSrc := string(r.src[start+sg.callStart : start+sg.callEnd])
		order := make([]int, len(segs))
		for j := range order {
			order[j] = j
		}
		slices.SortFunc(order, func(a, b int) int { return cmp.Compare(segs[b].relStart, segs[a].relStart) })
		for _, j := range order {
			if j == i {
				continue
			}
			inner := segs[j]
			if inner.relStart < sg.callStart || inner.relEnd > sg.callEnd {
				continue
			}
			direct := true
			for _, k := range order {
				if k == i || k == j {
					continue
				}
				mid := segs[k]
				if mid.relStart < sg.callStart || mid.relEnd > sg.callEnd {
					continue
				}
				if mid.relStart <= inner.relStart && inner.relEnd <= mid.relEnd {
					direct = false
					break
				}
			}
			if !direct {
				continue
			}
			callSrc = callSrc[:inner.relStart-sg.callStart] + inner.bind + callSrc[inner.relEnd-sg.callStart:]
		}
		sg.hoist = sg.bind + " := " + callSrc + sg.gap + "\n" + indent
	}
	for i := range segs {
		segs[i].top = true
		for j := range segs {
			if j != i && segs[j].relStart <= segs[i].relStart && segs[i].relEnd <= segs[j].relEnd {
				segs[i].top = false
				break
			}
		}
	}
	var hoist, rest strings.Builder
	prev := 0
	for _, sg := range segs {
		hoist.WriteString(sg.hoist)
		if !sg.top {
			continue
		}
		rest.WriteString(stmtSrc[prev:sg.relStart])
		rest.WriteString(sg.names)
		prev = sg.relEnd
	}
	rest.WriteString(stmtSrc[prev:])
	replacement := hoist.String() + rest.String()
	first := r.tries[0]
	endLine, endCol := offsetLineCol(r.src, stmtEnd)
	return lint.Result{
		Rule:     "correctness::misplaced_try",
		Severity: lint.Forbid,
		Pos:      startPos,
		End:      syntax.MakePos(startPos.Base(), endLine, endCol),
		Message:  "? must end an assignment, expression statement, defer, or errdefer",
		Provenance: lint.Provenance{
			OriginExpr:   syntax.String(first.call),
			OriginLine:   int(first.call.Pos().Line()),
			InferredType: types2.TypeString(first.tup.At(0).Type(), func(p *types2.Package) string { return p.Name() }),
		},
		Fixes: []lint.Fix{
			{Description: "Split the expression so `?` ends its own statement.", Replacement: replacement},
		},
	}, true
}

func splitStmtStart(s syntax.Stmt) syntax.Pos {
	if es, ok := s.(*syntax.ExprStmt); ok {
		return syntax.StartPos(es.X)
	}
	return syntax.StartPos(s)
}

func splitStmtEnd(src []byte, s syntax.Stmt) (int, bool) {
	end, ok := tryOffset(src, syntax.EndPos(s))
	if !ok {
		return 0, false
	}
	syntax.Inspect(s, func(n syntax.Node) bool {
		if call, isCall := n.(*syntax.CallExpr); isCall {
			if e, ok := callEnd(src, call); ok {
				end = max(end, e)
			}
		}
		return true
	})
	for end < len(src) && (src[end] == ')' || src[end] == ']' || src[end] == '}') {
		end++
	}
	if end < 0 || end > len(src) {
		return 0, false
	}
	return end, true
}

func usedNames(fn syntax.Node) map[string]bool {
	used := map[string]bool{}
	syntax.Inspect(fn, func(n syntax.Node) bool {
		if id, ok := n.(*syntax.Name); ok {
			used[id.Value] = true
		}
		return true
	})
	return used
}

func freshNames(used map[string]bool, call *syntax.CallExpr, n int) []string {
	base := "tmp"
	switch fun := syntax.Unparen(call.Fun).(type) {
	case *syntax.Name:
		if fun.Value != "" && fun.Value != "_" {
			base = fun.Value
		}
	case *syntax.SelectorExpr:
		if fun.Sel != nil && fun.Sel.Value != "" && fun.Sel.Value != "_" {
			base = fun.Sel.Value
		}
	}
	names := make([]string, 0, n)
	for i := 0; len(names) < n; i++ {
		suffix := ""
		if i > 0 {
			suffix = strconv.Itoa(i)
		}
		cand := "_" + base + suffix
		if isReservedIdent(cand) {
			cand = "_fix" + cand
		}
		if cand != "_" && !used[cand] {
			used[cand] = true
			names = append(names, cand)
		}
	}
	return names
}

func isReservedIdent(name string) bool {
	return strings.HasPrefix(name, "_logos_") || strings.HasPrefix(name, "Logos_")
}

func tryOffset(src []byte, pos syntax.Pos) (int, bool) {
	off := 0
	for line := uint(1); line < pos.Line(); line++ {
		i := bytes.IndexByte(src[off:], '\n')
		if i < 0 {
			return 0, false
		}
		off += i + 1
	}
	off += int(pos.Col()) - 1
	if off < 0 || off > len(src) {
		return 0, false
	}
	return off, true
}

func callEnd(src []byte, call *syntax.CallExpr) (int, bool) {
	if end, ok := syntax.ParenEnd(src, call.Pos(), syntax.LogosMode); ok {
		return tryOffset(src, end)
	}
	return 0, false
}

func offsetLineCol(src []byte, off int) (uint, uint) {
	line := uint(1)
	start := 0
	for i := 0; i < off && i < len(src); i++ {
		if src[i] == '\n' {
			line++
			start = i + 1
		}
	}
	return line, uint(off-start) + 1
}
