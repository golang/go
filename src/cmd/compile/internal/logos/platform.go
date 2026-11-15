// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"slices"

	"cmd/compile/internal/syntax"
)

// This file lowers M3-S5 platform-pointer stores (SPEC T-202): every
// expression in info.LogosChecks held a platform pointer for a declared
// non-nil location, so it becomes _logos_rt.MustNonNil(expr,
// &_logos_siteK). The native tree gains a call node while the eject text
// gains an exact-span splice of the same call, preserving surrounding
// comments and formatting. lowerPlatform runs first in lowerFile, so
// later passes fold its splices into theirs through textOf.

type platformJob struct {
	expr syntax.Expr
	span span
	set  func(syntax.Expr)
}

// lowerPlatform wraps every recorded platform store of this file.
func (fl *fileLower) lowerPlatform() {
	checks := fl.pkg.info.LogosChecks
	if len(checks) == 0 {
		return
	}
	var jobs []platformJob
	collect := func(e syntax.Expr, set func(syntax.Expr)) {
		if _, ok := checks[e]; !ok {
			return
		}
		sp, ok := fl.platformSpan(e)
		if !ok {
			return
		}
		jobs = append(jobs, platformJob{expr: e, span: sp, set: set})
	}
	collectList := func(list []syntax.Expr, set func(int, syntax.Expr)) {
		for i, e := range list {
			collect(e, func(n syntax.Expr) { set(i, n) })
		}
	}
	collectSingle := func(e syntax.Expr, set func(syntax.Expr)) {
		if l, ok := e.(*syntax.ListExpr); ok {
			collectList(l.ElemList, func(i int, n syntax.Expr) { l.ElemList[i] = n })
			return
		}
		collect(e, set)
	}
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.AssignStmt:
			collectSingle(n.Rhs, func(e syntax.Expr) { n.Rhs = e })
		case *syntax.VarDecl:
			if n.Values != nil {
				collectSingle(n.Values, func(e syntax.Expr) { n.Values = e })
			}
		case *syntax.CallExpr:
			collectList(n.ArgList, func(i int, e syntax.Expr) { n.ArgList[i] = e })
		case *syntax.ReturnStmt:
			if n.Results != nil {
				collectSingle(n.Results, func(e syntax.Expr) { n.Results = e })
			}
		case *syntax.CompositeLit:
			for i, e := range n.ElemList {
				if kv, ok := e.(*syntax.KeyValueExpr); ok {
					collect(kv.Value, func(v syntax.Expr) { kv.Value = v })
					continue
				}
				collect(e, func(v syntax.Expr) { n.ElemList[i] = v })
			}
		case *syntax.SendStmt:
			collect(n.Value, func(e syntax.Expr) { n.Value = e })
		}
		return true
	})
	slices.SortStableFunc(jobs, func(a, b platformJob) int {
		if d := (a.span.end - a.span.start) - (b.span.end - b.span.start); d != 0 {
			return d
		}
		return a.span.start - b.span.start
	})
	for _, j := range jobs {
		inner := fl.textOf(j.span)
		pos := j.expr.Pos()
		site := fl.newSite(pos, pos.Line())
		wrapped := callOf(pos, rtSel(pos, "MustNonNil"), []syntax.Expr{
			j.expr,
			at(&syntax.Operation{Op: syntax.And, X: nameOf(pos, site)}, pos),
		})
		j.set(wrapped)
		fl.wrapSpan[wrapped] = j.span
		fl.addT(j.span.start, j.span.end, cat(
			gen(rtAlias+".MustNonNil("), inner, gen(", &"+site+")")))
	}
}

// platformSpan resolves the exact source span of a recorded expression.
// The end is the expression's own end: the generic scanner stops before
// a depth-zero brace, so a for-post check cannot swallow the loop body.
func (fl *fileLower) platformSpan(e syntax.Expr) (span, bool) {
	if ws, ok := fl.wrapSpan[syntax.Unparen(e)]; ok {
		return ws, true
	}
	start := fl.exprStart(e)
	end := fl.exprEndOf(e)
	if end <= start {
		return span{}, false
	}
	return span{start: start, end: end}, true
}
