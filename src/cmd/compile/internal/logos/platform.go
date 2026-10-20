// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"go/token"
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
		j.set(callOf(pos, rtSel(pos, "MustNonNil"), []syntax.Expr{
			j.expr,
			at(&syntax.Operation{Op: syntax.And, X: nameOf(pos, site)}, pos),
		}))
		fl.add(j.span.start, j.span.end, rtAlias+".MustNonNil("+inner+", &"+site+")")
	}
}

// platformLeftmost finds the leftmost token of e, descending through
// selectors, indexes, slices, assertions, calls, and binary operations,
// whose Pos sits at an interior token.
func platformLeftmost(e syntax.Expr) syntax.Pos {
	for {
		switch n := syntax.Unparen(e).(type) {
		case *syntax.SelectorExpr:
			e = n.X
		case *syntax.IndexExpr:
			e = n.X
		case *syntax.SliceExpr:
			e = n.X
		case *syntax.AssertExpr:
			e = n.X
		case *syntax.CallExpr:
			e = n.Fun
		case *syntax.Operation:
			if n.Y == nil {
				return e.Pos()
			}
			e = n.X
		default:
			return e.Pos()
		}
	}
}

// platformSpan resolves the exact source span of a recorded expression.
func (fl *fileLower) platformSpan(e syntax.Expr) (span, bool) {
	start := fl.off(platformLeftmost(e))
	end, ok := platformEnd(fl.src, start)
	if !ok || end <= start {
		return span{}, false
	}
	return span{start: start, end: end}, true
}

// platformEnd scans one complete expression from off, tracking bracket
// depth, and stops before a depth-zero separator or closer, an
// automatic semicolon, or EOF.
func platformEnd(src []byte, off int) (int, bool) {
	end := -1
	depth := 0
	scanSegment(src, off, len(src), func(t tok) bool {
		if t.auto {
			return false
		}
		if t.comment {
			end = t.end
			return true
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			if depth == 0 {
				return false
			}
			depth--
		case token.COMMA, token.SEMICOLON:
			if depth == 0 {
				return false
			}
		}
		end = t.end
		return true
	})
	if end < 0 {
		return 0, false
	}
	return end, true
}
