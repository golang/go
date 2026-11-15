// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func (fl *fileLower) lowerConstSlices() {
	cov := fl.pkg.info.LogosConstCov
	if len(cov) == 0 {
		return
	}
	var calls []*syntax.CallExpr
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		call, ok := n.(*syntax.CallExpr)
		if !ok {
			return true
		}
		for _, a := range call.ArgList {
			if _, ok := cov[a]; ok {
				calls = append(calls, call)
				break
			}
		}
		return true
	})
	if len(calls) == 0 {
		return
	}
	for _, call := range calls {
		for i, a := range call.ArgList {
			iface, ok := cov[a]
			if !ok {
				continue
			}
			call.ArgList[i] = fl.covSliceWrap(a, iface)
		}
	}
	fl.needRT = true
	fl.flushTypeImports()
}

func (fl *fileLower) covSliceWrap(a syntax.Expr, iface types2.Type) syntax.Expr {
	pos := leftmostOf(a)
	if c, ok := syntax.Unparen(a).(*syntax.CompositeLit); ok && c.Type != nil {
		pos = syntax.StartPos(c)
	}
	text, ok := fl.spellType(iface)
	if !ok {
		base.Fatalf("logos: const slice element %s of %s cannot be spelled in this file", types2.TypeString(iface, nil), fl.baseName)
	}
	typ, ok := typeExprAt(text, pos)
	if !ok {
		base.Fatalf("logos: const slice element %q of %s is not a type", text, fl.baseName)
	}
	fun := at(&syntax.IndexExpr{X: rtSel(pos, "CovSlice"), Index: typ}, pos)
	start := fl.off(pos)
	end := fl.exprEndOf(a)
	fl.add(start, start, fmt.Sprintf("%s.CovSlice[%s](", rtAlias, text))
	fl.add(end, end, ")")
	return callOf(pos, fun, []syntax.Expr{a})
}
