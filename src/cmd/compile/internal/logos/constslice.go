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
	fl.flushTypeImports()
}

func (fl *fileLower) covSliceWrap(a syntax.Expr, iface types2.Type) syntax.Expr {
	pos := leftmostOf(a)
	if c, ok := syntax.Unparen(a).(*syntax.CompositeLit); ok && c.Type != nil {
		pos = syntax.StartPos(c)
	}
	if call := fl.covTypedWrap(a, iface, pos); call != nil {
		return call
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
	fl.needRT = true
	return callOf(pos, fun, []syntax.Expr{a})
}

func (fl *fileLower) covTypedWrap(a syntax.Expr, iface types2.Type, pos syntax.Pos) syntax.Expr {
	argType := fl.pkg.info.TypeOf(a)
	if argType == nil {
		return nil
	}
	slc, ok := types2.Unalias(argType).Underlying().(*types2.Slice)
	if !ok {
		return nil
	}
	elem := types2.Unalias(slc.Elem())
	if _, ok := elem.(*types2.TypeParam); ok {
		return nil
	}
	itf, ok := types2.Unalias(iface).Underlying().(*types2.Interface)
	if !ok || !types2.Implements(elem, itf) {
		return nil
	}
	spellE, ok := fl.spellType(elem)
	if !ok {
		return nil
	}
	spellI, ok := fl.spellType(iface)
	if !ok {
		return nil
	}
	text := "func(_logos_cs_s []" + spellE + ") []" + spellI + " {\n" + covTypedBody(spellI) + "}"
	fun := parseGeneratedDecls("var _ = "+text, pos)[0].(*syntax.VarDecl).Values
	start := fl.off(pos)
	end := fl.exprEndOf(a)
	fl.add(start, start, text+"(")
	fl.add(end, end, ")")
	return callOf(pos, fun, []syntax.Expr{a})
}

func covTypedBody(spellI string) string {
	return "if _logos_cs_s == nil {\nreturn nil\n}\n" +
		"_logos_cs_r := make([]" + spellI + ", len(_logos_cs_s))\n" +
		"for _logos_cs_i := range _logos_cs_s {\n" +
		"_logos_cs_r[_logos_cs_i] = " + spellI + "(_logos_cs_s[_logos_cs_i])\n" +
		"}\nreturn _logos_cs_r\n"
}
