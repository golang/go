// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"go/token"
	"slices"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func (fl *fileLower) lowerViewAs() {
	var sels []*syntax.SelectorExpr
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		sel, ok := n.(*syntax.SelectorExpr)
		if ok && logosComptimeViewAs(fl.pkg.info, sel) {
			sels = append(sels, sel)
		}
		return true
	})
	if len(sels) == 0 {
		return
	}
	done := make(map[*syntax.Name]bool, len(sels))
	for _, sel := range sels {
		qn := sel.X.(*syntax.Name)
		start := fl.off(qn.Pos())
		fl.add(start, start+len(qn.Value), rtAlias)
		qn.Value = rtAlias
		done[qn] = true
	}
	fl.needRT = true
	fl.dropComptimeImport(done)
}

func logosComptimeViewAs(info *types2.Info, sel *syntax.SelectorExpr) bool {
	qn, ok := sel.X.(*syntax.Name)
	if !ok || sel.Sel.Value != "ViewAs" {
		return false
	}
	pn, ok := info.Uses[qn].(*types2.PkgName)
	if !ok || pn.Imported().Path() != "comptime" {
		return false
	}
	fn, ok := info.Uses[sel.Sel].(*types2.Func)
	return ok && fn.Name() == "ViewAs" && fn.Pkg() != nil && fn.Pkg().Path() == "comptime"
}

func (fl *fileLower) dropComptimeImport(done map[*syntax.Name]bool) {
	keep := false
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		sel, ok := n.(*syntax.SelectorExpr)
		if !ok {
			return true
		}
		qn, ok := sel.X.(*syntax.Name)
		if !ok || done[qn] {
			return true
		}
		if pn, ok := fl.pkg.info.Uses[qn].(*types2.PkgName); ok && pn.Imported().Path() == "comptime" {
			keep = true
			return false
		}
		return true
	})
	if keep {
		return
	}
	var at []int
	for i, decl := range fl.file.DeclList {
		imp, ok := decl.(*syntax.ImportDecl)
		if !ok || imp.Path == nil || !isComptimePath(imp.Path.Value) {
			continue
		}
		at = append(at, i)
		fl.removeImportSpec(imp)
	}
	for i := len(at) - 1; i >= 0; i-- {
		fl.file.DeclList = slices.Delete(fl.file.DeclList, at[i], at[i]+1)
	}
}

func (fl *fileLower) removeImportSpec(imp *syntax.ImportDecl) {
	start := fl.off(syntax.StartPos(imp))
	end := fl.off(syntax.EndPos(imp))
	if imp.Group == nil {
		if kw, ok := fl.keywords.Start(fl.src, token.IMPORT, start); ok {
			start = kw
		}
	}
	scanSegment(fl.src, end, len(fl.src), func(t tok) bool {
		if t.kind == token.SEMICOLON && !t.auto {
			end = t.end
		}
		return t.comment
	})
	fl.add(start, end, keepComments(fl.src, start, end, ""))
}

func isComptimePath(v string) bool {
	return v == `"comptime"` || v == "`comptime`"
}
