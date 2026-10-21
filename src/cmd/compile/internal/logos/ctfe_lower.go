// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

// This file implements M5-S2 lowering (SPEC Sec 2.1, T-201): consts
// with evaluated values splice their initializers to literals,
// composite-valued consts become vars, and const func markers erase.
// It runs first in lowerFile, ahead of the other passes.

import (
	"cmd/compile/internal/logos/ctfe"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"slices"
)

// ctfeResult bundles one const rewrite.
type ctfeResult struct {
	obj  *types2.Const
	val  ctfe.Value
	init syntax.Expr
	lit  syntax.Expr
	text string
}

// lowerCTFE rewrites evaluated consts of one file.
func (fl *fileLower) lowerCTFE() {
	for i, decl := range fl.file.DeclList {
		if cfd, ok := decl.(*syntax.ConstFuncDecl); ok {
			fl.file.DeclList[i] = cfd.Func
			fl.add(fl.off(cfd.ConstPos), fl.off(cfd.Func.Pos()), "")
		}
	}
	fl.lowerCTFEFileList()
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		if b, ok := n.(*syntax.BlockStmt); ok {
			fl.lowerCTFEBlock(b)
		}
		return true
	})
}

// ctfeLookup returns the rewrite for one single-name const.
func (fl *fileLower) ctfeLookup(cd *syntax.ConstDecl) (ctfeResult, bool) {
	var zero ctfeResult
	if len(cd.NameList) != 1 {
		return zero, false
	}
	obj, ok := fl.pkg.info.Defs[cd.NameList[0]].(*types2.Const)
	if !ok {
		return zero, false
	}
	v, ok := fl.pkg.info.LogosCTFE[obj]
	if !ok {
		return zero, false
	}
	init := cd.Values
	if l, ok := init.(*syntax.ListExpr); ok {
		if len(l.ElemList) != 1 {
			return zero, false
		}
		init = l.ElemList[0]
	}
	lit, err := v.Expr(syntax.StartPos(init))
	if err != nil {
		return zero, false
	}
	return ctfeResult{obj: obj, val: v, init: init, lit: lit, text: renderExpr(lit)}, true
}

// lowerCTFEFileList rewrites consts in the file declaration list.
func (fl *fileLower) lowerCTFEFileList() {
	list := &fl.file.DeclList
	for i := 0; i < len(*list); i++ {
		cd, ok := (*list)[i].(*syntax.ConstDecl)
		if !ok {
			continue
		}
		r, ok := fl.ctfeLookup(cd)
		if !ok {
			continue
		}
		if g, ok := fl.pkg.constMaps[r.obj]; ok { // M5-S4 hook: const maps lower to tables and lookups.
			fl.lowerConstMapSpec(cd, g)
			fl.removeAt(list, i)
			i--
			continue
		}
		if !isCTFEComposite(r.val) {
			cd.Values = r.lit
			fl.add(fl.off(syntax.StartPos(r.init)), fl.endOff(r.init), r.text)
			continue
		}
		if cd.Group == nil {
			vd := &syntax.VarDecl{NameList: cd.NameList, Type: cd.Type, Values: r.lit}
			vd.SetPos(syntax.StartPos(cd))
			(*list)[i] = vd
			fl.markRodataVar(vd, r.val)
			if kw, ok := constKeywordStart(fl.src, fl.off(syntax.StartPos(cd))); ok {
				fl.add(kw, kw+5, "var")
			}
			fl.add(fl.off(syntax.StartPos(r.init)), fl.endOff(r.init), r.text)
			continue
		}
		i = fl.moveGroupedVar(cd, i, list, r, "")
	}
}

// lowerCTFEBlock rewrites consts in one block, inserting moved grouped
// vars as following statements.
func (fl *fileLower) lowerCTFEBlock(b *syntax.BlockStmt) {
	for i := 0; i < len(b.List); i++ {
		ds, ok := b.List[i].(*syntax.DeclStmt)
		if !ok {
			continue
		}
		for j := 0; j < len(ds.DeclList); j++ {
			cd, ok := ds.DeclList[j].(*syntax.ConstDecl)
			if !ok {
				continue
			}
			r, ok := fl.ctfeLookup(cd)
			if !ok {
				continue
			}
			if g, ok := fl.pkg.constMaps[r.obj]; ok { // M5-S4 hook: const maps lower to tables and lookups.
				fl.lowerConstMapSpec(cd, g)
				ds.DeclList = slices.Delete(ds.DeclList, j, j+1)
				j--
				continue
			}
			if !isCTFEComposite(r.val) {
				cd.Values = r.lit
				fl.add(fl.off(syntax.StartPos(r.init)), fl.endOff(r.init), r.text)
				continue
			}
			if cd.Group == nil {
				vd := &syntax.VarDecl{NameList: cd.NameList, Type: cd.Type, Values: r.lit}
				vd.SetPos(syntax.StartPos(cd))
				ds.DeclList[j] = vd
				if kw, ok := constKeywordStart(fl.src, fl.off(syntax.StartPos(cd))); ok {
					fl.add(kw, kw+5, "var")
				}
				fl.add(fl.off(syntax.StartPos(r.init)), fl.endOff(r.init), r.text)
				continue
			}
			j = fl.moveGroupedBlockVar(cd, j, ds, b, i, r)
		}
		if len(ds.DeclList) == 0 { // M5-S4: a statement emptied by const map removal leaves the block.
			b.List = slices.Delete(b.List, i, i+1)
			i--
		}
	}
}

// moveGroupedVar moves a grouped file-level composite const after its
// group and deletes the spec. Init order stays dependency-ordered. It
// returns the updated list index.
func (fl *fileLower) moveGroupedVar(cd *syntax.ConstDecl, i int, list *[]syntax.Decl, r ctfeResult, indent string) int {
	_, end := groupRun(list, i, cd.Group)
	fl.deleteSpec(cd)
	fl.removeAt(list, i)
	vd := &syntax.VarDecl{NameList: cd.NameList, Type: cd.Type, Values: r.lit}
	vd.SetPos(syntax.StartPos(cd))
	fl.markRodataMoved(vd, r.val)
	at := end - 1
	*list = append(*list, nil)
	copy((*list)[at+1:], (*list)[at:])
	(*list)[at] = vd
	fl.insertVarText(cd, r, indent, func() (int, bool) {
		if at+1 < len(*list) {
			return fl.lineStart((*list)[at+1]), true
		}
		return fl.orig, false
	})
	return i - 1
}

// moveGroupedBlockVar moves a grouped block-level composite const after
// its DeclStmt and deletes the spec. It returns the updated decl index.
func (fl *fileLower) moveGroupedBlockVar(cd *syntax.ConstDecl, j int, ds *syntax.DeclStmt, b *syntax.BlockStmt, i int, r ctfeResult) int {
	indent := fl.stmtIndentOf(cd)
	fl.deleteSpec(cd)
	ds.DeclList = append(ds.DeclList[:j], ds.DeclList[j+1:]...)
	vd := &syntax.VarDecl{NameList: cd.NameList, Type: cd.Type, Values: r.lit}
	vd.SetPos(syntax.StartPos(cd))
	nds := &syntax.DeclStmt{DeclList: []syntax.Decl{vd}}
	nds.SetPos(syntax.StartPos(cd))
	b.List = append(b.List, nil)
	copy(b.List[i+2:], b.List[i+1:])
	b.List[i+1] = nds
	if i+2 < len(b.List) {
		fl.add(fl.lineStartOf(b.List[i+2]), fl.lineStartOf(b.List[i+2]), indent+"var "+declNames(cd)+" = "+r.text+"\n")
	} else {
		fl.add(fl.off(b.Rbrace), fl.off(b.Rbrace), "\n"+indent+"var "+declNames(cd)+" = "+r.text+"\n")
	}
	return j - 1
}

// insertVarText inserts a var line before the next sibling's line or at
// end of file.
func (fl *fileLower) insertVarText(cd *syntax.ConstDecl, r ctfeResult, indent string, next func() (int, bool)) {
	line := indent + "var " + declNames(cd) + " = " + r.text + "\n"
	if rodataStaticValue(r.val) {
		line = rodataMarker + "\n" + line
	}
	off, ok := next()
	if ok {
		fl.add(off, off, line)
		return
	}
	fl.add(off, off, "\n"+line)
}

// lineStart returns the start offset of the line containing the
// declaration's first line.
func (fl *fileLower) lineStart(d syntax.Decl) int {
	off := fl.off(syntax.StartPos(d))
	for off > 0 && fl.src[off-1] != '\n' {
		off--
	}
	return off
}

// lineStartOf returns the start offset of a statement's line.
func (fl *fileLower) lineStartOf(s syntax.Stmt) int {
	off := fl.off(syntax.StartPos(s))
	for off > 0 && fl.src[off-1] != '\n' {
		off--
	}
	return off
}

// groupRun returns the [start, end) run sharing group g around i.
func groupRun(list *[]syntax.Decl, i int, g *syntax.Group) (int, int) {
	start, end := i, i+1
	for start > 0 {
		prev, ok := (*list)[start-1].(*syntax.ConstDecl)
		if !ok || prev.Group != g {
			break
		}
		start--
	}
	for end < len(*list) {
		next, ok := (*list)[end].(*syntax.ConstDecl)
		if !ok || next.Group != g {
			break
		}
		end++
	}
	return start, end
}

// removeAt deletes index i from a decl list.
func (fl *fileLower) removeAt(list *[]syntax.Decl, i int) {
	*list = append((*list)[:i], (*list)[i+1:]...)
}

// deleteSpec deletes one spec's source text, swallowing one separator
// unless a comment or paren intervenes (a lone ; is invalid Go).
func (fl *fileLower) deleteSpec(cd *syntax.ConstDecl) {
	start := fl.off(syntax.StartPos(cd))
	end := fl.endOff(cd.Values)
	fl.add(start, swallowSep(fl.src, end), "")
}

// endOff converts an expression end to an exclusive offset. EndPos
// stops at a closing bracket for calls, indexes, slices, and composite
// literals, so one closer byte joins the span when present.
func (fl *fileLower) endOff(e syntax.Expr) int {
	end := fl.off(syntax.EndPos(e))
	if end < len(fl.src) && (fl.src[end] == ')' || fl.src[end] == ']' || fl.src[end] == '}') {
		end++
	}
	return end
}

// constKeywordStart scans back from a spec name over spaces, tabs, and
// comments for the `const` keyword. Grouped specs fail the scan (a
// paren intervenes) and take the move path instead.
func constKeywordStart(src []byte, nameOff int) (int, bool) {
	off := nameOff
	for off > 0 && (src[off-1] == ' ' || src[off-1] == '\t') {
		off--
	}
	for off >= 2 && src[off-1] == '/' && src[off-2] == '*' {
		end := off - 2
		start := end
		for start >= 2 && (src[start-1] != '/' || src[start-2] != '*') {
			start--
		}
		if start < 2 {
			return 0, false
		}
		off = start - 2
		for off > 0 && (src[off-1] == ' ' || src[off-1] == '\t') {
			off--
		}
	}
	if off >= 5 && string(src[off-5:off]) == "const" {
		return off - 5, true
	}
	return 0, false
}

// declNames joins declaration names.
func declNames(cd *syntax.ConstDecl) string {
	names := ""
	for j, n := range cd.NameList {
		if j > 0 {
			names += ", "
		}
		names += n.Value
	}
	return names
}

// stmtIndentOf reads the leading whitespace of a decl's line.
func (fl *fileLower) stmtIndentOf(cd *syntax.ConstDecl) string {
	off := fl.off(syntax.StartPos(cd))
	line := off
	for line > 0 && fl.src[line-1] != '\n' {
		line--
	}
	indent := ""
	for _, c := range fl.src[line:off] {
		if c != ' ' && c != '\t' {
			break
		}
		indent += string(c)
	}
	return indent
}

// isCTFEComposite reports composite values.
func isCTFEComposite(v ctfe.Value) bool {
	t := v.Type()
	if t == nil {
		return false
	}
	switch t.Kind {
	case ctfe.KSlice, ctfe.KArray, ctfe.KMap, ctfe.KStruct:
		return true
	}
	return false
}

// swallowSep extends an offset past trailing spaces, one semicolon,
// and one newline, stopping at comments and parens.
func swallowSep(src []byte, off int) int {
	for off < len(src) && (src[off] == ' ' || src[off] == '\t') {
		off++
	}
	if off < len(src) && src[off] == ';' {
		off++
		for off < len(src) && (src[off] == ' ' || src[off] == '\t') {
			off++
		}
	}
	if off < len(src) && src[off] == '\n' {
		off++
	}
	return off
}
