// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/logos/ctfe"
	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"fmt"
	"go/token"
	"slices"
)

type ctfeResult struct {
	obj      *types2.Const
	val      ctfe.Value
	init     syntax.Expr
	lit      syntax.Expr
	text     string
	keepCall bool
}

type groupedMoveState struct {
	fileTails  map[*syntax.Group]syntax.Decl
	blockTails map[*syntax.DeclStmt]syntax.Stmt
	inserted   map[syntax.Node]bool
}

func (fl *fileLower) lowerCTFE() {
	for i, decl := range fl.file.DeclList {
		if cfd, ok := decl.(*syntax.ConstFuncDecl); ok {
			fl.file.DeclList[i] = cfd.Func
			fl.add(fl.off(cfd.ConstPos), fl.off(cfd.Func.Pos()), "")
		}
	}
	gst := &groupedMoveState{
		fileTails:  map[*syntax.Group]syntax.Decl{},
		blockTails: map[*syntax.DeclStmt]syntax.Stmt{},
		inserted:   map[syntax.Node]bool{},
	}
	fl.lowerCTFEFileList(gst)
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		if b, ok := n.(*syntax.BlockStmt); ok {
			fl.lowerCTFEBlock(b, gst)
		}
		return true
	})
}

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
	if m, ok := fl.materializeImportedConst(cd, obj, v, init); ok {
		return m, true
	}
	lit, err := v.Expr(syntax.StartPos(init))
	if err != nil {
		return fl.ctfeUnspelled(obj, v, init)
	}
	return ctfeResult{obj: obj, val: v, init: init, lit: lit, text: renderExpr(lit)}, true
}

func (fl *fileLower) ctfeUnspelled(obj *types2.Const, v ctfe.Value, init syntax.Expr) (ctfeResult, bool) {
	if _, ok := fl.pkg.constMaps[obj]; ok || fl.pkg.composites[obj] != nil {
		return ctfeResult{obj: obj, val: v, init: init}, true
	}
	if !v.IsComposite() {
		return ctfeResult{}, false
	}
	return fl.materializeComposite(obj, v, init, syntax.StartPos(init))
}

func (fl *fileLower) lowerCTFEFileList(gst *groupedMoveState) {
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
		if g, ok := fl.pkg.constMaps[r.obj]; ok {
			fl.lowerConstMapSpec(cd, g)
			*list = slices.Delete(*list, i, i+1)
			i--
			continue
		}
		if info := fl.pkg.composites[r.obj]; info != nil {
			fl.lowerCompositeSpec(cd, info)
			*list = slices.Delete(*list, i, i+1)
			i--
			continue
		}
		fl.recordCTFECompositeOrigin(r)
		if !r.val.IsComposite() && !r.keepCall {
			cd.Values = r.lit
			fl.add(fl.off(syntax.StartPos(r.init)), fl.endOff(r.init), r.text)
			continue
		}
		if cd.Group == nil {
			vd := &syntax.VarDecl{NameList: cd.NameList, Type: cd.Type, Values: r.lit}
			vd.SetPos(syntax.StartPos(cd))
			(*list)[i] = vd
			if r.keepCall {
				fl.markRodataMoved(vd, r.val)
			} else {
				fl.markRodataVar(vd, r.val)
			}
			if kw, ok := fl.keywords.Start(fl.src, token.CONST, fl.off(syntax.StartPos(cd))); ok {
				fl.add(kw, kw+5, "var")
			}
			if !r.keepCall {
				fl.add(fl.off(syntax.StartPos(r.init)), fl.endOff(r.init), r.text)
			}
			continue
		}
		i = fl.moveGroupedVar(cd, i, list, r, "", gst)
	}
}

func (fl *fileLower) recordCTFECompositeOrigin(r ctfeResult) {
	if r.obj == nil || !r.obj.Exported() || !r.val.IsComposite() || fl.pkg.tpkg.Scope().Lookup(r.obj.Name()) != r.obj {
		return
	}
	f := facts.Current()
	if f == nil || slices.Contains(f.ConstComps, r.obj.Name()) {
		return
	}
	f.ConstComps = append(f.ConstComps, r.obj.Name())
}

func (fl *fileLower) lowerCTFEBlock(b *syntax.BlockStmt, gst *groupedMoveState) {
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
			if g, ok := fl.pkg.constMaps[r.obj]; ok {
				fl.lowerConstMapSpec(cd, g)
				ds.DeclList = slices.Delete(ds.DeclList, j, j+1)
				j--
				continue
			}
			if info := fl.pkg.composites[r.obj]; info != nil {
				fl.lowerCompositeSpec(cd, info)
				ds.DeclList = slices.Delete(ds.DeclList, j, j+1)
				j--
				continue
			}
			if !r.val.IsComposite() && !r.keepCall {
				cd.Values = r.lit
				fl.add(fl.off(syntax.StartPos(r.init)), fl.endOff(r.init), r.text)
				continue
			}
			if cd.Group == nil {
				r = fl.hoistBlockConstData(cd, r)
				vd := &syntax.VarDecl{NameList: cd.NameList, Type: cd.Type, Values: r.lit}
				vd.SetPos(syntax.StartPos(cd))
				ds.DeclList[j] = vd
				if kw, ok := fl.keywords.Start(fl.src, token.CONST, fl.off(syntax.StartPos(cd))); ok {
					fl.add(kw, kw+5, "var")
				}
				if !r.keepCall {
					fl.add(fl.off(syntax.StartPos(r.init)), fl.endOff(r.init), r.text)
				}
				continue
			}
			j = fl.moveGroupedBlockVar(cd, j, blockMoveSite{ds: ds, b: b, i: i}, fl.hoistBlockConstData(cd, r), gst)
		}
		if len(ds.DeclList) == 0 {
			b.List = slices.Delete(b.List, i, i+1)
			i--
		}
	}
}

func (fl *fileLower) hoistBlockConstData(cd *syntax.ConstDecl, r ctfeResult) ctfeResult {
	if r.keepCall || !r.val.IsComposite() || !rodataStaticValue(r.val) || !ctfeBearsSlice(r.val.Type(), 0) {
		return r
	}
	spell, ok := fl.spellLowered(r.obj.Type())
	if !ok {
		return r
	}
	pos := syntax.StartPos(cd)
	name := fmt.Sprintf("_logos_cl%d_%s", fl.pkg.localConstSeq, r.obj.Name())
	fl.pkg.localConstSeq++
	fl.addGeneratedDecls(pos, fmt.Sprintf("%s\nvar %s %s = %s\n", rodataMarker, name, spell, r.text))
	r.text = name
	r.lit = nameOf(pos, name)
	return r
}

func ctfeBearsSlice(t *ctfe.Type, depth int) bool {
	if t == nil || depth > 64 {
		return false
	}
	switch t.Kind {
	case ctfe.KSlice:
		return true
	case ctfe.KArray:
		return ctfeBearsSlice(t.Elem, depth+1)
	case ctfe.KStruct:
		for _, f := range t.Fields {
			if ctfeBearsSlice(f.Type, depth+1) {
				return true
			}
		}
	}
	return false
}

func (fl *fileLower) moveGroupedVar(cd *syntax.ConstDecl, i int, list *[]syntax.Decl, r ctfeResult, indent string, gst *groupedMoveState) int {
	_, end := groupRun(list, i, cd.Group)
	fl.deleteSpec(cd)
	*list = slices.Delete(*list, i, i+1)
	vd := &syntax.VarDecl{NameList: cd.NameList, Type: cd.Type, Values: r.lit}
	vd.SetPos(syntax.StartPos(cd))
	fl.markRodataMoved(vd, r.val)
	at := end - 1
	if tail, ok := gst.fileTails[cd.Group]; ok {
		at = len(*list)
		for k, d := range *list {
			if d == tail {
				at = k + 1
				break
			}
		}
	}
	gst.fileTails[cd.Group] = vd
	gst.inserted[vd] = true
	*list = append(*list, nil)
	copy((*list)[at+1:], (*list)[at:])
	(*list)[at] = vd
	off := groupEnd(fl.src, fl.off(syntax.StartPos(cd)))
	for off < len(fl.src) && (fl.src[off] == ' ' || fl.src[off] == '\t') {
		off++
	}
	if off < len(fl.src) && (fl.src[off] == ';' || fl.src[off] == '\n') {
		off++
	}
	line := indent + "var " + declNames(cd) + " = " + r.text + "\n"
	if rodataStaticValue(r.val) {
		line = rodataMarker + "\n" + line
	}
	if off > 0 && fl.src[off-1] != '\n' {
		line = "\n" + line
	}
	fl.add(off, off, line)
	return i - 1
}

type blockMoveSite struct {
	ds *syntax.DeclStmt
	b  *syntax.BlockStmt
	i  int
}

func (fl *fileLower) moveGroupedBlockVar(cd *syntax.ConstDecl, j int, at blockMoveSite, r ctfeResult, gst *groupedMoveState) int {
	indent := fl.stmtIndentOf(cd)
	fl.deleteSpec(cd)
	at.ds.DeclList = append(at.ds.DeclList[:j], at.ds.DeclList[j+1:]...)
	vd := &syntax.VarDecl{NameList: cd.NameList, Type: cd.Type, Values: r.lit}
	vd.SetPos(syntax.StartPos(cd))
	nds := &syntax.DeclStmt{DeclList: []syntax.Decl{vd}}
	nds.SetPos(syntax.StartPos(cd))
	pos := at.i + 1
	if tail, ok := gst.blockTails[at.ds]; ok {
		pos = len(at.b.List)
		for k, s := range at.b.List {
			if s == tail {
				pos = k + 1
				break
			}
		}
	}
	gst.blockTails[at.ds] = nds
	gst.inserted[nds] = true
	at.b.List = append(at.b.List, nil)
	copy(at.b.List[pos+1:], at.b.List[pos:])
	at.b.List[pos] = nds
	if off, ok := fl.blockVarOffset(at.b, pos, gst); ok {
		fl.add(off, off, indent+"var "+declNames(cd)+" = "+r.text+"\n")
	} else {
		fl.add(fl.off(at.b.Rbrace), fl.off(at.b.Rbrace), "\n"+indent+"var "+declNames(cd)+" = "+r.text+"\n")
	}
	return j - 1
}

func (fl *fileLower) blockVarOffset(b *syntax.BlockStmt, at int, gst *groupedMoveState) (int, bool) {
	for k := at + 1; k < len(b.List); k++ {
		if gst.inserted[b.List[k]] {
			continue
		}
		return fl.lineStart(b.List[k]), true
	}
	return 0, false
}

func (fl *fileLower) lineStart(n syntax.Node) int {
	off := fl.off(syntax.StartPos(n))
	for off > 0 && fl.src[off-1] != '\n' {
		off--
	}
	return off
}

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

func (fl *fileLower) deleteSpec(cd *syntax.ConstDecl) {
	start := fl.off(syntax.StartPos(cd))
	end := fl.endOff(cd.Values)
	fl.add(start, swallowSep(fl.src, end), "")
}

func (fl *fileLower) endOff(e syntax.Expr) int {
	switch n := e.(type) {
	case *syntax.CallExpr, *syntax.IndexExpr, *syntax.SliceExpr, *syntax.ParenExpr, *syntax.AssertExpr:
		split, ok := splitCall(fl.src, fl.off(n.Pos()))
		if !ok {
			base.Fatalf("logos: expression at %s has no closing bracket", n.Pos().String())
		}
		return split.end
	case *syntax.CompositeLit:
		return fl.off(n.Rbrace) + 1
	case *syntax.FuncLit:
		return fl.off(n.Body.Rbrace) + 1
	case *syntax.Operation:
		if n.Y != nil {
			return fl.endOff(n.Y)
		}
		return fl.endOff(n.X)
	case *syntax.ListExpr:
		return fl.endOff(n.ElemList[len(n.ElemList)-1])
	}
	return fl.off(syntax.EndPos(e))
}

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

func swallowSep(src []byte, off int) int {
	for off < len(src) && (src[off] == ' ' || src[off] == '\t') {
		off++
	}
	semi := -1
	scanSegment(src, off, len(src), func(t tok) bool {
		if t.kind == token.SEMICOLON && !t.auto {
			semi = t.end
		}
		return t.comment
	})
	if semi >= 0 {
		off = semi
		for off < len(src) && (src[off] == ' ' || src[off] == '\t') {
			off++
		}
	}
	if off < len(src) && src[off] == '\n' {
		off++
	}
	return off
}
