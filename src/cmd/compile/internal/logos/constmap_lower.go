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
	"strconv"
	"strings"
)

const constMapSwitchMax = 8

const (
	constMapHashRecv = "_logos_v"
	constMapHashAcc  = "_logos_h"
)

type constMapGen struct {
	name        string
	obj         *types2.Const
	val         ctfe.Value
	init        syntax.Expr
	nativeDecls []syntax.Decl
	nkeys       int
	bearing     bool
	lead        syntax.Expr
}

func (g *constMapGen) lookupName() string { return g.name + "_lookup" }
func (g *constMapGen) sortedName() string { return g.name + "_sorted" }
func (g *constMapGen) mapName() string    { return g.name + "_map" }
func (g *constMapGen) tabName() string    { return g.name + "_tab" }
func (g *constMapGen) matName() string    { return g.name + "_mat" }
func (g *constMapGen) buildName() string  { return g.name + "_build" }
func (g *constMapGen) valName() string    { return g.name + "_val" }
func (g *constMapGen) allName() string    { return g.name + "_all" }

func constMapBearing(vals []ctfe.Value) bool {
	for _, v := range vals {
		if constMapValueBearing(v.Type()) {
			return true
		}
	}
	return false
}

func constMapValueBearing(t *ctfe.Type) bool {
	if t == nil {
		return true
	}
	switch t.Kind {
	case ctfe.KSlice, ctfe.KMap:
		return true
	case ctfe.KArray:
		return constMapValueBearing(t.Elem)
	case ctfe.KStruct:
		for _, f := range t.Fields {
			if constMapValueBearing(f.Type) {
				return true
			}
		}
	}
	return false
}

type constMapEmission struct {
	name        string
	kT          string
	vT          string
	mapT        string
	keyTexts    []string
	valTexts    []string
	builderBody string
	hashExpr    string
	order       []int
	layout      *constMapLayout
	isSwitch    bool
	bearing     bool
}

func emitConstMapDecls(e constMapEmission) string {
	var b strings.Builder
	name := e.name
	if !e.bearing {
		fmt.Fprintf(&b, "%s\nvar %s_sorted = []%s.ConstEntry[%s, %s]{", rodataMarker, name, rtAlias, e.kT, e.vT)
		for j, i := range e.order {
			if j > 0 {
				b.WriteString(", ")
			}
			fmt.Fprintf(&b, "{Key: %s, Val: %s}", e.keyTexts[i], e.valTexts[i])
		}
		b.WriteString("}\n")
	} else {
		fmt.Fprintf(&b, "%s\nvar %s_sorted = []%s.ConstEntry[%s, uint32]{", rodataMarker, name, rtAlias, e.kT)
		for j, i := range e.order {
			if j > 0 {
				b.WriteString(", ")
			}
			fmt.Fprintf(&b, "{Key: %s, Val: %d}", e.keyTexts[i], i)
		}
		b.WriteString("}\n")
	}
	fmt.Fprintf(&b, "var %s_mat %s.ConstMapMaterialization[%s, %s, %s]\n", name, rtAlias, e.mapT, e.kT, e.vT)
	fmt.Fprintf(&b, "func %s_build() %s { return %s }\n", name, e.mapT, e.builderBody)
	fmt.Fprintf(&b, "func %s_map() %s { return %s_mat.Get(%s_build) }\n", name, e.mapT, name, name)
	if e.bearing {
		fmt.Fprintf(&b, "func %s_val(_logos_idx uint32) %s {\n\tswitch _logos_idx {\n", name, e.vT)
		for i := range e.valTexts {
			fmt.Fprintf(&b, "\tcase %d:\n\t\treturn %s\n", i, e.valTexts[i])
		}
		fmt.Fprintf(&b, "\t}\n\tvar _logos_zero %s\n\treturn _logos_zero\n}\n", e.vT)
		fmt.Fprintf(&b, "func %s_all(_logos_yield func(%s, %s) bool) {\n\tfor _, _logos_e := range %s_sorted {\n\t\tif !_logos_yield(_logos_e.Key, %s_val(_logos_e.Val)) {\n\t\t\treturn\n\t\t}\n\t}\n}\n", name, e.kT, e.vT, name, name)
	}
	if e.isSwitch {
		if !e.bearing {
			fmt.Fprintf(&b, "func %s_lookup(_logos_k %s) (%s, bool) {\n\tswitch _logos_k {\n", name, e.kT, e.vT)
			for i := range e.keyTexts {
				fmt.Fprintf(&b, "\tcase %s:\n\t\treturn %s, true\n", e.keyTexts[i], e.valTexts[i])
			}
			fmt.Fprintf(&b, "\t}\n\tvar _logos_zero %s\n\treturn _logos_zero, false\n}\n", e.vT)
			return b.String()
		}
		fmt.Fprintf(&b, "func %s_lookup(_logos_k %s) (%s, bool) {\n\tswitch _logos_k {\n", name, e.kT, e.vT)
		for i := range e.keyTexts {
			fmt.Fprintf(&b, "\tcase %s:\n\t\treturn %s_val(%d), true\n", e.keyTexts[i], name, i)
		}
		fmt.Fprintf(&b, "\t}\n\tvar _logos_zero %s\n\treturn _logos_zero, false\n}\n", e.vT)
		return b.String()
	}
	if !e.bearing {
		fmt.Fprintf(&b, "%s\nvar %s = %s.ConstTable[%s, %s]{Salt: %d, Buckets: []%s.ConstBucket{", rodataMarker, name+"_tab", rtAlias, e.kT, e.vT, e.layout.salt, rtAlias)
		for i, bk := range e.layout.buckets {
			if i > 0 {
				b.WriteString(", ")
			}
			fmt.Fprintf(&b, "{Off: %d, Size: %d, Seed: %d}", bk.off, bk.size, bk.seed)
		}
		fmt.Fprintf(&b, "}, Slots: []%s.ConstSlot[%s, %s]{", rtAlias, e.kT, e.vT)
		for s, ki := range e.layout.slots {
			if s > 0 {
				b.WriteString(", ")
			}
			if ki < 0 {
				b.WriteString("{}")
				continue
			}
			fmt.Fprintf(&b, "{Key: %s, Val: %s, Occ: true}", e.keyTexts[ki], e.valTexts[ki])
		}
		b.WriteString("}}\n")
		fmt.Fprintf(&b, "func %s_lookup(_logos_k %s) (%s, bool) {\n\treturn %s.Get(%s, _logos_k)\n}\n", name, e.kT, e.vT, name+"_tab", e.hashExpr)
		return b.String()
	}
	fmt.Fprintf(&b, "%s\nvar %s = %s.ConstTable[%s, uint32]{Salt: %d, Buckets: []%s.ConstBucket{", rodataMarker, name+"_tab", rtAlias, e.kT, e.layout.salt, rtAlias)
	for i, bk := range e.layout.buckets {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "{Off: %d, Size: %d, Seed: %d}", bk.off, bk.size, bk.seed)
	}
	fmt.Fprintf(&b, "}, Slots: []%s.ConstSlot[%s, uint32]{", rtAlias, e.kT)
	for s, ki := range e.layout.slots {
		if s > 0 {
			b.WriteString(", ")
		}
		if ki < 0 {
			b.WriteString("{}")
			continue
		}
		fmt.Fprintf(&b, "{Key: %s, Val: %d, Occ: true}", e.keyTexts[ki], ki)
	}
	b.WriteString("}}\n")
	fmt.Fprintf(&b, "func %s_lookup(_logos_k %s) (%s, bool) {\n\tif _logos_idx, _logos_ok := %s.Get(%s, _logos_k); _logos_ok {\n\t\treturn %s_val(_logos_idx), true\n\t}\n\tvar _logos_zero %s\n\treturn _logos_zero, false\n}\n", name, e.kT, e.vT, name+"_tab", e.hashExpr, name, e.vT)
	return b.String()
}

func collectConstMaps(files []*syntax.File, info *types2.Info) map[*types2.Const]*constMapGen {
	clearConstMat()
	out := map[*types2.Const]*constMapGen{}
	seq := 0
	for _, file := range files {
		syntax.Inspect(file, func(n syntax.Node) bool {
			cd, ok := n.(*syntax.ConstDecl)
			if !ok || len(cd.NameList) != 1 {
				return true
			}
			obj, ok := info.Defs[cd.NameList[0]].(*types2.Const)
			if !ok {
				return true
			}
			v, ok := info.LogosCTFE[obj]
			if !ok || v.Type() == nil || v.Type().Kind != ctfe.KMap {
				return true
			}
			init := cd.Values
			g := &constMapGen{
				name: fmt.Sprintf("_logos_cm%d_%s", seq, cd.NameList[0].Value),
				obj:  obj,
				val:  v,
				init: init,
			}
			keys, vals := v.MapEntries()
			g.nkeys = len(keys)
			g.bearing = constMapBearing(vals)
			out[obj] = g
			seq++
			return true
		})
	}
	return out
}

func (fl *fileLower) constMapReadUse(x syntax.Expr) (*constMapGen, bool) {
	if g, ok := fl.constMapUse(x); ok {
		return g, true
	}
	return fl.compositeFamilyFor(x)
}

func (fl *fileLower) constMapUse(x syntax.Expr) (*constMapGen, bool) {
	switch e := syntax.Unparen(x).(type) {
	case *syntax.Name:
		if obj, ok := fl.pkg.info.Uses[e].(*types2.Const); ok {
			if g, ok := fl.pkg.constMaps[obj]; ok {
				return g, true
			}
		}
		if obj, ok := fl.pkg.info.Uses[e].(*types2.Var); ok {
			return fl.importedConstMap(obj, e.Pos())
		}
	case *syntax.SelectorExpr:
		if e.Sel == nil {
			return nil, false
		}
		if obj, ok := fl.pkg.info.Uses[e.Sel].(*types2.Var); ok && !obj.IsField() {
			if g, ok := fl.importedConstMap(obj, e.Pos()); ok {
				return g, true
			}
		}
	}
	return fl.compositeHiddenFor(x)
}

func importedConstMapRecord(obj *types2.Var) (facts.ConstMapFact, bool) {
	var zero facts.ConstMapFact
	if obj.Pkg() == nil {
		return zero, false
	}
	if scope := obj.Pkg().Scope(); scope == nil || scope.Lookup(obj.Name()) != types2.Object(obj) {
		return zero, false
	}
	f, ok := facts.Lookup(obj.Pkg().Path())
	if !ok || f == nil {
		return zero, false
	}
	for i := range f.ConstMaps {
		rec := f.ConstMaps[i]
		if rec.Object != obj.Name() || rec.Path != "" {
			continue
		}
		if len(rec.Keys) != len(rec.Vals) || len(rec.Order) != len(rec.Keys) {
			return zero, false
		}
		return rec, true
	}
	return zero, false
}

func (fl *fileLower) importedConstMap(obj *types2.Var, pos syntax.Pos) (*constMapGen, bool) {
	view := constAsView(obj.Type())
	if view == nil || !view.ConstMap {
		return nil, false
	}
	if obj.Pkg() == nil || obj.Pkg() == fl.pkg.tpkg {
		return nil, false
	}
	if fl.pkg.importedMaps == nil {
		fl.pkg.importedMaps = map[*types2.Var]*constMapGen{}
	}
	if fl.pkg.importedDone == nil {
		fl.pkg.importedDone = map[string]bool{}
	}
	if g, ok := fl.pkg.importedMaps[obj]; ok {
		return g, true
	}
	rec, ok := importedConstMapRecord(obj)
	if !ok {
		return nil, false
	}
	mt, ok := constMapElem(obj.Type()).Underlying().(*types2.Map)
	if !ok {
		return nil, false
	}
	name := fmt.Sprintf("_logos_icm%d_%s", fl.pkg.importedSeq, obj.Name())
	fl.pkg.importedSeq++
	g, ok := fl.emitImportedConstMap(importedMapSpec{label: obj.Pkg().Path() + "." + obj.Name(), mapType: constMapElem(obj.Type()), mt: mt, rec: rec, name: name}, pos)
	if ok {
		fl.pkg.importedMaps[obj] = g
	}
	return g, ok
}

type importedMapSpec struct {
	label   string
	mapType types2.Type
	mt      *types2.Map
	rec     facts.ConstMapFact
	name    string
}

func (fl *fileLower) emitImportedConstMap(s importedMapSpec, pos syntax.Pos) (*constMapGen, bool) {
	rec, mt := s.rec, s.mt
	if why, ok := fl.unspellableImportedConstMap(s.label, s.mapType, mt); !ok {
		base.Errorf("%s: %s", pos.String(), why)
		base.ExitIfErrors()
		return nil, false
	}
	quals, ambiguous := fl.constMapImportQuals(mt)
	if ambiguous != "" {
		base.Errorf("%s: logos: imported const map %s uses %s for types from different packages", pos.String(), s.label, ambiguous)
		base.ExitIfErrors()
		return nil, false
	}
	keys, ok := fl.qualifyConstMapTexts(rec.Keys, quals, pos)
	if !ok {
		base.Errorf("%s: logos: imported const map %s has a key that cannot be spelled in this package", pos.String(), s.label)
		base.ExitIfErrors()
		return nil, false
	}
	vals, ok := fl.qualifyConstMapTexts(rec.Vals, quals, pos)
	if !ok {
		base.Errorf("%s: logos: imported const map %s has a value that cannot be spelled in this package", pos.String(), s.label)
		base.ExitIfErrors()
		return nil, false
	}
	kT := fl.mustSpell(mt.Key(), pos)
	vT := fl.mustSpell(mt.Elem(), pos)
	mapT := fl.mustSpell(s.mapType, pos)
	entries := make([]constMapKey, len(rec.Keys))
	for i := range rec.Keys {
		entries[i] = constMapKey{hash: rec.Hashes[i], canon: rec.Canons[i]}
	}
	pairs := make([]string, len(rec.Keys))
	for i := range rec.Keys {
		pairs[i] = keys[i] + ": " + vals[i]
	}
	builderBody := mapT + "{" + strings.Join(pairs, ", ") + "}"
	text := finalizeConstMapEmission(pos, constMapEmission{
		name:        s.name,
		kT:          kT,
		vT:          vT,
		mapT:        mapT,
		keyTexts:    keys,
		valTexts:    vals,
		builderBody: builderBody,
		hashExpr:    fl.constKeyHashText(mt.Key(), "_logos_k", pos),
		order:       rec.Order,
		bearing:     rec.Bearing,
	}, entries)
	g := &constMapGen{name: s.name, nkeys: len(rec.Keys), bearing: rec.Bearing}
	if !fl.pkg.importedDone[s.name] {
		fl.pkg.importedDone[s.name] = true
		decls := parseConstMapDecls(text, pos)
		fl.markConstMapTables(decls, g)
		fl.constDecls = append(fl.constDecls, decls...)
		fl.constText += "\n" + text
	}
	fl.needRT = true
	return g, true
}

func stripConstMapView(t types2.Type) types2.Type {
	for {
		cv, ok := t.(*types2.ConstView)
		if !ok {
			return t
		}
		t = cv.Elem
	}
}

func (fl *fileLower) unspellableImportedConstMap(label string, mapType types2.Type, mt *types2.Map) (string, bool) {
	what := "logos: imported const map " + label
	if _, ok := fl.spellLowered(mt.Key()); !ok {
		return what + " has a key type " + mt.Key().String() + " that cannot be named in this package", false
	}
	if _, ok := fl.spellLowered(mt.Elem()); !ok {
		return what + " has a value type " + mt.Elem().String() + " that cannot be named in this package", false
	}
	if _, ok := fl.spellLowered(mapType); !ok {
		return what + " has a map type that cannot be named in this package", false
	}
	if !matFieldsAssignable(stripConstMapView(mt.Key()), fl.pkg.tpkg) || !matFieldsAssignable(stripConstMapView(mt.Elem()), fl.pkg.tpkg) {
		return what + " refers to an unexported field that cannot be named in this package", false
	}
	unexported := ""
	walkConstMapTypes(mt, func(n *types2.Named) bool {
		obj := n.Obj()
		if obj == nil {
			return true
		}
		if pkg := obj.Pkg(); pkg != nil && pkg != fl.pkg.tpkg && !obj.Exported() {
			unexported = obj.Name()
			return false
		}
		return true
	})
	if unexported != "" {
		return what + " uses unexported type " + unexported + " that cannot be named in this package", false
	}
	return "", true
}

func walkConstMapTypes(mt *types2.Map, visit func(n *types2.Named) bool) {
	seen := map[types2.Type]bool{}
	var walk func(t types2.Type)
	walk = func(t types2.Type) {
		if t == nil || seen[t] {
			return
		}
		seen[t] = true
		if cv, ok := t.(*types2.ConstView); ok {
			walk(cv.Elem)
			return
		}
		u := types2.Unalias(t)
		if n, ok := u.(*types2.Named); ok {
			if !visit(n) {
				return
			}
			if args := n.TypeArgs(); args != nil {
				for i := range args.Len() {
					walk(args.At(i))
				}
			}
			u = n.Underlying()
		}
		switch u := u.(type) {
		case *types2.Struct:
			for i := range u.NumFields() {
				walk(u.Field(i).Type())
			}
		case *types2.Array:
			walk(u.Elem())
		case *types2.Slice:
			walk(u.Elem())
		case *types2.Map:
			walk(u.Key())
			walk(u.Elem())
		case *types2.Pointer:
			walk(u.Elem())
		}
	}
	walk(mt.Key())
	walk(mt.Elem())
}

func (fl *fileLower) constMapImportQuals(mt *types2.Map) (map[string]*types2.Package, string) {
	quals := map[string]*types2.Package{}
	ambiguous := ""
	walkConstMapTypes(mt, func(n *types2.Named) bool {
		obj := n.Obj()
		if obj == nil {
			return true
		}
		pkg := obj.Pkg()
		if pkg == nil || pkg == fl.pkg.tpkg {
			return true
		}
		if prev, dup := quals[obj.Name()]; dup {
			if prev != pkg {
				ambiguous = obj.Name()
				return false
			}
			return true
		}
		quals[obj.Name()] = pkg
		return true
	})
	return quals, ambiguous
}

func (fl *fileLower) constMapImportQual(pkg *types2.Package) (string, bool) {
	if q, ok := fl.importNames()[pkg]; ok && q == "" {
		return "", false
	}
	return fl.synthQualifier(pkg), true
}

func (fl *fileLower) constMapQualName(value string, quals map[string]*types2.Package) (string, bool) {
	pkg, ok := quals[value]
	if !ok {
		return "", false
	}
	return fl.constMapImportQual(pkg)
}

func (fl *fileLower) rewriteConstMapType(t syntax.Expr, quals map[string]*types2.Package, pos syntax.Pos) syntax.Expr {
	switch t := t.(type) {
	case *syntax.Name:
		if q, ok := fl.constMapQualName(t.Value, quals); ok {
			return selOf(pos, nameOf(pos, q), t.Value)
		}
		return t
	case *syntax.SelectorExpr:
		if _, ok := t.X.(*syntax.Name); ok {
			if q, ok := fl.constMapQualName(t.Sel.Value, quals); ok {
				t.X = nameOf(pos, q)
			}
		}
		return t
	case *syntax.SliceType:
		t.Elem = fl.rewriteConstMapType(t.Elem, quals, pos)
		return t
	case *syntax.ArrayType:
		t.Elem = fl.rewriteConstMapType(t.Elem, quals, pos)
		return t
	case *syntax.MapType:
		t.Key = fl.rewriteConstMapType(t.Key, quals, pos)
		t.Value = fl.rewriteConstMapType(t.Value, quals, pos)
		return t
	case *syntax.ParenExpr:
		t.X = fl.rewriteConstMapType(t.X, quals, pos)
		return t
	case *syntax.IndexExpr:
		t.X = fl.rewriteConstMapType(t.X, quals, pos)
		t.Index = fl.rewriteConstMapType(t.Index, quals, pos)
		return t
	default:
		return t
	}
}

func (fl *fileLower) rewriteConstMapValue(x syntax.Expr, quals map[string]*types2.Package, pos syntax.Pos) syntax.Expr {
	switch x := x.(type) {
	case *syntax.CompositeLit:
		if x.Type != nil {
			x.Type = fl.rewriteConstMapType(x.Type, quals, pos)
		}
		for i, e := range x.ElemList {
			x.ElemList[i] = fl.rewriteConstMapElem(e, quals, pos)
		}
		return x
	case *syntax.CallExpr:
		x.Fun = fl.rewriteConstMapType(x.Fun, quals, pos)
		for i, a := range x.ArgList {
			x.ArgList[i] = fl.rewriteConstMapValue(a, quals, pos)
		}
		return x
	case *syntax.Operation:
		x.X = fl.rewriteConstMapValue(x.X, quals, pos)
		if x.Y != nil {
			x.Y = fl.rewriteConstMapValue(x.Y, quals, pos)
		}
		return x
	case *syntax.SliceExpr:
		x.X = fl.rewriteConstMapValue(x.X, quals, pos)
		for i, idx := range x.Index {
			if idx != nil {
				x.Index[i] = fl.rewriteConstMapValue(idx, quals, pos)
			}
		}
		return x
	case *syntax.ParenExpr:
		x.X = fl.rewriteConstMapValue(x.X, quals, pos)
		return x
	default:
		return x
	}
}

func (fl *fileLower) rewriteConstMapElem(e syntax.Expr, quals map[string]*types2.Package, pos syntax.Pos) syntax.Expr {
	if kv, ok := e.(*syntax.KeyValueExpr); ok {
		if _, isName := kv.Key.(*syntax.Name); !isName {
			kv.Key = fl.rewriteConstMapValue(kv.Key, quals, pos)
		}
		kv.Value = fl.rewriteConstMapValue(kv.Value, quals, pos)
		return kv
	}
	return fl.rewriteConstMapValue(e, quals, pos)
}

func parseConstMapEntryExpr(text string, pos syntax.Pos) (syntax.Expr, bool) {
	file, err := syntax.Parse(syntax.NewFileBase("logos-constmap-entry"), strings.NewReader("package p\nvar _ = "+text), nil, nil, 0)
	if err != nil || file == nil || len(file.DeclList) != 1 {
		return nil, false
	}
	vd, ok := file.DeclList[0].(*syntax.VarDecl)
	if !ok || vd.Values == nil {
		return nil, false
	}
	e := vd.Values
	syntax.Inspect(e, func(n syntax.Node) bool {
		if n != nil {
			n.SetPos(pos)
		}
		return true
	})
	return e, true
}

func (fl *fileLower) qualifyConstMapTexts(texts []string, quals map[string]*types2.Package, pos syntax.Pos) ([]string, bool) {
	out := make([]string, len(texts))
	for i, tx := range texts {
		e, ok := parseConstMapEntryExpr(tx, pos)
		if !ok {
			return nil, false
		}
		out[i] = renderExpr(fl.rewriteConstMapValue(e, quals, pos))
	}
	return out, true
}

func (g *constMapGen) constMapEntries() (keys, vals []ctfe.Value) {
	return g.val.MapEntries()
}

func constMapElem(t types2.Type) types2.Type {
	if cv, ok := t.(*types2.ConstView); ok {
		return cv.Elem
	}
	return t
}
func (fl *fileLower) lowerConstMapSpec(cd *syntax.ConstDecl, g *constMapGen) {

	fl.needRT = true
	if _, ejectText, ok := fl.constMapOwnedTexts(g, cd.Pos()); ok {
		decls := g.nativeDecls
		fl.markConstMapTables(decls, g)
		fl.constDecls = append(fl.constDecls, decls...)
		fl.constText += "\n" + ejectText
		fl.dropConstSpec(cd)
		return
	}
	parts := fl.buildConstMapParts(g, cd.Pos())
	text := parts.text
	if name := cd.NameList[0].Value; token.IsExported(name) && fl.pkg.tpkg.Scope().Lookup(name) == g.obj {
		text += fmt.Sprintf("\nvar %s %s = %s()\n", name, parts.mapT, g.mapName())
		fl.recordConstMapFact(g, parts)
	}
	decls := parseConstMapDecls(text, cd.Pos())
	fl.markConstMapTables(decls, g)
	fl.constDecls = append(fl.constDecls, decls...)
	fl.constText += "\n" + text
	fl.dropConstSpec(cd)
}

func (fl *fileLower) recordConstMapFact(g *constMapGen, parts constMapParts) {
	f := facts.Current()
	if f == nil {
		return
	}
	name := g.obj.Name()
	for i := range f.ConstMaps {
		if f.ConstMaps[i].Object == name {
			return
		}
	}
	f.ConstMaps = append(f.ConstMaps, facts.ConstMapFact{
		Object:  name,
		Bearing: parts.bearing,
		Order:   parts.order,
		Hashes:  parts.hashes,
		Canons:  parts.canons,
		Keys:    parts.keyTexts,
		Vals:    parts.valTexts,
	})
}

func (fl *fileLower) markConstMapTables(decls []syntax.Decl, g *constMapGen) {
	for _, d := range decls {
		vd, ok := d.(*syntax.VarDecl)
		if !ok || len(vd.NameList) != 1 {
			continue
		}
		v := vd.NameList[0].Value
		if v != g.sortedName() && v != g.tabName() {
			continue
		}
		fl.rodataVars = append(fl.rodataVars, rodataVar{vd: vd})
	}
}

func (fl *fileLower) dropConstSpec(cd *syntax.ConstDecl) {
	if cd.Group == nil {
		if kw, ok := fl.keywords.Start(fl.src, token.CONST, fl.off(syntax.StartPos(cd))); ok {
			fl.add(kw, swallowSep(fl.src, fl.endOff(cd.Values)), "")
			return
		}
	}
	fl.deleteSpec(cd)
}

type constMapParts struct {
	text     string
	mapT     string
	keyTexts []string
	valTexts []string
	hashes   []uint64
	canons   []string
	order    []int
	bearing  bool
}

func (fl *fileLower) constMapText(g *constMapGen, pos syntax.Pos) string {
	return fl.buildConstMapParts(g, pos).text
}

func (fl *fileLower) buildConstMapParts(g *constMapGen, pos syntax.Pos) constMapParts {
	elem := constMapElem(g.obj.Type())
	mt, ok := elem.Underlying().(*types2.Map)
	if !ok {
		base.Fatalf("logos: const map %s is not a map at %s", g.obj.Name(), pos.String())
	}
	kT := fl.mustSpell(mt.Key(), pos)
	vT := fl.mustSpell(mt.Elem(), pos)
	mapT := fl.mustSpell(elem, pos)
	keys, vals := g.constMapEntries()
	keyText := make([]string, len(keys))
	valText := make([]string, len(vals))
	hashes := make([]uint64, len(keys))
	canons := make([]string, len(keys))
	for i := range keys {
		keyText[i] = fl.mustRenderWith(keys[i], mt.Key(), pos)
		valText[i] = fl.mustRenderWith(vals[i], mt.Elem(), pos)
		hashes[i] = constKeyHash(keys[i])
		canons[i] = constKeyCanon(keys[i])
	}
	order := make([]int, len(keys))
	for i := range order {
		order[i] = i
	}
	slices.SortFunc(order, func(a, b int) int { return ctfe.CompareKeys(keys[a], keys[b]) })
	entries := make([]constMapKey, len(keys))
	for i := range keys {
		entries[i] = constMapKey{hash: hashes[i], canon: canons[i]}
	}
	bearing := constMapBearing(vals)
	pairs := make([]string, len(keys))
	for i := range keys {
		pairs[i] = keyText[i] + ": " + valText[i]
	}
	body := mapT + "{" + strings.Join(pairs, ", ") + "}"
	if e, err := g.val.Expr(pos); err == nil {
		body = renderExpr(e)
	}
	text := finalizeConstMapEmission(pos, constMapEmission{
		name:        g.name,
		kT:          kT,
		vT:          vT,
		mapT:        mapT,
		keyTexts:    keyText,
		valTexts:    valText,
		builderBody: body,
		hashExpr:    fl.constKeyHashText(mt.Key(), "_logos_k", pos),
		order:       order,
		bearing:     bearing,
	}, entries)
	return constMapParts{
		text:     text,
		mapT:     mapT,
		keyTexts: keyText,
		valTexts: valText,
		hashes:   hashes,
		canons:   canons,
		order:    order,
		bearing:  bearing,
	}
}

func constMapSeedBudget(pos syntax.Pos) constMapBudget {
	if base.Debug.LogosConstmapSeeds < 0 {
		base.Errorf("%s: invalid -d=logosconstmapseeds value %d (want a non-negative seed cap)", pos.String(), base.Debug.LogosConstmapSeeds)
		base.ExitIfErrors()
	}
	if base.Debug.LogosConstmapSeeds > 0 {
		return constMapBudget{MaxSeeds: base.Debug.LogosConstmapSeeds, Rounds: defaultConstMapBudget.Rounds}
	}
	return defaultConstMapBudget
}

func finalizeConstMapEmission(pos syntax.Pos, e constMapEmission, entries []constMapKey) string {
	isSwitch := len(entries) <= constMapSwitchMax
	var layout *constMapLayout
	if !isSwitch {
		var err error
		layout, err = buildConstMap(entries, constMapSeedBudget(pos))
		if err != nil {
			base.Errorf("%s: %v", pos.String(), err)
			base.ExitIfErrors()
		}
	}
	e.layout = layout
	e.isSwitch = isSwitch
	return emitConstMapDecls(e)
}

func (fl *fileLower) constKeyHashText(t types2.Type, x string, pos syntax.Pos) string {
	switch u := t.Underlying().(type) {
	case *types2.Basic:
		switch {
		case u.Info()&types2.IsString != 0:
			return rtAlias + ".HashString(string(" + x + "))"
		case u.Info()&types2.IsBoolean != 0:
			return rtAlias + ".HashBool(bool(" + x + "))"
		case u.Info()&types2.IsInteger != 0:
			return rtAlias + ".HashUint64(uint64(" + x + "))"
		case u.Info()&types2.IsFloat != 0:
			return rtAlias + ".HashFloat64(float64(" + x + "))"
		case u.Info()&types2.IsComplex != 0:
			return rtAlias + ".HashCombine(" + rtAlias + ".HashFloat64(float64(real(" + x + "))), " + rtAlias + ".HashFloat64(float64(imag(" + x + "))))"
		}
	case *types2.Struct:
		var b strings.Builder
		fmt.Fprintf(&b, "func("+constMapHashRecv+" %s) uint64 { "+constMapHashAcc+" := uint64(0); ", fl.mustSpell(t, pos))
		for i := range u.NumFields() {
			f := u.Field(i)
			if f.Name() == "_" {
				continue
			}
			fmt.Fprintf(&b, constMapHashAcc+" = %s.HashCombine("+constMapHashAcc+", %s); ", rtAlias, fl.constKeyHashText(f.Type(), constMapHashRecv+"."+f.Name(), pos))
		}
		fmt.Fprintf(&b, "return "+constMapHashAcc+" }(%s)", x)
		return b.String()
	case *types2.Array:
		return fmt.Sprintf("func(_logos_v %s) uint64 { _logos_h := uint64(0); for _, _logos_e := range _logos_v { _logos_h = %s.HashCombine(_logos_h, %s) }; return _logos_h }(%s)",
			fl.mustSpell(t, pos), rtAlias, fl.constKeyHashText(u.Elem(), "_logos_e", pos), x)
	}
	base.Fatalf("logos: const map key type %s cannot be hashed at %s", t, pos.String())
	return ""
}

func (fl *fileLower) mustSpell(t types2.Type, pos syntax.Pos) string {
	s, ok := fl.spellLowered(t)
	if !ok {
		base.Fatalf("logos: cannot spell const map type %s at %s", t, pos.String())
	}
	return s
}

func (fl *fileLower) mustRender(v ctfe.Value, pos syntax.Pos) string {
	e, err := v.Expr(pos)
	if err != nil {
		base.Fatalf("logos: cannot spell const map element at %s: %v", pos.String(), err)
	}
	return renderExpr(e)
}

func (fl *fileLower) mustRenderWith(v ctfe.Value, t types2.Type, pos syntax.Pos) string {
	if e, err := v.Expr(pos); err == nil {
		return renderExpr(e)
	}
	if _, s, ok := fl.buildMatValue(v, t, pos); ok {
		return s
	}
	return fl.mustRender(v, pos)
}

func parseConstMapDecls(text string, pos syntax.Pos) []syntax.Decl {
	file, err := syntax.Parse(syntax.NewFileBase("logos-constmap"), strings.NewReader("package p\n"+text), nil, nil, 0)
	if err != nil {
		base.Fatalf("logos: cannot parse generated const map code at %s: %v", pos.String(), err)
	}
	for _, d := range file.DeclList {
		syntax.Inspect(d, func(n syntax.Node) bool {
			if n != nil {
				n.SetPos(pos)
				setRbrace(n, pos)
			}
			return true
		})
	}
	return file.DeclList
}

func setRbrace(n syntax.Node, pos syntax.Pos) {
	switch n := n.(type) {
	case *syntax.BlockStmt:
		n.Rbrace = pos
	case *syntax.SwitchStmt:
		n.Rbrace = pos
	case *syntax.SelectStmt:
		n.Rbrace = pos
	case *syntax.CompositeLit:
		n.Rbrace = pos
	}
}

func (fl *fileLower) lowerConstMapRanges() {
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		rc, ok := n.(*syntax.RangeClause)
		if !ok {
			return true
		}
		g, ok := fl.constMapUse(rc.X)
		if !ok {
			fl.lowerSortedRange(rc)
			return true
		}
		pos := rc.X.Pos()
		start, end := fl.off(syntax.StartPos(rc.X)), fl.endOff(rc.X)
		if g.bearing {
			fl.add(start, end, g.allName())
			fl.needRT = true
			rc.X = nameOf(pos, g.allName())
			return true
		}
		fl.add(start, end, rtAlias+".ConstAll("+g.sortedName()+")")
		fl.needRT = true
		rc.X = callOf(pos, selOf(pos, nameOf(pos, rtAlias), "ConstAll"), []syntax.Expr{nameOf(pos, g.sortedName())})
		return true
	})
}

func (fl *fileLower) lowerSortedRange(rc *syntax.RangeClause) {
	if !fl.rangesConstMap(rc.X) {
		return
	}
	pos := rc.X.Pos()
	start, end := fl.off(syntax.StartPos(rc.X)), fl.endOff(rc.X)
	fl.add(start, start, rtAlias+".SortedAll(")
	fl.add(end, end, ")")
	fl.needRT = true
	rc.X = callOf(pos, selOf(pos, nameOf(pos, rtAlias), "SortedAll"), []syntax.Expr{rc.X})
}

func (fl *fileLower) rangesConstMap(x syntax.Expr) bool {
	t := fl.pkg.info.TypeOf(x)
	if t == nil {
		return false
	}
	if view := constAsView(t); view != nil {
		_, isMap := compositeUnderlying(view.Elem).(*types2.Map)
		return isMap
	}
	ref, ok := fl.compositeRefOf(x)
	if !ok || ref.method != nil {
		return false
	}
	_, isMap := compositeUnderlying(t).(*types2.Map)
	return isMap
}

func (fl *fileLower) lowerConstMapName(e *syntax.Name) (syntax.Expr, bool) {
	g, ok := fl.constMapUse(e)
	if !ok {
		return nil, false
	}
	start := fl.off(e.Pos())
	fl.add(start, start+len(e.Value), g.mapName()+"()")
	return callOf(e.Pos(), nameOf(e.Pos(), g.mapName()), nil), true
}
func (fl *fileLower) lowerConstMapSelector(e *syntax.SelectorExpr, g *constMapGen) syntax.Expr {
	fl.add(fl.off(syntax.StartPos(e)), fl.endOff(e), g.mapName()+"()")
	return callOf(e.Pos(), nameOf(e.Pos(), g.mapName()), nil)
}

func (fl *fileLower) lowerConstMapLen(e *syntax.CallExpr) (syntax.Expr, bool) {
	fn, ok := e.Fun.(*syntax.Name)
	if !ok || fn.Value != "len" || len(e.ArgList) != 1 {
		return nil, false
	}
	if _, ok := fl.pkg.info.Uses[fn].(*types2.Builtin); !ok {
		return nil, false
	}
	g, ok := fl.constMapUse(e.ArgList[0])
	if !ok {
		return nil, false
	}
	n := strconv.Itoa(g.nkeys)
	fl.add(fl.off(syntax.StartPos(e)), fl.endOff(e), n)
	return at(&syntax.BasicLit{Value: n, Kind: syntax.IntLit}, e.Pos()), true
}

func (fl *fileLower) lowerConstMapRead(e *syntax.IndexExpr, g *constMapGen) syntax.Expr {
	pos := e.Pos()
	id := fl.nextWrapID()
	mStart, lbrack, rbrack := fl.lookupSpans(e)
	e.Index = fl.lowerOptExpr(e.Index)
	args := []syntax.Expr{e.Index}
	head, open := g.lookupName(), "("
	if g.lead != nil {
		args = []syntax.Expr{g.lead, e.Index}
		head += "(" + string(fl.src[fl.off(syntax.StartPos(g.lead)):fl.exprEndOf(g.lead)]) + ", "
		open = ""
	}
	lookup := callOf(pos, nameOf(pos, g.lookupName()), args)
	opt := fl.asOption(fl.optType(e))
	if opt == nil {
		fl.addOwned(id, mStart, lbrack, head)
		fl.add(lbrack, lbrack+1, open)
		fl.add(rbrack, rbrack+1, ")")
		fl.wrapSpan[lookup] = span{start: mStart, end: rbrack + 1}
		fl.wrapOwner[lookup] = id
		return lookup
	}
	wrap := "Opt"
	if types2.LogosOptionIsBare(opt.Elem) {
		wrap = "Bare"
	}
	fl.needRT = true
	out := callOf(pos, selOf(pos, nameOf(pos, rtAlias), wrap), []syntax.Expr{lookup})
	fl.addOwned(id, mStart, lbrack, rtAlias+"."+wrap+"("+head)
	fl.add(lbrack, lbrack+1, open)
	fl.add(rbrack, rbrack+1, "))")
	fl.wrapSpan[out] = span{start: mStart, end: rbrack + 1}
	fl.wrapOwner[out] = id
	return out
}
