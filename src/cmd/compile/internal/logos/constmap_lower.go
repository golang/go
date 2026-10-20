// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

// This file lowers M5-S4 const maps (SPEC T-Sec9). A const map declaration
// becomes plain Go: a lookup function, a sorted entry list for range, and a
// materializer that builds one regular map on first plain use. A map with at
// most 8 keys looks up through a switch; a larger map looks up through a
// rt.ConstTable whose layout the seed search fixed at compile time.
//
// The rewrites of one read:
//
//	m[k]         -> rt.Opt(_logos_cmN_m_lookup(k))   option read (SPEC T-202)
//	m[k] (comma) -> _logos_cmN_m_lookup(k)           comma-ok read, (V, bool)
//	range m      -> rt.ConstAll(_logos_cmN_m_sorted) sorted entries
//	len(m)       -> the entry count, a constant
//	m (other)    -> _logos_cmN_m_map()               the materialized map
//
// Lowering runs in two steps. Collect registers every const map of the
// package before any file lowers, so a use in one file sees the declaration
// in another. The declaring file then removes the const spec and appends the
// generated declarations, which are parsed from text and spelled identically
// in the ejected source.

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/logos/ctfe"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// constMapSwitchMax is the largest key count that lowers to a switch.
const constMapSwitchMax = 8

// constMapGen is one registered const map of the package.
type constMapGen struct {
	name string // prefix of the generated names, unique in the package
	obj  *types2.Const
	val  ctfe.Value
}

func (g *constMapGen) lookupName() string { return g.name + "_lookup" }
func (g *constMapGen) sortedName() string { return g.name + "_sorted" }
func (g *constMapGen) mapName() string    { return g.name + "_map" }
func (g *constMapGen) tabName() string    { return g.name + "_tab" }

// collectConstMaps registers every single-name const of the package whose
// evaluated value is a map. Names carry a package-wide sequence number, so
// const maps in different functions never collide.
func collectConstMaps(files []*syntax.File, info *types2.Info) map[*types2.Const]*constMapGen {
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
			out[obj] = &constMapGen{
				name: fmt.Sprintf("_logos_cm%d_%s", seq, cd.NameList[0].Value),
				obj:  obj,
				val:  v,
			}
			seq++
			return true
		})
	}
	return out
}

// constMapUse returns the const map that expression x names, if any.
func (fl *fileLower) constMapUse(x syntax.Expr) (*constMapGen, bool) {
	name, ok := syntax.Unparen(x).(*syntax.Name)
	if !ok {
		return nil, false
	}
	obj, ok := fl.pkg.info.Uses[name].(*types2.Const)
	if !ok {
		return nil, false
	}
	g, ok := fl.pkg.constMaps[obj]
	return g, ok
}

// constMapEntries returns the keys and values of g in insertion order.
func (g *constMapGen) constMapEntries() (keys, vals []ctfe.Value) {
	return g.val.MapEntries()
}

// constMapElem returns the map type of a const map's declared type.
func constMapElem(t types2.Type) types2.Type {
	if cv, ok := t.(*types2.ConstView); ok {
		return cv.Elem
	}
	return t
}

// lowerConstMapSpec removes one const map declaration from the source and
// records its generated declarations for this file.
func (fl *fileLower) lowerConstMapSpec(cd *syntax.ConstDecl, g *constMapGen) {
	fl.needRT = true
	text := fl.constMapText(g, cd.Pos())
	decls := parseConstMapDecls(text, cd.Pos())
	fl.markConstMapTables(decls, g)
	fl.constDecls = append(fl.constDecls, decls...)
	fl.constText += "\n" + text
	fl.dropConstSpec(cd)
}

// markConstMapTables records the sorted-entry and perfect-hash table vars
// of g for read-only placement (SPEC Sec 9.1). Their initializers are
// constant literals, so static initialization fills them; the
// materialized map var is built at run time and stays writable. The
// generated text already carries the marker lines.
func (fl *fileLower) markConstMapTables(decls []syntax.Decl, g *constMapGen) {
	for _, d := range decls {
		vd, ok := d.(*syntax.VarDecl)
		if !ok || len(vd.NameList) != 1 || vd.NameList[0].Value == g.name+"_map" {
			continue
		}
		fl.rodataVars = append(fl.rodataVars, rodataVar{vd: vd})
	}
}

// dropConstSpec deletes a const spec's source text. A standalone spec also
// takes its const keyword; a grouped spec uses the separator-swallowing
// deletion of deleteSpec.
func (fl *fileLower) dropConstSpec(cd *syntax.ConstDecl) {
	if cd.Group == nil {
		if kw, ok := constKeywordStart(fl.src, fl.off(syntax.StartPos(cd))); ok {
			fl.add(kw, swallowSep(fl.src, fl.endOff(cd.Values)), "")
			return
		}
	}
	fl.deleteSpec(cd)
}

// constMapText spells the generated declarations of one const map.
func (fl *fileLower) constMapText(g *constMapGen, pos syntax.Pos) string {
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
	for i := range keys {
		keyText[i] = fl.mustRender(keys[i], pos)
		valText[i] = fl.mustRender(vals[i], pos)
	}
	order := make([]int, len(keys))
	for i := range order {
		order[i] = i
	}
	slices.SortFunc(order, func(a, b int) int { return constKeyCompare(keys[a], keys[b]) })

	var b strings.Builder
	name := g.name
	fmt.Fprintf(&b, "%s\nvar %s_sorted = []%s.ConstEntry[%s, %s]{", rodataMarker, name, rtAlias, kT, vT)
	for j, i := range order {
		if j > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "{Key: %s, Val: %s}", keyText[i], valText[i])
	}
	b.WriteString("}\n")
	fmt.Fprintf(&b, "var %s_map = %s.Materialize(func() %s { return %s })\n", name, rtAlias, mapT, fl.mustRender(g.val, pos))

	if len(keys) <= constMapSwitchMax {
		fmt.Fprintf(&b, "func %s_lookup(k %s) (%s, bool) {\n\tswitch k {\n", name, kT, vT)
		for i := range keys {
			fmt.Fprintf(&b, "\tcase %s:\n\t\treturn %s, true\n", keyText[i], valText[i])
		}
		fmt.Fprintf(&b, "\t}\n\tvar zero %s\n\treturn zero, false\n}\n", vT)
		return b.String()
	}

	entries := make([]constMapKey, len(keys))
	for i := range keys {
		entries[i] = constMapKey{hash: constKeyHash(keys[i]), canon: constKeyCanon(keys[i])}
	}
	layout, err := buildConstMap(entries, defaultConstMapBudget)
	if err != nil {
		base.Errorf("%s: %v", pos.String(), err)
		base.ExitIfErrors()
	}
	fmt.Fprintf(&b, "%s\nvar %s = %s.ConstTable[%s, %s]{Salt: %d, Buckets: []%s.ConstBucket{", rodataMarker, g.tabName(), rtAlias, kT, vT, layout.salt, rtAlias)
	for i, bk := range layout.buckets {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "{Off: %d, Size: %d, Seed: %d}", bk.off, bk.size, bk.seed)
	}
	fmt.Fprintf(&b, "}, Slots: []%s.ConstSlot[%s, %s]{", rtAlias, kT, vT)
	for s, ki := range layout.slots {
		if s > 0 {
			b.WriteString(", ")
		}
		if ki < 0 {
			b.WriteString("{}")
			continue
		}
		fmt.Fprintf(&b, "{Key: %s, Val: %s, Occ: true}", keyText[ki], valText[ki])
	}
	b.WriteString("}}\n")
	fmt.Fprintf(&b, "func %s_lookup(k %s) (%s, bool) {\n\treturn %s.Get(%s, k)\n}\n",
		name, kT, vT, g.tabName(), fl.constKeyHashText(mt.Key(), "k", pos))
	return b.String()
}

// constKeyHashText spells the base hash of key x of type t: the expression
// that mirrors constKeyHash over ctfe values.
func (fl *fileLower) constKeyHashText(t types2.Type, x string, pos syntax.Pos) string {
	switch u := t.Underlying().(type) {
	case *types2.Basic:
		switch {
		case u.Info()&types2.IsString != 0:
			return rtAlias + ".HashString(" + x + ")"
		case u.Info()&types2.IsBoolean != 0:
			return rtAlias + ".HashBool(" + x + ")"
		case u.Info()&types2.IsInteger != 0:
			return rtAlias + ".HashUint64(uint64(" + x + "))"
		case u.Info()&types2.IsFloat != 0:
			return rtAlias + ".HashFloat64(float64(" + x + "))"
		}
	case *types2.Struct:
		var b strings.Builder
		fmt.Fprintf(&b, "func(v %s) uint64 { h := uint64(0); ", fl.mustSpell(t, pos))
		for i := range u.NumFields() {
			f := u.Field(i)
			if f.Name() == "_" {
				continue
			}
			fmt.Fprintf(&b, "h = %s.HashCombine(h, %s); ", rtAlias, fl.constKeyHashText(f.Type(), "v."+f.Name(), pos))
		}
		fmt.Fprintf(&b, "return h }(%s)", x)
		return b.String()
	case *types2.Array:
		return fmt.Sprintf("func(v %s) uint64 { h := uint64(0); for _, e := range v { h = %s.HashCombine(h, %s) }; return h }(%s)",
			fl.mustSpell(t, pos), rtAlias, fl.constKeyHashText(u.Elem(), "e", pos), x)
	}
	base.Fatalf("logos: const map key type %s cannot be hashed at %s", t, pos.String())
	return ""
}

// mustSpell spells a lowered type or stops compilation.
func (fl *fileLower) mustSpell(t types2.Type, pos syntax.Pos) string {
	s, ok := fl.spellLowered(t)
	if !ok {
		base.Fatalf("logos: cannot spell const map type %s at %s", t, pos.String())
	}
	return s
}

// mustRender prints a key or value of a const map.
func (fl *fileLower) mustRender(v ctfe.Value, pos syntax.Pos) string {
	e, err := v.Expr(pos)
	if err != nil {
		base.Fatalf("logos: cannot spell const map element at %s: %v", pos.String(), err)
	}
	return renderExpr(e)
}

// parseConstMapDecls parses generated declaration text into nodes at pos,
// so every generated node carries the declaration's position.
func parseConstMapDecls(text string, pos syntax.Pos) []syntax.Decl {
	file, err := syntax.Parse(syntax.NewFileBase("logos-constmap"), strings.NewReader("package p\n"+text), nil, nil, 0)
	if err != nil {
		base.Fatalf("logos: cannot parse generated const map code at %s: %v", pos.String(), err)
	}
	for _, d := range file.DeclList {
		syntax.Inspect(d, func(n syntax.Node) bool {
			if n != nil {
				n.SetPos(pos)
			}
			return true
		})
	}
	return file.DeclList
}

// lowerConstMapRanges rewrites `range m` over a const map to the sorted
// iterator. It runs before the option lowering so the iterator call is an
// ordinary expression for every later pass.
func (fl *fileLower) lowerConstMapRanges() {
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		rc, ok := n.(*syntax.RangeClause)
		if !ok {
			return true
		}
		g, ok := fl.constMapUse(rc.X)
		if !ok {
			return true
		}
		pos := rc.X.Pos()
		start, end := fl.off(syntax.StartPos(rc.X)), fl.endOff(rc.X)
		fl.add(start, end, rtAlias+".ConstAll("+g.sortedName()+")")
		fl.needRT = true
		rc.X = callOf(pos, selOf(pos, nameOf(pos, rtAlias), "ConstAll"), []syntax.Expr{nameOf(pos, g.sortedName())})
		return true
	})
}

// lowerConstMapName replaces a plain use of a const map with a call to its
// materialized map.
func (fl *fileLower) lowerConstMapName(e *syntax.Name) (syntax.Expr, bool) {
	g, ok := fl.constMapUse(e)
	if !ok {
		return nil, false
	}
	start := fl.off(e.Pos())
	fl.add(start, start+len(e.Value), g.mapName()+"()")
	return callOf(e.Pos(), nameOf(e.Pos(), g.mapName()), nil), true
}

// lowerConstMapLen folds len of a const map to its entry count.
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
	keys, _ := g.constMapEntries()
	n := strconv.Itoa(len(keys))
	fl.add(fl.off(syntax.StartPos(e)), fl.endOff(e), n)
	return at(&syntax.BasicLit{Value: n, Kind: syntax.IntLit}, e.Pos()), true
}

// lowerConstMapRead lowers one read of a const map. A single-value read
// becomes an option (rt.Opt, or rt.Bare for payloads that keep nil as none);
// a comma-ok read becomes the (V, bool) lookup itself.
func (fl *fileLower) lowerConstMapRead(e *syntax.IndexExpr, g *constMapGen) syntax.Expr {
	pos := e.Pos()
	id := fl.nextWrapID()
	mStart, _, lbrack, rbrack := fl.lookupSpans(e)
	e.Index = fl.lowerOptExpr(e.Index)
	lookup := callOf(pos, nameOf(pos, g.lookupName()), []syntax.Expr{e.Index})
	opt := fl.asOption(fl.optType(e))
	if opt == nil {
		fl.addOwned(id, mStart, lbrack, g.lookupName())
		fl.add(lbrack, lbrack+1, "(")
		fl.add(rbrack, rbrack+1, ")")
		fl.wrapSpan[lookup] = span{start: mStart, end: rbrack + 1}
		fl.wrapOwner[lookup] = id
		return lookup
	}
	wrap := "Opt"
	if keepMapRead(opt.Elem) {
		wrap = "Bare"
	}
	fl.needRT = true
	out := callOf(pos, selOf(pos, nameOf(pos, rtAlias), wrap), []syntax.Expr{lookup})
	fl.addOwned(id, mStart, lbrack, rtAlias+"."+wrap+"("+g.lookupName())
	fl.add(lbrack, lbrack+1, "(")
	fl.add(rbrack, rbrack+1, "))")
	fl.wrapSpan[out] = span{start: mStart, end: rbrack + 1}
	fl.wrapOwner[out] = id
	return out
}
