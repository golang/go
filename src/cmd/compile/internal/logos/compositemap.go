// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"go/constant"
	"go/token"
	"maps"
	"slices"
	"strconv"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/logos/ctfe"
	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

type compositePathStep struct {
	field   string
	index   int
	isField bool
	dynamic bool
}

type compositeHidden struct {
	steps   []compositePathStep
	lens    []int
	mapType types2.Type
	val     ctfe.Value
	gen     *constMapGen
	parts   constMapParts
}

type compositeFamily struct {
	name    string
	length  int
	members map[int]*compositeHidden
}

type compositeInfo struct {
	obj      *types2.Const
	val      ctfe.Value
	typ      types2.Type
	base     string
	hiddens  []*compositeHidden
	byPath   map[string]*compositeHidden
	families map[string]*compositeFamily
	famOrder []*compositeFamily
}

func (c *compositeInfo) builder() string { return c.base + "_build" }
func (c *compositeInfo) ro() string      { return c.base + "_ro" }

func compositeUnderlying(t types2.Type) types2.Type {
	for {
		switch u := types2.Unalias(t).(type) {
		case *types2.ConstView:
			t = u.Elem
		case *types2.ReadonlyDescendants:
			t = u.Elem
		case *types2.Named:
			return u.Underlying()
		default:
			return u
		}
	}
}

func compositeTypeReachesMap(t types2.Type) bool {
	seen := map[types2.Type]bool{}
	var walk func(t types2.Type) bool
	walk = func(t types2.Type) bool {
		if t == nil || seen[t] {
			return false
		}
		seen[t] = true
		switch u := compositeUnderlying(t).(type) {
		case *types2.Map:
			return true
		case *types2.Struct:
			for i := range u.NumFields() {
				if walk(u.Field(i).Type()) {
					return true
				}
			}
		case *types2.Slice:
			return walk(u.Elem())
		case *types2.Array:
			return walk(u.Elem())
		case *types2.Pointer:
			return walk(u.Elem())
		}
		return false
	}
	return walk(t)
}

func compositeIsBare(v ctfe.Value, t types2.Type) bool {
	if _, ok := compositeUnderlying(t).(*types2.Map); !ok {
		return false
	}
	return v.Type() != nil && v.Type().Kind == ctfe.KMap
}

func compositeIsBearing(obj *types2.Const, v ctfe.Value) bool {
	if obj == nil || !v.IsValid() || obj.Type() == nil {
		return false
	}
	if compositeIsBare(v, obj.Type()) {
		return false
	}
	return compositeTypeReachesMap(obj.Type())
}

func compositePathText(steps []compositePathStep) string {
	var b strings.Builder
	for _, s := range steps {
		switch {
		case s.dynamic:
			b.WriteString("[*]")
		case s.isField:
			b.WriteString("." + s.field)
		default:
			b.WriteString("[" + strconv.Itoa(s.index) + "]")
		}
	}
	return b.String()
}

func parseCompositePath(text string) ([]compositePathStep, bool) {
	var steps []compositePathStep
	for text != "" {
		switch text[0] {
		case '.':
			end := strings.IndexAny(text[1:], ".[")
			if end < 0 {
				end = len(text) - 1
			}
			steps = append(steps, compositePathStep{field: text[1 : 1+end], isField: true})
			text = text[1+end:]
		case '[':
			end := strings.IndexByte(text, ']')
			if end < 2 {
				return nil, false
			}
			i, err := strconv.Atoi(text[1:end])
			if err != nil {
				return nil, false
			}
			steps = append(steps, compositePathStep{index: i})
			text = text[end+1:]
		default:
			return nil, false
		}
	}
	return steps, len(steps) > 0
}

func collectCompositeHiddens(val ctfe.Value, typ types2.Type, steps []compositePathStep, lens []int, visit func([]compositePathStep, []int, types2.Type, ctfe.Value)) {
	if val.IsNilValue() {
		return
	}
	switch u := compositeUnderlying(typ).(type) {
	case *types2.Map:
		visit(slices.Clone(steps), slices.Clone(lens), typ, val)
	case *types2.Struct:
		fields := val.Fields()
		for i := range min(u.NumFields(), len(fields)) {
			ns := append(slices.Clone(steps), compositePathStep{field: u.Field(i).Name(), isField: true})
			collectCompositeHiddens(fields[i], u.Field(i).Type(), ns, append(slices.Clone(lens), 0), visit)
		}
	case *types2.Array:
		elems := val.Elems()
		for i, e := range elems {
			collectCompositeHiddens(e, u.Elem(), append(slices.Clone(steps), compositePathStep{index: i}), append(slices.Clone(lens), len(elems)), visit)
		}
	case *types2.Slice:
		backing, lo, length, _, ok := val.SliceHeader()
		if !ok {
			return
		}
		for i := range min(length, len(backing)-lo) {
			collectCompositeHiddens(backing[lo+i], u.Elem(), append(slices.Clone(steps), compositePathStep{index: i}), append(slices.Clone(lens), length), visit)
		}
	}
}

func compositeHiddenMap(mt types2.Type) (*types2.Map, bool) {
	m, ok := compositeUnderlying(mt).(*types2.Map)
	return m, ok
}

func (fl *fileLower) buildCompositeHiddenParts(name string, mapType types2.Type, val ctfe.Value, pos syntax.Pos) (constMapParts, bool) {
	var zero constMapParts
	mmt, ok := compositeHiddenMap(mapType)
	if !ok {
		return zero, false
	}
	kT := fl.mustSpell(mmt.Key(), pos)
	vT := fl.mustSpell(mmt.Elem(), pos)
	mapT := fl.mustSpell(mapType, pos)
	keys, vals := val.MapEntries()
	keyText, valText := fl.renderHiddenEntries(keys, vals, mmt, pos)
	hashes, canons := hiddenHashes(keys)
	order := hiddenOrder(keys)
	entries := hiddenEntries(hashes, canons)
	bearing := constMapBearing(vals)
	text := finalizeConstMapEmission(pos, constMapEmission{
		name:        name,
		kT:          kT,
		vT:          vT,
		mapT:        mapT,
		keyTexts:    keyText,
		valTexts:    valText,
		builderBody: fl.mustRenderWith(val, mapType, pos),
		hashExpr:    fl.constKeyHashText(mmt.Key(), "_logos_k", pos),
		order:       order,
		bearing:     bearing,
	}, entries)
	return constMapParts{text: text, mapT: mapT, keyTexts: keyText, valTexts: valText, hashes: hashes, canons: canons, order: order, bearing: bearing}, true
}

func (fl *fileLower) renderHiddenEntries(keys, vals []ctfe.Value, mmt *types2.Map, pos syntax.Pos) ([]string, []string) {
	keyText := make([]string, len(keys))
	valText := make([]string, len(vals))
	for i := range keys {
		keyText[i] = fl.mustRenderWith(keys[i], mmt.Key(), pos)
		valText[i] = fl.mustRenderWith(vals[i], mmt.Elem(), pos)
	}
	return keyText, valText
}

func hiddenHashes(keys []ctfe.Value) ([]uint64, []string) {
	hashes := make([]uint64, len(keys))
	canons := make([]string, len(keys))
	for i := range keys {
		hashes[i] = constKeyHash(keys[i])
		canons[i] = constKeyCanon(keys[i])
	}
	return hashes, canons
}

func hiddenOrder(keys []ctfe.Value) []int {
	order := make([]int, len(keys))
	for i := range order {
		order[i] = i
	}
	slices.SortFunc(order, func(a, b int) int { return ctfe.CompareKeys(keys[a], keys[b]) })
	return order
}

func hiddenEntries(hashes []uint64, canons []string) []constMapKey {
	entries := make([]constMapKey, len(hashes))
	for i := range hashes {
		entries[i] = constMapKey{hash: hashes[i], canon: canons[i]}
	}
	return entries
}

func (fl *fileLower) buildNilMappedValue(v ctfe.Value, t types2.Type, pos syntax.Pos) (syntax.Expr, string, bool) {
	if v.IsNilValue() {
		return fl.nilMappedNull(t, pos)
	}
	switch compositeUnderlying(t).(type) {
	case *types2.Map:
		return nilOf(pos), "nil", true
	case *types2.Struct:
		return fl.nilMappedStruct(v, t, pos)
	case *types2.Array:
		return fl.nilMappedArray(v, t, pos)
	case *types2.Slice:
		return fl.nilMappedSlice(v, t, pos)
	}
	return fl.buildMatValue(v, t, pos)
}

func (fl *fileLower) nilMappedNull(t types2.Type, pos syntax.Pos) (syntax.Expr, string, bool) {
	spell, ok := fl.spellLowered(t)
	if !ok {
		return nil, "", false
	}
	te, ok := reparseTypeAt(spell, pos)
	if !ok {
		return nil, "", false
	}
	return at(&syntax.CallExpr{Fun: te, ArgList: []syntax.Expr{nilOf(pos)}}, pos), spell + "(nil)", true
}

func (fl *fileLower) nilMappedStruct(v ctfe.Value, t types2.Type, pos syntax.Pos) (syntax.Expr, string, bool) {
	u, ok := compositeUnderlying(t).(*types2.Struct)
	if !ok {
		return nil, "", false
	}
	fields := v.Fields()
	if len(fields) != u.NumFields() {
		return nil, "", false
	}
	spell, ok := fl.spellLowered(t)
	if !ok {
		return nil, "", false
	}
	te, ok := reparseTypeAt(spell, pos)
	if !ok {
		return nil, "", false
	}
	elems := make([]syntax.Expr, len(fields))
	texts := make([]string, len(fields))
	for i := range fields {
		e, s, ok := fl.buildNilMappedValue(fields[i], u.Field(i).Type(), pos)
		if !ok {
			return nil, "", false
		}
		elems[i], texts[i] = e, s
	}
	lit := at(&syntax.CompositeLit{Type: te, ElemList: elems, Rbrace: pos}, pos)
	setRbrace(lit, pos)
	return lit, spell + "{" + strings.Join(texts, ", ") + "}", true
}

func (fl *fileLower) nilMappedArray(v ctfe.Value, t types2.Type, pos syntax.Pos) (syntax.Expr, string, bool) {
	u, ok := compositeUnderlying(t).(*types2.Array)
	if !ok {
		return nil, "", false
	}
	arr := v.Elems()
	if int64(len(arr)) != u.Len() {
		return nil, "", false
	}
	spell, ok := fl.spellLowered(t)
	if !ok {
		return nil, "", false
	}
	te, ok := reparseTypeAt(spell, pos)
	if !ok {
		return nil, "", false
	}
	elems := make([]syntax.Expr, len(arr))
	texts := make([]string, len(arr))
	for i := range arr {
		e, s, ok := fl.buildNilMappedValue(arr[i], u.Elem(), pos)
		if !ok {
			return nil, "", false
		}
		elems[i], texts[i] = e, s
	}
	lit := at(&syntax.CompositeLit{Type: te, ElemList: elems, Rbrace: pos}, pos)
	setRbrace(lit, pos)
	return lit, spell + "{" + strings.Join(texts, ", ") + "}", true
}

func (fl *fileLower) nilMappedSlice(v ctfe.Value, t types2.Type, pos syntax.Pos) (syntax.Expr, string, bool) {
	u, ok := compositeUnderlying(t).(*types2.Slice)
	if !ok {
		return nil, "", false
	}
	backing, lo, length, cap_, ok := v.SliceHeader()
	if !ok || lo < 0 || length < 0 || lo > cap_ || lo+length > cap_ || cap_ > len(backing) {
		return nil, "", false
	}
	spell, ok := fl.spellLowered(t)
	if !ok {
		return nil, "", false
	}
	te, ok := reparseTypeAt(spell, pos)
	if !ok {
		return nil, "", false
	}
	vis := backing[lo : lo+length]
	elems := make([]syntax.Expr, len(vis))
	texts := make([]string, len(vis))
	for i := range vis {
		e, s, ok := fl.buildNilMappedValue(vis[i], u.Elem(), pos)
		if !ok {
			return nil, "", false
		}
		elems[i], texts[i] = e, s
	}
	lit := at(&syntax.CompositeLit{Type: te, ElemList: elems, Rbrace: pos}, pos)
	setRbrace(lit, pos)
	return lit, spell + "{" + strings.Join(texts, ", ") + "}", true
}

func collectComposites(files []*syntax.File, info *types2.Info) (map[*types2.Const]*compositeInfo, int) {
	out := map[*types2.Const]*compositeInfo{}
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
			if v, ok := info.LogosCTFE[obj]; ok && compositeIsBearing(obj, v) {
				out[obj] = newCompositeInfo(obj, v, seq)
				seq++
			}
			return true
		})
	}
	return out, seq
}

func newCompositeInfo(obj *types2.Const, val ctfe.Value, seq int) *compositeInfo {
	c := &compositeInfo{obj: obj, val: val, typ: obj.Type(), base: fmt.Sprintf("_logos_cc%d_%s", seq, obj.Name()), byPath: map[string]*compositeHidden{}, families: map[string]*compositeFamily{}}
	collectCompositeHiddens(val, c.typ, nil, nil, func(steps []compositePathStep, lens []int, mt types2.Type, v ctfe.Value) {
		keys, vals := v.MapEntries()
		gen := &constMapGen{name: fmt.Sprintf("%s_h%d", c.base, len(c.hiddens)), val: v, nkeys: len(keys), bearing: constMapBearing(vals)}
		h := &compositeHidden{steps: steps, lens: lens, mapType: mt, val: v, gen: gen}
		c.hiddens = append(c.hiddens, h)
		c.byPath[compositePathText(steps)] = h
		c.joinFamilies(h)
	})
	return c
}

func (c *compositeInfo) joinFamilies(h *compositeHidden) {
	for p, s := range h.steps {
		if s.isField {
			continue
		}
		template := slices.Clone(h.steps)
		template[p] = compositePathStep{dynamic: true}
		key := compositePathText(template)
		fam, ok := c.families[key]
		if !ok {
			fam = &compositeFamily{name: fmt.Sprintf("%s_f%d", c.base, len(c.families)), length: h.lens[p], members: map[int]*compositeHidden{}}
			c.families[key] = fam
			c.famOrder = append(c.famOrder, fam)
		}
		fam.members[s.index] = h
	}
}

func (fl *fileLower) emitComposite(info *compositeInfo, pos syntax.Pos) bool {
	for _, h := range info.hiddens {
		if !fl.emitCompositeHidden(h, pos) {
			return false
		}
	}
	for _, fam := range info.famOrder {
		fl.emitCompositeFamily(fam, pos)
	}
	return fl.emitCompositeBuilder(info, pos) && fl.emitCompositeRO(info, pos)
}

const compositeIndexConstraint = "~int | ~int8 | ~int16 | ~int32 | ~int64 | ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr"

func (fl *fileLower) emitCompositeFamily(fam *compositeFamily, pos syntax.Pos) {
	indexes := slices.Sorted(maps.Keys(fam.members))
	first := fam.members[indexes[0]]
	mmt, _ := compositeHiddenMap(first.mapType)
	kT := fl.mustSpell(mmt.Key(), pos)
	vT := fl.mustSpell(mmt.Elem(), pos)
	var b strings.Builder
	fmt.Fprintf(&b, "func %s_lookup[_logos_I %s](_logos_i _logos_I, _logos_k %s) (%s, bool) {\n\tvar _logos_n [%d]struct{}\n\t_ = _logos_n[_logos_i]\n\tswitch int(_logos_i) {\n", fam.name, compositeIndexConstraint, kT, vT, fam.length)
	for _, i := range indexes {
		fmt.Fprintf(&b, "\tcase %d:\n\t\treturn %s(_logos_k)\n", i, fam.members[i].gen.lookupName())
	}
	fmt.Fprintf(&b, "\t}\n\tvar _logos_zero %s\n\treturn _logos_zero, false\n}\n", vT)
	fl.constDecls = append(fl.constDecls, parseConstMapDecls(b.String(), pos)...)
	fl.constText += "\n" + b.String()
}

func (fl *fileLower) emitCompositeHidden(h *compositeHidden, pos syntax.Pos) bool {
	fl.needRT = true
	parts, ok := fl.buildCompositeHiddenParts(h.gen.name, h.mapType, h.val, pos)
	if !ok {
		return false
	}
	decls := parseConstMapDecls(parts.text, pos)
	fl.markConstMapTables(decls, h.gen)
	fl.constDecls = append(fl.constDecls, decls...)
	fl.constText += "\n" + parts.text
	h.parts = parts
	return true
}

func (fl *fileLower) emitCompositeBuilder(info *compositeInfo, pos syntax.Pos) bool {
	spell, ok := fl.spellLowered(info.typ)
	if !ok {
		return false
	}
	_, matText, ok := fl.buildMatValue(info.val, info.typ, pos)
	if !ok {
		return false
	}
	buildText := fmt.Sprintf("func %s() %s { return %s }\n", info.builder(), spell, matText)
	fl.constDecls = append(fl.constDecls, parseConstMapDecls(buildText, pos)...)
	fl.constText += "\n" + buildText
	return true
}

func (fl *fileLower) emitCompositeRO(info *compositeInfo, pos syntax.Pos) bool {
	spell, ok := fl.spellLowered(info.typ)
	if !ok {
		return false
	}
	_, nilText, ok := fl.buildNilMappedValue(info.val, info.typ, pos)
	if !ok {
		return false
	}
	roText := fmt.Sprintf("%s\nvar %s %s = %s\n", rodataMarker, info.ro(), spell, nilText)
	rdecls := parseConstMapDecls(roText, pos)
	for _, d := range rdecls {
		if vd, ok := d.(*syntax.VarDecl); ok {
			fl.rodataVars = append(fl.rodataVars, rodataVar{vd: vd})
		}
	}
	fl.constDecls = append(fl.constDecls, rdecls...)
	fl.constText += "\n" + roText
	return true
}

func (fl *fileLower) lowerCompositeSpec(cd *syntax.ConstDecl, info *compositeInfo) {
	pos := cd.Pos()
	if !fl.emitComposite(info, pos) {
		base.Fatalf("logos: cannot lower map-bearing composite %s at %s", info.obj.Name(), pos.String())
	}
	if name := cd.NameList[0].Value; token.IsExported(name) && fl.pkg.tpkg.Scope().Lookup(name) == info.obj {
		spell, ok := fl.spellLowered(info.typ)
		if !ok {
			base.Fatalf("logos: cannot spell exported composite %s at %s", info.obj.Name(), pos.String())
		}
		text := fmt.Sprintf("\nvar %s %s = %s()\n", name, spell, info.builder())
		fl.constDecls = append(fl.constDecls, parseConstMapDecls(text, pos)...)
		fl.constText += "\n" + text
		fl.recordCompositeFacts(info)
	}
	fl.dropConstSpec(cd)
}

func (fl *fileLower) recordCompositeFacts(info *compositeInfo) {
	f := facts.Current()
	name := info.obj.Name()
	if f == nil || slices.Contains(f.ConstComps, name) {
		return
	}
	f.ConstComps = append(f.ConstComps, name)
	for _, h := range info.hiddens {
		p := h.parts
		f.ConstMaps = append(f.ConstMaps, facts.ConstMapFact{Object: name, Path: compositePathText(h.steps), Bearing: p.bearing, Order: slices.Clone(p.order), Hashes: slices.Clone(p.hashes), Canons: slices.Clone(p.canons), Keys: slices.Clone(p.keyTexts), Vals: slices.Clone(p.valTexts)})
	}
}

type compositeRef struct {
	info   *compositeInfo
	imp    *types2.Var
	root   syntax.Expr
	steps  []compositePathStep
	method *types2.Func
	recv   syntax.Expr
	lead   syntax.Expr
}

func (r compositeRef) static() bool {
	return !slices.ContainsFunc(r.steps, func(s compositePathStep) bool { return s.dynamic })
}

func (fl *fileLower) compositeRefOf(e syntax.Expr) (compositeRef, bool) {
	switch n := syntax.Unparen(e).(type) {
	case *syntax.Name:
		return fl.compositeNameRef(n)
	case *syntax.SelectorExpr:
		if ref, ok := fl.compositeQualifiedRef(n); ok {
			return ref, true
		}
		return fl.compositeSelectorRef(n)
	case *syntax.IndexExpr:
		return fl.compositeIndexRef(n)
	}
	return compositeRef{}, false
}

func (fl *fileLower) compositeNameRef(n *syntax.Name) (compositeRef, bool) {
	switch obj := fl.pkg.info.Uses[n].(type) {
	case *types2.Const:
		if info := fl.pkg.composites[obj]; info != nil && n.Value == obj.Name() {
			return compositeRef{info: info, root: n, recv: n}, true
		}
	case *types2.Var:
		if obj.Pkg() != fl.pkg.tpkg && importedCompositeFacts(obj) != nil {
			return compositeRef{imp: obj, root: n, recv: n}, true
		}
	}
	return compositeRef{}, false
}

func (fl *fileLower) compositeQualifiedRef(n *syntax.SelectorExpr) (compositeRef, bool) {
	pkg, ok := n.X.(*syntax.Name)
	if !ok {
		return compositeRef{}, false
	}
	if _, ok := fl.pkg.info.Uses[pkg].(*types2.PkgName); !ok {
		return compositeRef{}, false
	}
	obj, ok := fl.pkg.info.Uses[n.Sel].(*types2.Var)
	if !ok || obj.Pkg() == fl.pkg.tpkg || importedCompositeFacts(obj) == nil {
		return compositeRef{}, false
	}
	return compositeRef{imp: obj, root: n, recv: n}, true
}

func (fl *fileLower) compositeSelectorRef(n *syntax.SelectorExpr) (compositeRef, bool) {
	sel := fl.pkg.info.Selections[n]
	if sel == nil {
		return compositeRef{}, false
	}
	ref, ok := fl.compositeRefOf(n.X)
	if !ok || ref.method != nil {
		return compositeRef{}, false
	}
	path := sel.Index()
	if sel.Kind() == types2.MethodVal {
		ref.method, _ = sel.Obj().(*types2.Func)
		path = path[:len(path)-1]
		ref.recv = n.X
	} else if sel.Kind() == types2.FieldVal {
		ref.recv = n
	} else {
		return compositeRef{}, false
	}
	steps, ok := compositeFieldSteps(sel.Recv(), path)
	ref.steps = append(slices.Clone(ref.steps), steps...)
	return ref, ok
}

func compositeFieldSteps(t types2.Type, path []int) ([]compositePathStep, bool) {
	steps := make([]compositePathStep, 0, len(path))
	for _, fi := range path {
		deref := false
		if p, ok := compositeUnderlying(t).(*types2.Pointer); ok {
			t, deref = p.Elem(), true
		}
		st, ok := compositeUnderlying(t).(*types2.Struct)
		if !ok || fi >= st.NumFields() {
			return nil, false
		}
		steps = append(steps, compositePathStep{field: st.Field(fi).Name(), isField: true, dynamic: deref})
		t = st.Field(fi).Type()
	}
	return steps, true
}

func (fl *fileLower) compositeIndexRef(n *syntax.IndexExpr) (compositeRef, bool) {
	switch compositeUnderlying(fl.pkg.info.TypeOf(n.X)).(type) {
	case *types2.Array, *types2.Slice:
	default:
		return compositeRef{}, false
	}
	ref, ok := fl.compositeRefOf(n.X)
	if !ok || ref.method != nil {
		return compositeRef{}, false
	}
	step := compositePathStep{dynamic: true}
	if cv, ok := typeConst(fl.pkg.info, n.Index); ok && cv.Kind() == constant.Int {
		if i, exact := constant.Int64Val(cv); exact && i >= 0 {
			step = compositePathStep{index: int(i)}
		}
	}
	if step.dynamic {
		ref.lead = n.Index
	}
	ref.steps = append(slices.Clone(ref.steps), step)
	ref.recv = n
	return ref, true
}

func (fl *fileLower) compositeFamilyFor(e syntax.Expr) (*constMapGen, bool) {
	ref, ok := fl.compositeRefOf(e)
	if !ok || ref.info == nil || ref.method != nil || ref.lead == nil || !fl.simpleIndex(ref.lead) {
		return nil, false
	}
	fam, ok := ref.info.families[compositePathText(ref.steps)]
	if !ok {
		return nil, false
	}
	return &constMapGen{name: fam.name, lead: ref.lead}, true
}

func (fl *fileLower) simpleIndex(e syntax.Expr) bool {
	switch n := syntax.Unparen(e).(type) {
	case *syntax.BasicLit:
		return true
	case *syntax.Name:
		v, ok := fl.pkg.info.Uses[n].(*types2.Var)
		if !ok {
			return false
		}
		b, ok := v.Type().Underlying().(*types2.Basic)
		return ok && b.Info()&types2.IsInteger != 0
	case *syntax.Operation:
		return fl.simpleIndex(n.X) && (n.Y == nil || fl.simpleIndex(n.Y))
	}
	return false
}

func (fl *fileLower) compositeHiddenFor(e syntax.Expr) (*constMapGen, bool) {
	ref, ok := fl.compositeRefOf(e)
	if !ok || ref.method != nil || !ref.static() {
		return nil, false
	}
	key := compositePathText(ref.steps)
	if ref.info != nil {
		h, ok := ref.info.byPath[key]
		if !ok {
			return nil, false
		}
		return h.gen, true
	}
	return fl.importedCompositeHidden(ref.imp, key, syntax.StartPos(ref.root))
}

func (fl *fileLower) compositeRewriteRoot(e syntax.Expr) (syntax.Expr, bool) {
	ref, ok := fl.compositeRefOf(e)
	if !ok || fl.compositeDone[ref.root] {
		return nil, false
	}
	fl.compositeDone[ref.root] = true
	if ref.info == nil {
		return fl.compositeImportedRewrite(e, ref), true
	}
	return fl.compositeLocalRewrite(e, ref), true
}

func (fl *fileLower) compositeLocalRewrite(e syntax.Expr, ref compositeRef) syntax.Expr {
	t := fl.pkg.info.TypeOf(ref.recv)
	switch {
	case t != nil && !compositeTypeReachesMap(t):
		fl.compositeRename(ref.root.(*syntax.Name), ref.info.ro())
		return e
	case ref.method != nil && compositePointerReceiver(ref.method):
		return fl.compositeBuildAddr(e, ref)
	}
	return fl.compositeBuild(e, ref)
}

func (fl *fileLower) compositeImportedRewrite(e syntax.Expr, ref compositeRef) syntax.Expr {
	t := fl.pkg.info.TypeOf(ref.recv)
	switch {
	case t != nil && compositeReferenceFree(t):
		return e
	case ref.method != nil && compositePointerReceiver(ref.method):
		return fl.compositeFreezeAddr(e, ref)
	}
	return fl.compositeFreeze(e, ref)
}

func compositeReferenceFree(t types2.Type) bool {
	switch u := compositeUnderlying(t).(type) {
	case *types2.Basic:
		return u.Kind() != types2.UnsafePointer
	case *types2.Array:
		return compositeReferenceFree(u.Elem())
	case *types2.Struct:
		for i := range u.NumFields() {
			if !compositeReferenceFree(u.Field(i).Type()) {
				return false
			}
		}
		return true
	}
	return false
}

func (fl *fileLower) compositeShapeArg(call *syntax.CallExpr) {
	fn, ok := call.Fun.(*syntax.Name)
	if !ok || (fn.Value != "len" && fn.Value != "cap") || len(call.ArgList) != 1 {
		return
	}
	if _, ok := fl.pkg.info.Uses[fn].(*types2.Builtin); !ok {
		return
	}
	arg := call.ArgList[0]
	ref, ok := fl.compositeRefOf(arg)
	if !ok || fl.compositeDone[ref.root] {
		return
	}
	switch compositeUnderlying(fl.pkg.info.TypeOf(arg)).(type) {
	case *types2.Array, *types2.Slice:
		fl.compositeDone[ref.root] = true
		if ref.info != nil {
			fl.compositeRename(ref.root.(*syntax.Name), ref.info.ro())
		}
	}
}

func (fl *fileLower) compositeRename(root *syntax.Name, to string) {
	start := fl.off(root.Pos())
	fl.add(start, start+len(root.Value), to)
	root.Value = to
}

func compositePointerReceiver(fn *types2.Func) bool {
	sig, ok := fn.Type().(*types2.Signature)
	if !ok || sig.Recv() == nil {
		return false
	}
	t := types2.Unalias(sig.Recv().Type())
	if cv, ok := t.(*types2.ConstView); ok {
		t = types2.Unalias(cv.Elem)
	}
	_, ptr := t.(*types2.Pointer)
	return ptr
}

func (fl *fileLower) compositeBuild(e syntax.Expr, ref compositeRef) syntax.Expr {
	pos := syntax.StartPos(ref.root)
	call := callOf(pos, nameOf(pos, ref.info.builder()), nil)
	call.SetTypeInfo(ref.root.GetTypeInfo())
	return fl.compositeSwapRoot(e, ref, ref.info.builder()+"()", call)
}

func compositeSliceRoot(ref compositeRef) bool {
	_, ok := compositeUnderlying(ref.root.GetTypeInfo().Type).(*types2.Slice)
	return ok
}

func (fl *fileLower) compositeBuildAddr(e syntax.Expr, ref compositeRef) syntax.Expr {
	if compositeSliceRoot(ref) {
		return fl.compositeBuild(e, ref)
	}
	pos := syntax.StartPos(ref.root)
	fresh := callOf(pos, nameOf(pos, "new"), []syntax.Expr{callOf(pos, nameOf(pos, ref.info.builder()), nil)})
	tv := ref.root.GetTypeInfo()
	tv.Type = types2.NewPointer(tv.Type)
	fresh.SetTypeInfo(tv)
	return fl.compositeSwapRoot(e, ref, "new("+ref.info.builder()+"())", fresh)
}

func (fl *fileLower) compositeFreeze(e syntax.Expr, ref compositeRef) syntax.Expr {
	pos := syntax.StartPos(ref.root)
	text, call := fl.compositeFreezeCall(ref)
	deref := at(&syntax.ParenExpr{X: at(&syntax.Operation{Op: syntax.Mul, X: call}, pos)}, pos)
	deref.SetTypeInfo(ref.root.GetTypeInfo())
	return fl.compositeSwapRoot(e, ref, "(*"+text+")", deref)
}

func (fl *fileLower) compositeFreezeAddr(e syntax.Expr, ref compositeRef) syntax.Expr {
	if compositeSliceRoot(ref) {
		return fl.compositeFreeze(e, ref)
	}
	text, call := fl.compositeFreezeCall(ref)
	tv := ref.root.GetTypeInfo()
	tv.Type = types2.NewPointer(tv.Type)
	call.SetTypeInfo(tv)
	return fl.compositeSwapRoot(e, ref, text, call)
}

func (fl *fileLower) compositeFreezeCall(ref compositeRef) (string, *syntax.CallExpr) {
	fl.needRT = true
	pos := syntax.StartPos(ref.root)
	return rtAlias + ".Freeze(&" + renderExpr(ref.root) + ")", callOf(pos, rtSel(pos, "Freeze"), []syntax.Expr{addrOf(pos, ref.root)})
}

func (fl *fileLower) compositeRootSpan(root syntax.Expr) span {
	if sel, ok := root.(*syntax.SelectorExpr); ok {
		return span{start: fl.off(syntax.StartPos(root)), end: fl.off(sel.Sel.Pos()) + len(sel.Sel.Value)}
	}
	name := root.(*syntax.Name)
	start := fl.off(name.Pos())
	return span{start: start, end: start + len(name.Value)}
}

func (fl *fileLower) compositeSwapRoot(e syntax.Expr, ref compositeRef, text string, repl syntax.Expr) syntax.Expr {
	s := fl.compositeRootSpan(ref.root)
	fl.add(s.start, s.end, text)
	fl.wrapSpan[repl] = s
	return compositeReplaceRoot(e, ref.root, repl)
}

func compositeReplaceRoot(e syntax.Expr, root, repl syntax.Expr) syntax.Expr {
	if e == root {
		return repl
	}
	switch n := e.(type) {
	case *syntax.ParenExpr:
		n.X = compositeReplaceRoot(n.X, root, repl)
	case *syntax.SelectorExpr:
		n.X = compositeReplaceRoot(n.X, root, repl)
	case *syntax.IndexExpr:
		n.X = compositeReplaceRoot(n.X, root, repl)
	}
	return e
}

func (fl *fileLower) rejectCompositeSurvivors() {
	selectors := map[syntax.Node]bool{}
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		if sel, ok := n.(*syntax.SelectorExpr); ok {
			selectors[sel.Sel] = true
		}
		if e, ok := n.(syntax.Expr); ok && !selectors[n] {
			if ref, ok := fl.compositeRefOf(e); ok && ref.root == e && !fl.compositeDone[e] {
				base.Fatalf("logos: const composite %s at %s was not lowered", renderExpr(e), syntax.StartPos(e).String())
			}
		}
		return true
	})
}

type importedComposite struct {
	facts   *facts.Facts
	prefix  string
	hiddens map[string]*constMapGen
	tried   map[string]bool
}

func importedCompositeFacts(obj *types2.Var) *facts.Facts {
	if obj.Pkg() == nil || obj.IsField() || obj.Pkg().Scope().Lookup(obj.Name()) != types2.Object(obj) {
		return nil
	}
	f, ok := facts.Lookup(obj.Pkg().Path())
	if !ok || f == nil || !slices.Contains(f.ConstComps, obj.Name()) {
		return nil
	}
	return f
}

func (fl *fileLower) importedCompositeFor(obj *types2.Var, pos syntax.Pos) (*importedComposite, bool) {
	if info, ok := fl.pkg.importedComposites[obj]; ok {
		return info, true
	}
	f := importedCompositeFacts(obj)
	if f == nil {
		return nil, false
	}
	for _, rec := range f.ConstMaps {
		if rec.Object != obj.Name() || rec.Path == "" {
			continue
		}
		if _, _, ok := fl.importedCompositeRecordType(obj, rec, pos); !ok {
			return nil, false
		}
	}
	info := &importedComposite{
		facts:   f,
		prefix:  fmt.Sprintf("_logos_icomp%d_%s", fl.pkg.compositeSeq, obj.Name()),
		hiddens: map[string]*constMapGen{},
		tried:   map[string]bool{},
	}
	fl.pkg.compositeSeq++
	fl.pkg.importedComposites[obj] = info
	return info, true
}

func (fl *fileLower) importedCompositeHidden(obj *types2.Var, path string, pos syntax.Pos) (*constMapGen, bool) {
	info, ok := fl.importedCompositeFor(obj, pos)
	if !ok {
		return nil, false
	}
	if g, ok := info.hiddens[path]; ok {
		return g, true
	}
	if info.tried[path] {
		return nil, false
	}
	info.tried[path] = true
	for i, rec := range info.facts.ConstMaps {
		if rec.Object != obj.Name() || rec.Path != path {
			continue
		}
		g, ok := fl.importedCompositeMap(obj, rec, fmt.Sprintf("%s_h%d", info.prefix, i), pos)
		if ok {
			info.hiddens[path] = g
		}
		return g, ok
	}
	return nil, false
}

func (fl *fileLower) importedCompositeRecordType(obj *types2.Var, rec facts.ConstMapFact, pos syntax.Pos) (types2.Type, *types2.Map, bool) {
	var mapType types2.Type
	steps, ok := parseCompositePath(rec.Path)
	if ok {
		mapType, ok = compositeWalkType(obj.Type(), steps)
	}
	mt, isMap := compositeUnderlying(mapType).(*types2.Map)
	if !ok || !isMap {
		base.Errorf("%s: logos: imported const composite %s has a record that does not match its type", pos.String(), obj.Pkg().Path()+"."+obj.Name()+rec.Path)
		base.ExitIfErrors()
		return nil, nil, false
	}
	return mapType, mt, true
}

func (fl *fileLower) importedCompositeMap(obj *types2.Var, rec facts.ConstMapFact, name string, pos syntax.Pos) (*constMapGen, bool) {
	mapType, mt, ok := fl.importedCompositeRecordType(obj, rec, pos)
	if !ok {
		return nil, false
	}
	label := obj.Pkg().Path() + "." + obj.Name() + rec.Path
	if _, spellable := fl.unspellableImportedConstMap(label, mapType, mt); !spellable {
		return nil, false
	}
	return fl.emitImportedConstMap(importedMapSpec{label: label, mapType: mapType, mt: mt, rec: rec, name: name}, pos)
}

func compositeWalkType(t types2.Type, steps []compositePathStep) (types2.Type, bool) {
	for _, s := range steps {
		var ok bool
		if t, ok = compositeStepType(t, s); !ok {
			return nil, false
		}
	}
	return t, true
}

func compositeStepType(t types2.Type, s compositePathStep) (types2.Type, bool) {
	switch u := compositeUnderlying(t).(type) {
	case *types2.Struct:
		for i := range u.NumFields() {
			if s.isField && u.Field(i).Name() == s.field {
				return u.Field(i).Type(), true
			}
		}
	case *types2.Array:
		if !s.isField && !s.dynamic && int64(s.index) < u.Len() {
			return u.Elem(), true
		}
	case *types2.Slice:
		if !s.isField && !s.dynamic {
			return u.Elem(), true
		}
	}
	return nil, false
}
