// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

type freezeClass int

const (
	freezePlain freezeClass = iota
	freezeLeaf
	freezeRef
	freezeBody
	freezeDynamic
)

type freezeEntry struct {
	typ  types2.Type
	name string
}

type freezeCopy struct {
	text string
	expr syntax.Expr
}

type freezeSite struct {
	fl    *fileLower
	pos   syntax.Pos
	local []freezeEntry
	vars  []string
	defs  []string
}

func (fl *fileLower) lowerFreezes() {
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		if call, ok := n.(*syntax.CallExpr); ok {
			fl.lowerFreezeCall(call)
		}
		return true
	})
}

func (fl *fileLower) lowerFreezeCall(call *syntax.CallExpr) {
	name := freezeCallee(call.Fun)
	if name == nil || len(call.ArgList) != 1 || call.HasDots || !fl.isRTFreeze(name) {
		return
	}
	inst, ok := fl.pkg.info.Instances[name]
	if !ok || inst.TypeArgs.Len() != 1 {
		return
	}
	cp, ok := fl.freezeCopier(inst.TypeArgs.At(0), syntax.StartPos(call))
	if !ok {
		return
	}
	start := fl.off(name.Pos())
	fl.add(start, start+len(name.Value), "FreezeRoot")
	name.Value = "FreezeRoot"
	fl.addAfterExpression(fl.exprEndOf(call.ArgList[0]), ", "+cp.text)
	call.ArgList = append(call.ArgList, cp.expr)
}

func freezeCallee(fun syntax.Expr) *syntax.Name {
	fun = syntax.Unparen(fun)
	if ix, ok := fun.(*syntax.IndexExpr); ok {
		fun = syntax.Unparen(ix.X)
	}
	switch f := fun.(type) {
	case *syntax.SelectorExpr:
		return f.Sel
	case *syntax.Name:
		return f
	}
	return nil
}

func (fl *fileLower) isRTFreeze(name *syntax.Name) bool {
	fn, ok := fl.pkg.info.Uses[name].(*types2.Func)
	return ok && fn.Name() == "Freeze" && fn.Pkg() != nil && fn.Pkg().Path() == rtImportPath
}

func (fl *fileLower) freezeCopier(t types2.Type, pos syntax.Pos) (freezeCopy, bool) {
	switch fl.freezeKind(t) {
	case freezePlain:
		return freezeCopy{text: "nil", expr: nilOf(pos)}, true
	case freezeDynamic:
		return freezeCopy{}, false
	}
	if _, known := freezeLookup(fl.pkg.freezeCopiers, t); !known && !fl.freezeSpellable(t) {
		return freezeCopy{}, false
	}
	site := &freezeSite{fl: fl, pos: pos}
	name := site.ensure(t)
	if len(site.defs) == 0 {
		return freezeCopy{text: name, expr: nameOf(pos, name)}, true
	}
	text := site.closure(name, site.spell(t))
	decl := parseGeneratedDecls("var _ = "+text, pos)[0].(*syntax.VarDecl)
	return freezeCopy{text: text, expr: decl.Values}, true
}

func (fl *fileLower) freezeCore(t types2.Type) types2.Type {
	for {
		if o := fl.asOption(t); o != nil && types2.LogosOptionIsBare(o.Elem) {
			t = o.Elem
			continue
		}
		switch u := types2.Unalias(t).(type) {
		case *types2.ConstView:
			t = u.Elem
		case *types2.ReadonlyDescendants:
			t = u.Elem
		default:
			return u
		}
	}
}

func (fl *fileLower) freezeUnder(t types2.Type) types2.Type {
	core := fl.freezeCore(t)
	if o := fl.asOption(core); o != nil {
		return o
	}
	if _, ok := core.(*types2.TypeParam); ok {
		return core
	}
	u := core.Underlying()
	if tup, ok := u.(*types2.LogosTuple); ok {
		return tup.Underlying()
	}
	return u
}

func (fl *fileLower) freezeKind(t types2.Type) freezeClass {
	switch u := fl.freezeUnder(t).(type) {
	case *types2.Basic:
		return freezeBasicKind(u)
	case *types2.Signature, *types2.Chan:
		return freezeLeaf
	case *types2.Pointer, *types2.Slice, *types2.Map:
		return freezeRef
	case *types2.Array:
		return fl.freezeHolderKind(u.Elem())
	case *types2.Option:
		return fl.freezeHolderKind(u.Elem)
	case *types2.Struct:
		return fl.freezeStructKind(u)
	case *types2.Interface:
		return fl.freezeInterfaceKind(t)
	}
	return freezeDynamic
}

func freezeBasicKind(b *types2.Basic) freezeClass {
	if b.Kind() == types2.UnsafePointer {
		return freezeLeaf
	}
	return freezePlain
}

func (fl *fileLower) freezeHolderKind(elem types2.Type) freezeClass {
	if fl.freezeKind(elem) == freezePlain {
		return freezePlain
	}
	return freezeBody
}

func (fl *fileLower) freezeStructKind(st *types2.Struct) freezeClass {
	kind := freezePlain
	for i := range st.NumFields() {
		f := st.Field(i)
		if fl.freezeKind(f.Type()) == freezePlain {
			continue
		}
		if !fl.freezeNameable(f) {
			return freezeDynamic
		}
		kind = freezeBody
	}
	return kind
}

func (fl *fileLower) freezeNameable(f *types2.Var) bool {
	return f.Name() != "_" && (f.Exported() || f.Pkg() == fl.pkg.tpkg)
}

func (fl *fileLower) freezeInterfaceKind(t types2.Type) freezeClass {
	if _, ok := types2.LogosUnionTerms(fl.freezeCore(t)); ok {
		return freezeBody
	}
	return freezeDynamic
}

func (fl *fileLower) freezeLeafName(t types2.Type) string {
	switch fl.freezeUnder(t).(type) {
	case *types2.Signature:
		return "func"
	case *types2.Chan:
		return "chan"
	}
	return "unsafe.Pointer"
}

func (fl *fileLower) freezeSpellable(t types2.Type) bool {
	_, ok := fl.spellLowered(t)
	return ok
}

func freezeScoped(t types2.Type) bool {
	seen := make(map[types2.Type]bool)
	var walk func(types2.Type) bool
	walk = func(t types2.Type) bool {
		if t == nil || seen[t] {
			return false
		}
		seen[t] = true
		return freezeLocal(t) || slices.ContainsFunc(freezeParts(t), walk)
	}
	return walk(t)
}

func freezeLocal(t types2.Type) bool {
	var obj *types2.TypeName
	switch u := t.(type) {
	case *types2.TypeParam:
		return true
	case *types2.Named:
		obj = u.Obj()
	case *types2.Alias:
		obj = u.Obj()
	default:
		return false
	}
	return obj.Pkg() != nil && obj.Parent() != obj.Pkg().Scope()
}

func freezeParts(t types2.Type) []types2.Type {
	switch u := t.(type) {
	case *types2.Named:
		return freezeTypeArgs(u.TypeArgs())
	case *types2.Alias:
		return freezeTypeArgs(u.TypeArgs())
	case *types2.Map:
		return []types2.Type{u.Key(), u.Elem()}
	case interface{ Elem() types2.Type }:
		return []types2.Type{u.Elem()}
	case *types2.ConstView:
		return []types2.Type{u.Elem}
	case *types2.ReadonlyDescendants:
		return []types2.Type{u.Elem}
	case *types2.Option:
		return []types2.Type{u.Elem}
	}
	return freezeMemberParts(t)
}

func freezeTypeArgs(list *types2.TypeList) []types2.Type {
	out := make([]types2.Type, list.Len())
	for i := range out {
		out[i] = list.At(i)
	}
	return out
}

func freezeMemberParts(t types2.Type) []types2.Type {
	var out []types2.Type
	switch u := t.(type) {
	case *types2.LogosTuple:
		return freezeMemberParts(u.Underlying())
	case *types2.Struct:
		for i := range u.NumFields() {
			out = append(out, u.Field(i).Type())
		}
	case *types2.Signature:
		out = append(freezeTupleTypes(u.Params()), freezeTupleTypes(u.Results())...)
	case *types2.Interface:
		for i := range u.NumMethods() {
			out = append(out, u.Method(i).Type())
		}
		for i := range u.NumEmbeddeds() {
			out = append(out, u.EmbeddedType(i))
		}
	}
	return out
}

func freezeTupleTypes(tup *types2.Tuple) []types2.Type {
	out := make([]types2.Type, tup.Len())
	for i := range out {
		out[i] = tup.At(i).Type()
	}
	return out
}

func freezeLookup(reg []freezeEntry, t types2.Type) (string, bool) {
	for _, e := range reg {
		if types2.Identical(e.typ, t) {
			return e.name, true
		}
	}
	return "", false
}

func (s *freezeSite) ensure(t types2.Type) string {
	hoist := !freezeScoped(t)
	reg := &s.local
	if hoist {
		reg = &s.fl.pkg.freezeCopiers
	}
	if name, ok := freezeLookup(*reg, t); ok {
		return name
	}
	s.fl.pkg.freezeSeq++
	name := fmt.Sprintf("_logos_fz%d", s.fl.pkg.freezeSeq)
	*reg = append(*reg, freezeEntry{typ: t, name: name})
	spell := s.spell(t)
	params := "(_logos_fr *" + rtAlias + ".Freezer, _logos_d, _logos_s *" + spell + ")"
	body := s.body(t)
	s.fl.needRT = true
	if hoist {
		text := "func " + name + params + " {\n" + body + "}\n"
		s.fl.constDecls = append(s.fl.constDecls, parseGeneratedDecls(text, s.pos)...)
		s.fl.constText += "\n" + text
		return name
	}
	s.vars = append(s.vars, "var "+name+" func(*"+rtAlias+".Freezer, *"+spell+", *"+spell+")")
	s.defs = append(s.defs, name+" = func"+params+" {\n"+body+"}")
	return name
}

func (s *freezeSite) closure(root, spell string) string {
	sig := "func(*" + rtAlias + ".Freezer, *" + spell + ", *" + spell + ")"
	return "func() " + sig + " {\n" + strings.Join(s.vars, "\n") + "\n" + strings.Join(s.defs, "\n") + "\nreturn " + root + "\n}()"
}

func (s *freezeSite) spell(t types2.Type) string {
	text, ok := s.fl.spellLowered(t)
	if !ok {
		base.Fatalf("logos: cannot spell frozen type %s at %s", t, s.pos.String())
	}
	return text
}

func (s *freezeSite) body(t types2.Type) string {
	switch u := s.fl.freezeUnder(t).(type) {
	case *types2.Struct:
		return s.structBody(u)
	case *types2.Array:
		return "*_logos_d = *_logos_s\nfor _logos_i := range _logos_s {\n" + s.stmt(u.Elem(), "_logos_d[_logos_i]", "_logos_s[_logos_i]") + "}\n"
	case *types2.Option:
		return "*_logos_d = *_logos_s\n" + s.stmt(u.Elem, "_logos_d.Value", "_logos_s.Value")
	case *types2.Interface:
		return s.unionBody(t, u)
	}
	return s.stmt(t, "*_logos_d", "*_logos_s")
}

func (s *freezeSite) structBody(st *types2.Struct) string {
	var b strings.Builder
	b.WriteString("*_logos_d = *_logos_s\n")
	for i := range st.NumFields() {
		name := st.Field(i).Name()
		b.WriteString(s.stmt(st.Field(i).Type(), "_logos_d."+name, "_logos_s."+name))
	}
	return b.String()
}

func (s *freezeSite) unionBody(t types2.Type, iface *types2.Interface) string {
	members, _ := types2.LogosUnionTerms(s.fl.freezeCore(t))
	plain := []string{"nil"}
	var cases strings.Builder
	for _, m := range members {
		spell, ok := s.fl.spellLowered(m)
		if !ok {
			return rtAlias + ".FreezeDynamic(_logos_fr, _logos_d, _logos_s)\n"
		}
		if !types2.Implements(m, iface) {
			continue
		}
		if s.fl.freezeKind(m) == freezePlain {
			plain = append(plain, spell)
			continue
		}
		cases.WriteString("case " + spell + ":\nvar _logos_c " + spell + "\n" + s.stmt(m, "_logos_c", "_logos_v") + "*_logos_d = _logos_c\n")
	}
	guard := "switch (*_logos_s).(type) {\n"
	if cases.Len() > 0 {
		guard = "switch _logos_v := (*_logos_s).(type) {\n"
	}
	return guard + "case " + strings.Join(plain, ", ") + ":\n*_logos_d = *_logos_s\n" + cases.String() + "default:\n" + rtAlias + ".FreezeDynamic(_logos_fr, _logos_d, _logos_s)\n}\n"
}

func (s *freezeSite) stmt(t types2.Type, dst, src string) string {
	switch kind := s.fl.freezeKind(t); {
	case kind == freezePlain:
		return ""
	case kind == freezeLeaf:
		return "if " + src + " != nil {\n" + rtAlias + ".FreezeReject(" + strconv.Quote(s.fl.freezeLeafName(t)) + ")\n}\n"
	case kind == freezeRef:
		return dst + " = " + s.refCall(t, src) + "\n"
	case kind == freezeDynamic || !s.fl.freezeSpellable(t):
		return rtAlias + ".FreezeDynamic(_logos_fr, &" + dst + ", &" + src + ")\n"
	}
	return s.ensure(t) + "(_logos_fr, &" + dst + ", &" + src + ")\n"
}

func (s *freezeSite) refCall(t types2.Type, src string) string {
	var fn string
	var elems []types2.Type
	switch u := s.fl.freezeUnder(t).(type) {
	case *types2.Pointer:
		fn, elems = "FreezePointer", []types2.Type{u.Elem()}
	case *types2.Slice:
		fn, elems = "FreezeSlice", []types2.Type{u.Elem()}
	case *types2.Map:
		fn, elems = "FreezeMap", []types2.Type{u.Key(), u.Elem()}
	}
	call := rtAlias + "." + fn + "(_logos_fr, " + src
	for _, e := range elems {
		call += ", " + s.copier(e)
	}
	return call + ")"
}

func (s *freezeSite) copier(t types2.Type) string {
	switch kind := s.fl.freezeKind(t); {
	case kind == freezePlain:
		return "nil"
	case kind == freezeDynamic || !s.fl.freezeSpellable(t):
		return rtAlias + ".FreezeDynamic"
	}
	return s.ensure(t)
}
