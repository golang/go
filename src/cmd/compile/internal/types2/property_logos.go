// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
	"slices"
	"strings"
	"sync"
)

// LoGo/s M4-S1 type rules for property interfaces (SPEC Sec 2.3,
// T-203.2). A property interface is an interface with at least one
// `const F T` or `F T` element; the parser holds each element as a
// *syntax.PropertyType. interfaceType lowers every element to accessor
// methods `Logos_GetF() T` (plus `Logos_SetF(T)` for read-write
// properties), so export data, Go importers, and the method-set
// machinery all see ordinary methods, while satisfaction stays
// conformance-driven: only a struct whose declaration lists `: I`
// satisfies I, sourced per property from an exported field or a method
// `F() T` (plus `SetF(T)` for read-write). Upstream files call in
// through hook lines only: interface.go (element lowering),
// struct_assert_logos.go (`: I` checking), instantiate.go
// (assignability), call.go (property selectors), expr.go (`&p.F`).

const (
	logosPropUnsatisfied = "[correctness::property_unsatisfied]"
	logosPropRequires    = "[correctness::property_requires_conformance]"
	logosPropNeedsPtr    = "[correctness::property_needs_pointer]"
	logosPropConflict    = "[correctness::property_conflict]"
	logosPropAddress     = "[correctness::property_address]"
)

func logosGetterName(prop string) string { return "Logos_Get" + prop }

func logosSetterName(prop string) string { return "Logos_Set" + prop }

func logosIsAccessor(name string) bool {
	return strings.HasPrefix(name, "Logos_Get") || strings.HasPrefix(name, "Logos_Set")
}

// logosPropertyInterface returns the interface when t is a property
// interface: its (flattened) method set holds a generated getter. A
// hand-written `Logos_Get` method cannot occur in .lgo (reserved, C6),
// so presence implies a property element in this or an embedded
// interface, including across package boundaries via export data.
func logosPropertyInterface(t Type) *Interface {
	if t == nil {
		return nil
	}
	iface, _ := t.Underlying().(*Interface)
	if iface == nil {
		return nil
	}
	for _, m := range iface.typeSet().methods {
		if strings.HasPrefix(m.name, "Logos_Get") {
			return iface
		}
	}
	return nil
}

// logosProperty is one property of an interface: the value type and
// whether only a getter exists (read-only).
type logosProperty struct {
	name     string
	typ      Type
	readOnly bool
}

// logosProperties reads the property set back out of an interface's
// lowered accessors, sorted by name for deterministic diagnostics.
// Malformed shapes (only possible from hand-written Go) are skipped.
func logosProperties(iface *Interface) []logosProperty {
	const get = "Logos_Get"
	var props []logosProperty
	for _, m := range iface.typeSet().methods {
		name, ok := strings.CutPrefix(m.name, get)
		if !ok {
			continue
		}
		sig, ok := m.typ.(*Signature)
		if !ok || sig.params.Len() != 0 || sig.results.Len() != 1 {
			continue
		}
		pr := logosProperty{name: name, typ: sig.results.At(0).typ, readOnly: true}
		for _, s := range iface.typeSet().methods {
			if s.name != logosSetterName(name) {
				continue
			}
			ssig, ok := s.typ.(*Signature)
			if !ok || ssig.params.Len() != 1 || ssig.results.Len() != 0 {
				continue
			}
			if Identical(ssig.params.At(0).typ, pr.typ) {
				pr.readOnly = false
			}
		}
		props = append(props, pr)
	}
	slices.SortFunc(props, func(a, b logosProperty) int { return strings.Compare(a.name, b.name) })
	return props
}

// logosFindProperty returns the property named sel of iface.
func logosFindProperty(iface *Interface, sel string) (logosProperty, bool) {
	for _, pr := range logosProperties(iface) {
		if pr.name == sel {
			return pr, true
		}
	}
	return logosProperty{}, false
}

// logosPropDecl tracks accessors already generated for one interface
// type, so repeat declarations of one name merge (read-only plus
// read-write becomes read-write, #131) instead of erroring as
// duplicate methods.
type logosPropDecl struct {
	typ    Type
	getter bool
	setter bool
}

// logosPropertyMethod lowers one `const F T` / `F T` element to its
// accessor methods, appending them to ityp.methods without recordDef:
// the generated `Logos_` names are reserved (C6) and must neither trip
// that check nor pollute Info.Defs. A repeat name with an identical
// type merges; with a different type it reports property_conflict. It
// returns the seen map, allocating it on first use so interface checks
// without properties pay nothing.
func (check *Checker) logosPropertyMethod(ityp *Interface, f *syntax.Field, pt *syntax.PropertyType, def *TypeName, seen map[string]*logosPropDecl) map[string]*logosPropDecl {
	name := f.Name.Value
	ptyp := check.typ(pt.Type)
	if !isValid(ptyp) {
		return seen
	}
	d := seen[name]
	if d != nil && !Identical(d.typ, ptyp) {
		check.errorf(f.Name, InvalidIfaceAssign, "%s property %s has conflicting types %s and %s", logosPropConflict, name, d.typ, ptyp)
		return seen
	}
	if seen == nil {
		seen = make(map[string]*logosPropDecl)
	}
	if d == nil {
		d = &logosPropDecl{typ: ptyp}
		seen[name] = d
	}
	var recvTyp Type = ityp
	if def != nil {
		if named := asNamed(def.typ); named != nil {
			recvTyp = named
		}
	}
	pos := f.Name.Pos()
	if !d.getter {
		d.getter = true
		sig := &Signature{recv: newVar(RecvVar, pos, check.pkg, "", recvTyp)}
		sig.results = NewTuple(NewParam(pos, check.pkg, "", ptyp))
		ityp.methods = append(ityp.methods, NewFunc(pos, check.pkg, logosGetterName(name), sig))
	}
	if !pt.Const && !d.setter {
		d.setter = true
		sig := &Signature{recv: newVar(RecvVar, pos, check.pkg, "", recvTyp)}
		sig.params = NewTuple(NewParam(pos, check.pkg, "", ptyp))
		ityp.methods = append(ityp.methods, NewFunc(pos, check.pkg, logosSetterName(name), sig))
	}
	return seen
}

// LogosPropDecl records one struct's asserted property interfaces for
// the M4-S2 lowering (SPEC T-203.2): the declaring type name and the
// asserted property-interface types. logosRecordConforms appends one per
// declaration to Info.LogosPropDecls; Lower reads them to generate the
// Logos_GetF/Logos_SetF accessors, and the prop facts writer reads them
// to emit each type's declared conformances. Upstream go/types drops
// this with the other logos-only declarations, so the mirror stays clean.
type LogosPropDecl struct {
	Def    *TypeName
	Ifaces []Type
}

// logosRecordConforms remembers every asserted property interface of
// the struct under declaration. It runs synchronously inside
// logosAssertedStruct (not in the delayed assert check) so later
// assignments in the same package, including package-level variable
// initializers checked before delayed actions run, see the declaration.
// It also appends to Info.LogosPropDecls for the lowering and facts.
func (check *Checker) logosRecordConforms(def *TypeName, asserts []Type) {
	named, _ := def.Type().(*Named)
	if named == nil {
		return
	}
	var props []Type
	for _, T := range asserts {
		if !isValid(T) || logosPropertyInterface(T) == nil {
			continue
		}
		if check.logosConforms == nil {
			check.logosConforms = make(map[*Named][]Type)
		}
		check.logosConforms[named] = append(check.logosConforms[named], T)
		props = append(props, T)
	}
	if len(props) > 0 {
		check.Info.LogosPropDecls = append(check.Info.LogosPropDecls, &LogosPropDecl{Def: def, Ifaces: props})
	}
}

// logosDeclaredConforms reports whether the struct named declares `: I`
// conformance to the asserted type T (whose underlying is iface).
// Recorded entries compare by object identity through Identical, with
// an underlying-pointer fallback for imported spellings. A struct from
// another package consults the imported conformances last: those come
// from the imported package's prop facts through LogosNoteImportedConforms.
func (check *Checker) logosDeclaredConforms(named *Named, T, iface Type) bool {
	match := func(R Type) bool {
		if Identical(R, T) {
			return true
		}
		Ri, ok := R.Underlying().(*Interface)
		Ii, _ := iface.Underlying().(*Interface)
		return ok && Ii != nil && Ri == Ii
	}
	for _, R := range check.logosConforms[named] {
		if match(R) {
			return true
		}
	}
	if origin := named.Origin(); origin != named {
		for _, R := range check.logosConforms[origin] {
			if match(R) {
				return true
			}
		}
	}
	return check.logosImportedConforms(named, T)
}

// LogosProp is one property of a property interface for M4-S2 consumers
// (SPEC T-203.2): the property name, its value type, and whether only a
// getter exists (read-only). It mirrors logosProperty without exposing it.
type LogosProp struct {
	Name     string
	Typ      Type
	ReadOnly bool
}

// LogosPropsOf returns the property set of t when t is a property
// interface: each lowered Logos_GetF accessor becomes one property, with
// read-only decided by the matching Logos_SetF accessor exactly as the
// checker decides. The set is sorted by name for deterministic lowering.
// It reports false for non-property types.
func LogosPropsOf(t Type) ([]LogosProp, bool) {
	iface := logosPropertyInterface(t)
	if iface == nil {
		return nil, false
	}
	inner := logosProperties(iface)
	out := make([]LogosProp, len(inner))
	for i, pr := range inner {
		out[i] = LogosProp{Name: pr.name, Typ: pr.typ, ReadOnly: pr.readOnly}
	}
	return out, true
}

// LogosConformsIface names one asserted property interface in package
// path and name form, so a checker in another package can match it
// against its own imported spelling without sharing objects.
type LogosConformsIface struct {
	PkgPath string
	Name    string
}

// logosImportedConforms remembers `: I` declarations of imported
// packages for .lgo importers (M4-S2, SPEC T-203.2): struct package path
// to struct name to asserted interfaces. The importer overlay records
// them from the imported package's prop facts; string keys keep the
// table valid across package clones (the option overlay rebuilds
// objects) and avoid pinning imported objects.
var logosImportedConforms = struct {
	mu sync.Mutex
	m  map[string]map[string][]LogosConformsIface
}{m: make(map[string]map[string][]LogosConformsIface)}

// LogosNoteImportedConforms records that the struct typeName of the
// package at structPath declares conformance to ifaces. It is called by
// the importer overlay for .lgo importers from the prop facts of the
// imported package. Repeat notes append; matching tolerates duplicates.
func LogosNoteImportedConforms(structPath, typeName string, ifaces []LogosConformsIface) {
	if structPath == "" || typeName == "" || len(ifaces) == 0 {
		return
	}
	logosImportedConforms.mu.Lock()
	defer logosImportedConforms.mu.Unlock()
	byName := logosImportedConforms.m[structPath]
	if byName == nil {
		byName = make(map[string][]LogosConformsIface)
		logosImportedConforms.m[structPath] = byName
	}
	byName[typeName] = append(byName[typeName], ifaces...)
}

// logosNamedKey names the interface T as the package path and type name
// of its (unaliased, origin) named type. It reports false for types with
// no such name.
func logosNamedKey(T Type) (string, string, bool) {
	named, ok := Unalias(T).(*Named)
	if !ok {
		return "", "", false
	}
	obj := named.Origin().Obj()
	if obj == nil || obj.Pkg() == nil || obj.Name() == "" {
		return "", "", false
	}
	return obj.Pkg().Path(), obj.Name(), true
}

// logosImportedConforms reports whether the struct named declares `: I`
// conformance to T through the imported conformances of named's own
// package. Both sides compare by package path and type name, which is
// what archive facts carry across compilations.
func (check *Checker) logosImportedConforms(named *Named, T Type) bool {
	obj := named.Origin().Obj()
	if obj == nil || obj.Pkg() == nil {
		return false
	}
	tp, tn, ok := logosNamedKey(T)
	if !ok || tp == "" || tn == "" {
		return false
	}
	logosImportedConforms.mu.Lock()
	defer logosImportedConforms.mu.Unlock()
	for _, r := range logosImportedConforms.m[obj.Pkg().Path()][obj.Name()] {
		if r.PkgPath == tp && r.Name == tn {
			return true
		}
	}
	return false
}

// logosPropFailKind classifies why a struct does not satisfy a property
// interface. Callers format the diagnostic for their position.
type logosPropFailKind int

const (
	logosPropOK logosPropFailKind = iota
	logosPropNoGetter
	logosPropNoSetter
	logosPropAmbig
	logosPropOrdinary
	logosPropFailNeedsPtr
)

// logosPropertySource finds the accessor sources of one property on V:
// a field F of identical type sources getter and setter; a method
// `F() T` sources the getter and `SetF(T)` the setter. Embedded fields
// resolve by Go selector depth; an equal-depth match is ambiguous.
func (check *Checker) logosPropertySource(V Type, name string, want Type) (getter, setter, ambig bool) {
	obj, index, _ := lookupFieldOrMethodImpl(V, false, check.pkg, name, false)
	if obj == nil {
		if index != nil {
			return false, false, true
		}
	} else if fld, ok := obj.(*Var); ok {
		if Identical(fld.typ, want) {
			return true, true, false
		}
		return false, false, false
	} else if m, ok := obj.(*Func); ok {
		check.objDecl(m)
		if sig, ok := m.typ.(*Signature); ok && sig.params.Len() == 0 && sig.results.Len() == 1 && Identical(sig.results.At(0).typ, want) {
			getter = true
		}
	}
	sobj, sindex, _ := lookupFieldOrMethodImpl(V, false, check.pkg, "Set"+name, false)
	if sobj == nil {
		return getter, false, sindex != nil
	}
	if sm, ok := sobj.(*Func); ok {
		check.objDecl(sm)
		if sig, ok := sm.typ.(*Signature); ok && sig.params.Len() == 1 && sig.results.Len() == 0 && Identical(sig.params.At(0).typ, want) {
			return getter, true, false
		}
	}
	return getter, false, false
}

// logosCheckPropertySources evaluates V against every property of
// iface plus its ordinary (non-accessor) methods, returning the first
// failure. A read-write property on a non-pointer V fails with
// logosPropFailNeedsPtr.
func (check *Checker) logosCheckPropertySources(V Type, iface *Interface) (logosPropFailKind, string) {
	for _, pr := range logosProperties(iface) {
		getter, setter, ambig := check.logosPropertySource(V, pr.name, pr.typ)
		switch {
		case ambig:
			return logosPropAmbig, pr.name
		case !getter:
			return logosPropNoGetter, pr.name
		case !pr.readOnly && !setter:
			return logosPropNoSetter, pr.name
		}
	}
	if name := check.logosMissingOrdinary(V, iface); name != "" {
		return logosPropOrdinary, name
	}
	_, isPtr := deref(V)
	if !isPtr {
		for _, pr := range logosProperties(iface) {
			if !pr.readOnly {
				return logosPropFailNeedsPtr, pr.name
			}
		}
	}
	return logosPropOK, ""
}

// logosMissingOrdinary returns the first ordinary (non-accessor) method
// of iface that V does not implement with an identical signature, or
// "" when V implements them all.
func (check *Checker) logosMissingOrdinary(V Type, iface *Interface) string {
	for _, m := range iface.typeSet().methods {
		if logosIsAccessor(m.name) {
			continue
		}
		obj, _, _ := lookupFieldOrMethodImpl(V, false, m.pkg, m.name, false)
		f, _ := obj.(*Func)
		if obj == nil || f == nil {
			return m.name
		}
		check.objDecl(f)
		if f.nointerface || !Identical(f.typ, m.typ) {
			return m.name
		}
	}
	return ""
}

// logosCheckPropertyAssert validates one asserted property interface of
// an AssertedStruct: merges its properties into merged for the
// cross-interface conflict check, reporting property_conflict at the
// assert expression. Source and method checks run separately once all
// asserts merged.
func (check *Checker) logosCheckPropertyAssert(e *syntax.AssertedStruct, T Type, i int, merged map[string]logosMergedProp) bool {
	iface := logosPropertyInterface(T)
	if iface == nil {
		return false
	}
	errored := false
	for _, pr := range logosProperties(iface) {
		if prev, ok := merged[pr.name]; ok {
			if !Identical(prev.typ, pr.typ) {
				check.errorf(e.Asserts[i], InvalidIfaceAssign, "%s property %s is %s in %s and %s in %s", logosPropConflict, pr.name, prev.typ, prev.from, pr.typ, T)
				errored = true
				continue
			}
			if !pr.readOnly {
				prev.readOnly = false
				merged[pr.name] = prev
			}
			continue
		}
		merged[pr.name] = logosMergedProp{typ: pr.typ, readOnly: pr.readOnly, from: T, pos: e.Asserts[i]}
	}
	return errored
}

// logosMergedProp is one property merged across all asserted property
// interfaces of a struct: pos anchors the source diagnostic at the
// assert that contributed it.
type logosMergedProp struct {
	typ      Type
	readOnly bool
	from     Type
	pos      syntax.Expr
}

// logosCheckMergedProp validates one merged property against V,
// reporting the first failure at the contributing assert. It reports
// whether it errored.
func (check *Checker) logosCheckMergedProp(sname string, V Type, mp logosMergedProp) bool {
	getter, setter, ambig := check.logosPropertySource(V, sname, mp.typ)
	switch {
	case ambig:
		check.errorf(mp.pos, AmbiguousSelector, "ambiguous selector %s.%s", V, sname)
		return true
	case !getter:
		check.errorf(mp.pos, InvalidIfaceAssign, "%s %s declares %s but has no field or method for property %s", logosPropUnsatisfied, V, mp.from, sname)
		return true
	case !mp.readOnly && !setter:
		check.errorf(mp.pos, InvalidIfaceAssign, "%s %s declares %s but has no field or Set%s method for mutable property %s", logosPropUnsatisfied, V, mp.from, sname, sname)
		return true
	}
	return false
}

// logosPropGate reports whether the property-interface verdict in
// implements must take over: only when the checker is active, the
// call is not inside a constraint, the package uses loGo/s, the
// asserted type is a property interface, and V is not itself an
// interface (interfaces compare by method set below, where the
// lowered accessors exist on both sides). A nil *Checker reports
// false so an implements call outside a checker keeps upstream flow.
func (check *Checker) logosPropGate(Ti *Interface, Vu Type, constraint bool) bool {
	if check == nil || constraint || !check.conf.Logos {
		return false
	}
	if _, ok := Vu.(*Interface); ok {
		return false
	}
	return logosPropertyInterface(Ti) != nil
}

// logosImplementsProperty decides whether V satisfies the property
// interface Ti (the underlying of the asserted type T) outside a
// declaration: only a (pointer to a) struct with an explicit `: T`
// entry conforms, with per-property sources and pointer shape
// enforced. Interface V falls back to the caller.
func (check *Checker) logosImplementsProperty(V, T Type, Ti *Interface, cause *string) bool {
	fail := func(format string, args ...any) bool {
		if cause != nil {
			*cause = check.sprintf(format, args...)
		}
		return false
	}
	props := logosProperties(Ti)
	getter := ""
	if len(props) > 0 {
		getter = logosGetterName(props[0].name)
	}
	base, _ := deref(V)
	named := asNamed(base)
	if named == nil {
		if _, ok := base.Underlying().(*Struct); !ok {
			return fail("%s does not implement %s (missing method %s)", V, Ti, getter)
		}
	}
	kind, name := check.logosCheckPropertySources(V, Ti)
	declared := named != nil && check.logosDeclaredConforms(named, T, Ti)
	switch {
	case declared && kind == logosPropOK:
		return true
	case declared && kind == logosPropFailNeedsPtr:
		return fail("%s %s has mutable property %s; use *%s", logosPropNeedsPtr, V, name, base)
	case declared && kind == logosPropAmbig:
		return fail("ambiguous selector %s.%s", V, name)
	case declared && kind == logosPropOrdinary:
		return fail("%s %s declares %s but does not implement method %s", logosAssertFail, V, Ti, name)
	case declared && kind == logosPropNoSetter:
		return fail("%s %s declares %s but has no field or Set%s method for mutable property %s", logosPropUnsatisfied, V, Ti, name, name)
	case declared:
		return fail("%s %s declares %s but has no field or method for property %s", logosPropUnsatisfied, V, Ti, name)
	case kind == logosPropOK:
		sname := V.String()
		if named != nil {
			sname = named.Obj().Name()
		}
		return fail("%s %s has the fields of %s but does not declare it; write type %s struct : %s { ... }", logosPropRequires, V, Ti, sname, Ti)
	default:
		return fail("%s does not implement %s (missing method %s)", V, Ti, getter)
	}
}

// logosPropertySelector types `x.Sel` where x has a property-interface
// type and Sel names a property, with the property value type. A
// read-write property yields an assignable operand; S2 lowers reads,
// writes, and compound updates to accessor calls. It reports whether it
// handled the selector.
func (check *Checker) logosPropertySelector(x *operand, e *syntax.SelectorExpr, sel string) bool {
	if !check.conf.Logos || x.typ() == nil {
		return false
	}
	iface := logosPropertyInterface(x.typ())
	if iface == nil {
		return false
	}
	pr, ok := logosFindProperty(iface, sel)
	if !ok {
		return false
	}
	if pr.readOnly {
		x.mode_ = value
	} else {
		x.mode_ = variable
	}
	x.typ_ = pr.typ
	x.expr = e
	if check.logosPropSel == nil {
		check.logosPropSel = make(map[*syntax.SelectorExpr]bool)
	}
	check.logosPropSel[e] = true
	return true
}

// logosPropertyAddress rejects `&x.F` where `x.F` typed as a property
// selection (recorded in logosPropSel when the selector checked, since
// Info.Types is absent outside the test harness). It reports whether it
// handled the address operation (on success x is invalidated).
func (check *Checker) logosPropertyAddress(x *operand, e *syntax.Operation) bool {
	if !check.conf.Logos || !x.isValid() {
		return false
	}
	sel, ok := syntax.Unparen(e.X).(*syntax.SelectorExpr)
	if !ok || !check.logosPropSel[sel] {
		return false
	}
	check.errorf(sel.Sel, UnaddressableOperand, "%s cannot take the address of property %s", logosPropAddress, sel.Sel.Value)
	x.invalidate()
	return true
}
