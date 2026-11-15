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

type logosProperty struct {
	name     string
	typ      Type
	readOnly bool
}

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

func logosFindProperty(iface *Interface, sel string) (logosProperty, bool) {
	for _, pr := range logosProperties(iface) {
		if pr.name == sel {
			return pr, true
		}
	}
	return logosProperty{}, false
}

type logosPropDecl struct {
	typ    Type
	getter bool
	setter bool
}

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
	hasMethod := func(name string) bool {
		for _, m := range ityp.methods {
			if m.name == name {
				return true
			}
		}
		return false
	}
	if !d.getter && !hasMethod(logosGetterName(name)) {
		d.getter = true
		sig := &Signature{recv: newVar(RecvVar, pos, check.pkg, "", recvTyp)}
		sig.results = NewTuple(NewParam(pos, check.pkg, "", ptyp))
		ityp.methods = append(ityp.methods, NewFunc(pos, check.pkg, logosGetterName(name), sig))
	}
	if !pt.Const && !d.setter && !hasMethod(logosSetterName(name)) {
		d.setter = true
		sig := &Signature{recv: newVar(RecvVar, pos, check.pkg, "", recvTyp)}
		sig.params = NewTuple(NewParam(pos, check.pkg, "", ptyp))
		ityp.methods = append(ityp.methods, NewFunc(pos, check.pkg, logosSetterName(name), sig))
	}
	return seen
}

type LogosPropDecl struct {
	Def    *TypeName
	Ifaces []Type
}

var logosDeclaredAccessors = struct {
	mu sync.Mutex
	m  map[*Named][]Type
}{m: make(map[*Named][]Type)}

func logosNoteDeclaredAccessors(named *Named, declared []Type) {
	if len(declared) == 0 {
		return
	}
	key := named.Origin()
	logosDeclaredAccessors.mu.Lock()
	defer logosDeclaredAccessors.mu.Unlock()
	logosDeclaredAccessors.m[key] = append(logosDeclaredAccessors.m[key], declared...)
}

func logosHolderProp(holder *Named, prop string) (Type, bool, bool) {
	logosDeclaredAccessors.mu.Lock()
	defer logosDeclaredAccessors.mu.Unlock()
	for _, iface := range logosDeclaredAccessors.m[holder] {
		if logosPropertyInterface(iface) == nil {
			continue
		}
		set, _ := LogosPropsOf(iface)
		for _, pr := range set {
			if pr.Name == prop {
				return pr.Typ, pr.ReadOnly, true
			}
		}
	}
	return nil, false, false
}

func (check *Checker) logosRecordConforms(def *TypeName, asserts []Type) {
	named, _ := def.Type().(*Named)
	if named == nil {
		return
	}
	var declared []Type
	for _, T := range asserts {
		if !isValid(T) {
			continue
		}
		if check.logosConforms == nil {
			check.logosConforms = make(map[*Named][]Type)
		}
		check.logosConforms[named] = append(check.logosConforms[named], T)
		declared = append(declared, T)
	}
	logosNoteDeclaredAccessors(named, declared)
	check.later(func() {
		check.logosRecordPropDecl(def, declared)
	}).describef(def, "record property declarations of %s", def.name)
}

func (check *Checker) logosRecordPropDecl(def *TypeName, declared []Type) {
	var props []Type
	for _, T := range declared {
		if logosPropertyInterface(T) != nil {
			props = append(props, T)
		}
	}
	if len(props) > 0 {
		check.Info.LogosPropDecls = append(check.Info.LogosPropDecls, &LogosPropDecl{Def: def, Ifaces: props})
	}
}

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
		smap := makeSubstMap(origin.TypeParams().list(), named.TypeArgs().list())
		for _, R := range check.logosConforms[origin] {
			if match(check.subst(nopos, R, smap, nil, check.context())) {
				return true
			}
		}
	}
	return check.logosImportedConforms(named, T)
}

type LogosProp struct {
	Name     string
	Typ      Type
	ReadOnly bool
}

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

func LogosPropertySelection(pkg *Package, t Type, name string) (prop LogosProp, promoted, ok bool) {
	sel := logosSelectProperty(pkg, t, name)
	if sel.kind != logosSelectInterface && sel.kind != logosSelectPromoted {
		return LogosProp{}, false, false
	}
	return LogosProp{Name: sel.prop.name, Typ: sel.prop.typ, ReadOnly: sel.prop.readOnly}, sel.kind == logosSelectPromoted, true
}

type LogosConformsIface struct {
	PkgPath string
	Name    string
}

var logosImportedConforms = struct {
	mu sync.Mutex
	m  map[string]map[string][]LogosConformsIface
}{m: make(map[string]map[string][]LogosConformsIface)}

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

type logosSelectKind uint8

const (
	logosSelectOrdinary logosSelectKind = iota
	logosSelectInterface
	logosSelectPromoted
	logosSelectAmbiguous
)

type logosSelection struct {
	kind logosSelectKind
	prop logosProperty
}

type logosMember struct {
	obj   Object
	index []int
}

func (m logosMember) found() bool { return m.obj != nil || m.index != nil }

func (m logosMember) depth() int { return len(m.index) - 1 }

func logosLookupMember(T Type, pkg *Package, name string) logosMember {
	obj, index, _ := lookupFieldOrMethodImpl(T, true, pkg, name, false)
	return logosMember{obj: obj, index: index}
}

func logosHasEmbedded(T Type) bool {
	base, _ := deref(T)
	st, _ := base.Underlying().(*Struct)
	if st == nil {
		return false
	}
	for _, f := range st.fields {
		if f.embedded {
			return true
		}
	}
	return false
}

const logosSyntheticSlot = 1 << 20

func logosHolderIsInterface(T Type, index []int) bool {
	typ, _ := deref(T)
	for _, i := range index[:len(index)-1] {
		st, _ := typ.Underlying().(*Struct)
		if st == nil {
			return false
		}
		typ, _ = deref(st.fields[i].typ)
	}
	return len(index) > 1 && IsInterface(typ)
}

func logosAccessorMember(pkg *Package, T Type, prop string) logosMember {
	real := logosLookupMember(T, pkg, logosGetterName(prop))
	syn := logosSyntheticAccessor(pkg, T, prop, real)
	if !syn.found() {
		return real
	}
	if !real.found() || syn.depth() < real.depth() {
		return syn
	}
	if syn.depth() > real.depth() || slices.Equal(real.index[:len(real.index)-1], syn.index[:len(syn.index)-1]) {
		return real
	}
	return logosMember{index: syn.index}
}

func logosSyntheticAccessor(pkg *Package, T Type, prop string, real logosMember) logosMember {
	base, _ := deref(T)
	st, _ := base.Underlying().(*Struct)
	if st == nil {
		return logosMember{}
	}
	var best logosMember
	bestDepth := -1
	offer := func(m logosMember) {
		d := m.depth()
		if bestDepth >= 0 {
			if d > bestDepth {
				return
			}
			if d == bestDepth && slices.Equal(best.index[:len(best.index)-1], m.index[:len(m.index)-1]) {
				return
			}
			if d == bestDepth {
				m = logosMember{index: m.index}
			}
		}
		best, bestDepth = m, d
	}
	if real.found() {
		offer(real)
	}
	type frame struct {
		typ   Type
		index []int
	}
	seen := make(map[*Named]bool)
	queue := []frame{{typ: T}}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		base, _ := deref(cur.typ)
		named, _ := Unalias(base).(*Named)
		if named != nil {
			if len(cur.index) > 0 && named.TypeArgs().Len() == 0 && named.TypeParams().Len() == 0 {
				if ptyp, _, ok := logosHolderProp(named, prop); ok {
					sig := &Signature{recv: newVar(RecvVar, nopos, pkg, "", named)}
					sig.results = NewTuple(NewParam(nopos, pkg, "", ptyp))
					offer(logosMember{obj: NewFunc(nopos, pkg, logosGetterName(prop), sig), index: concat(cur.index, logosSyntheticSlot)})
				}
			}
			if seen[named] {
				continue
			}
			seen[named] = true
		}
		st, _ := base.Underlying().(*Struct)
		if st == nil {
			continue
		}
		for i, f := range st.fields {
			if f.embedded {
				queue = append(queue, frame{typ: f.typ, index: concat(cur.index, i)})
			}
		}
	}
	return best
}

func logosSelectProperty(pkg *Package, T Type, name string) logosSelection {
	if iface := logosPropertyInterface(T); iface != nil {
		if pr, ok := logosFindProperty(iface, name); ok {
			return logosSelection{kind: logosSelectInterface, prop: pr}
		}
		return logosSelection{}
	}
	if !logosHasEmbedded(T) {
		return logosSelection{}
	}
	accessor := logosAccessorMember(pkg, T, name)
	if !accessor.found() || accessor.obj != nil && !logosHolderIsInterface(T, accessor.index) {
		return logosSelection{}
	}
	ordinary := logosLookupMember(T, pkg, name)
	if ordinary.found() && ordinary.depth() < accessor.depth() {
		return logosSelection{}
	}
	if accessor.obj == nil || ordinary.found() && ordinary.depth() == accessor.depth() {
		return logosSelection{kind: logosSelectAmbiguous}
	}
	pr, ok := logosPromotedProperty(pkg, T, name, accessor)
	if !ok {
		return logosSelection{}
	}
	return logosSelection{kind: logosSelectPromoted, prop: pr}
}

func logosPromotedProperty(pkg *Package, T Type, name string, getter logosMember) (logosProperty, bool) {
	get, _ := getter.obj.(*Func)
	if get == nil {
		return logosProperty{}, false
	}
	gsig, ok := get.typ.(*Signature)
	if !ok || gsig.params.Len() != 0 || gsig.results.Len() != 1 {
		return logosProperty{}, false
	}
	pr := logosProperty{name: name, typ: gsig.results.At(0).typ, readOnly: true}
	setter := logosLookupMember(T, pkg, logosSetterName(name))
	set, _ := setter.obj.(*Func)
	if set == nil || !slices.Equal(setter.index[:len(setter.index)-1], getter.index[:len(getter.index)-1]) {
		return pr, true
	}
	if ssig, ok := set.typ.(*Signature); ok && ssig.params.Len() == 1 && ssig.results.Len() == 0 && Identical(ssig.params.At(0).typ, pr.typ) {
		pr.readOnly = false
	}
	return pr, true
}

func logosAllPromoted(pkg *Package, V Type, iface *Interface) bool {
	props := logosProperties(iface)
	for _, pr := range props {
		if logosSelectProperty(pkg, V, pr.name).kind != logosSelectPromoted {
			return false
		}
	}
	return len(props) > 0
}

type logosPropFailKind int

const (
	logosPropOK logosPropFailKind = iota
	logosPropNoGetter
	logosPropNoSetter
	logosPropAmbig
	logosPropOrdinary
	logosPropFailNeedsPtr
)

func (check *Checker) logosPropertySource(V Type, name string, want Type) (getter, setter, ambig bool) {
	switch sel := logosSelectProperty(check.pkg, V, name); sel.kind {
	case logosSelectAmbiguous:
		return false, false, true
	case logosSelectPromoted:
		if !Identical(sel.prop.typ, want) {
			return false, false, false
		}
		return true, !sel.prop.readOnly, false
	}
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
			if sig.Recv() == nil || logosAsConstView(sig.Recv().Type()) == nil {
				getter = true
			}
		}
	}
	sobj, sindex, _ := lookupFieldOrMethodImpl(V, false, check.pkg, "Set"+name, false)
	if sobj == nil {
		return getter, false, sindex != nil
	}
	if sm, ok := sobj.(*Func); ok {
		check.objDecl(sm)
		if sig, ok := sm.typ.(*Signature); ok && sig.params.Len() == 1 && sig.results.Len() == 0 && Identical(sig.params.At(0).typ, want) {
			if sig.Recv() == nil || logosAsConstView(sig.Recv().Type()) == nil {
				return getter, true, false
			}
		}
	}
	return getter, false, false
}

func (check *Checker) logosCheckPropertySources(V Type, iface *Interface, pos syntax.Pos) (logosPropFailKind, string) {
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
	if name := check.logosMissingOrdinary(V, iface, pos); name != "" {
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

func (check *Checker) logosMutatingGetter(V Type, iface *Interface, pos syntax.Pos) string {
	for _, pr := range logosProperties(iface) {
		if !pr.readOnly {
			continue
		}
		if logosSelectProperty(check.pkg, V, pr.name).kind == logosSelectPromoted {
			continue
		}
		obj, _, _ := lookupFieldOrMethodImpl(V, false, check.pkg, pr.name, false)
		f, _ := obj.(*Func)
		if f == nil {
			continue
		}
		check.objDecl(f)
		if !check.logosReadonlyProofAt(V, f, pr.name, pos) {
			return pr.name
		}
	}
	return ""
}

func (check *Checker) logosMissingOrdinary(V Type, iface *Interface, pos syntax.Pos) string {
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
		if m.LogosReadonly() {
			if !check.logosConcreteReadonlySatisfiedAt(V, f, m, pos) {
				return m.name
			}
		}
	}
	return ""
}

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

type logosMergedProp struct {
	typ      Type
	readOnly bool
	from     Type
	pos      syntax.Expr
}

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

func (check *Checker) logosPropGate(Ti *Interface, Vu Type) bool {
	return check.logosPropGateFor(Ti, Vu, false)
}
func (check *Checker) logosPropGateConstraint(Ti *Interface, Vu Type) bool {
	return check.logosPropGateFor(Ti, Vu, true)
}
func (check *Checker) logosPropGateFor(Ti *Interface, Vu Type, constraint bool) bool {
	if check == nil || !check.conf.Logos {
		return false
	}
	if _, ok := Vu.(*Interface); ok {
		return false
	}
	if constraint && isTypeParam(Vu) {
		return false
	}
	return logosPropertyInterface(Ti) != nil
}

func (check *Checker) logosImplementsProperty(V, T Type, Ti *Interface, cause *string, pos syntax.Pos) bool {
	fail := func(format string, args ...any) bool {
		if cause != nil {
			*cause = check.sprintf(format, args...)
		}
		return false
	}
	if source, descendants, _ := logosReadonlyBase(V); descendants {
		for _, pr := range logosProperties(Ti) {
			if !pr.readOnly || !Identical(logosReadonlyProject(V, pr.typ, false), pr.typ) {
				return fail("%s %s cannot erase readonly qualification of property %s in %s", logosConstEscape, V, pr.name, T)
			}
		}
		for _, m := range Ti.typeSet().methods {
			if !logosIsAccessor(m.name) {
				return fail("%s %s cannot erase readonly qualification through method %s in %s", logosConstEscape, V, m.name, T)
			}
		}
		V = source
	}
	base, _ := deref(V)
	if _, ok := base.Underlying().(*Interface); ok {
		return check.hasAllMethodsAt(V, Ti, true, Identical, cause, pos)
	}
	named := asNamed(base)
	kind, name := check.logosCheckPropertySources(V, Ti, pos)
	declared := named != nil && check.logosDeclaredConforms(named, T, Ti) || logosAllPromoted(check.pkg, V, Ti)
	if named == nil {
		if _, ok := base.Underlying().(*Struct); !ok {
			return fail("%s %s does not implement %s (missing method %s)", logosAssertFail, V, Ti, name)
		}
	}
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
	case kind == logosPropOK || kind == logosPropFailNeedsPtr:
		sname := V.String()
		if named != nil {
			sname = named.Obj().Name()
		}
		return fail("%s %s has the fields of %s but does not declare it; write type %s struct : %s { ... }", logosPropRequires, V, Ti, sname, Ti)
	default:
		return fail("%s %s does not implement %s (missing method %s)", logosAssertFail, V, Ti, name)
	}
}

func (check *Checker) logosPropertySelector(x *operand, e *syntax.SelectorExpr, sel string) bool {
	if !check.conf.Logos || x.typ() == nil {
		return false
	}
	s := logosSelectProperty(check.pkg, x.typ(), sel)
	if s.kind == logosSelectOrdinary || x.mode() == typexpr && s.kind != logosSelectInterface {
		return false
	}
	if s.kind == logosSelectAmbiguous {
		check.errorf(e.Sel, AmbiguousSelector, "ambiguous selector %s.%s", x.expr, sel)
		x.invalidate()
		x.expr = e
		return true
	}
	if s.prop.readOnly {
		x.mode_ = value
	} else {
		x.mode_ = variable
	}
	x.typ_ = s.prop.typ
	x.expr = e
	if check.logosPropSel == nil {
		check.logosPropSel = make(map[*syntax.SelectorExpr]bool)
	}
	check.logosPropSel[e] = true
	return true
}

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
