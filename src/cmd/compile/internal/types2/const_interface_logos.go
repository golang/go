// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

func logosInterfaceHasReadonly(T *Interface) bool {
	if T == nil {
		return false
	}
	for _, m := range T.typeSet().methods {
		if m.LogosReadonly() {
			return true
		}
	}
	return false
}

func logosReadonlyObligation(m *Func) bool {
	return m != nil && m.LogosReadonly()
}

func logosInterfaceReadonlyMismatch(f, m *Func) bool {
	if !logosReadonlyObligation(m) {
		return false
	}
	return !logosReadonlyObligation(f)
}

func logosQualifiedIface(T Type) *Interface {
	if T == nil {
		return nil
	}
	u, _ := Unalias(T).Underlying().(*Interface)
	if u != nil && logosInterfaceHasReadonly(u) {
		return u
	}
	return nil
}

func logosAssertReadonlyGap(V, T Type) (*Func, *Func) {
	tu, _ := T.Underlying().(*Interface)
	if tu == nil {
		return nil, nil
	}
	var vset *_TypeSet
	for _, m := range tu.typeSet().methods {
		ro := logosReadonlyObligation(m)
		if !ro && !logosSigErasableQualified(m.typ, nil) {
			continue
		}
		if vset == nil {
			vu, _ := V.Underlying().(*Interface)
			if vu == nil {
				return m, nil
			}
			vset = vu.typeSet()
		}
		_, f := vset.LookupMethod(m.pkg, m.name, false)
		if f == nil || !Identical(f.typ, m.typ) || (ro && !logosReadonlyObligation(f)) {
			return m, f
		}
	}
	return nil, nil
}

func (check *Checker) logosAssertReadonly(V, T Type, cause *string) bool {
	if check != nil {
		if logosContainsTypeParam(T, nil) {
			return true
		}
	}
	m, f := logosAssertReadonlyGap(V, T)
	if m == nil {
		return true
	}
	if cause != nil {
		switch {
		case f == nil:
			*cause = check.sprintf("(readonly method %s is not declared by %s)", m.Name(), V)
		case !logosReadonlyObligation(f):
			*cause = check.sprintf("(method %s is not readonly in %s)", m.Name(), V)
		default:
			*cause = check.sprintf("(readonly method %s has a different signature in %s)", m.Name(), V)
		}
	}
	return false
}

func (check *Checker) logosAssertReadonlyError(e syntax.Expr, x *operand, T Type, typeSwitch bool, cause string) bool {
	if !IsInterface(T) {
		if !logosErasableQualified(T, nil) {
			return false
		}
		what := "type assertion"
		if typeSwitch {
			what = "type switch case"
		}
		check.errorf(e, ImpossibleAssert, "%s unchecked qualified %s: %s\n\t%s does not guarantee an identity-checkable %s", logosConstMutation, what, e, x, T)
		return true
	}
	what := "type assertion"
	if typeSwitch {
		what = "type switch case"
	}
	check.errorf(e, ImpossibleAssert, "%s unchecked qualified %s: %s\n\t%s does not guarantee %s %s", logosConstMutation, what, e, x, T, cause)
	return true
}

func (check *Checker) logosConcreteReadonlySatisfiedAt(V Type, f, m *Func, pos syntax.Pos) bool {
	if !logosReadonlyObligation(m) {
		return true
	}
	if f == nil {
		return false
	}
	if check == nil {
		return logosConcreteReadonlySatisfiedNil(f, m)
	}
	return check.logosReadonlyProofAt(V, f, m.Name(), pos)
}

func (check *Checker) logosReadonlyProofAt(V Type, f *Func, obligation string, pos syntax.Pos) bool {
	sig, ok := f.Type().(*Signature)
	if !ok || sig.Recv() == nil {
		return false
	}
	rt := sig.Recv().Type()
	local := f.Pkg() == check.pkg
	if !local {
		return check.logosImportedReadonly(f)
	}
	if logosAsConstView(rt) != nil {
		f.SetLogosReadonly(true)
		return true
	}
	if logosReferenceFree(rt) {
		f.SetLogosReadonly(true)
		return true
	}
	check.needsCleanup(&logosSatisfyReadonlyDeferred{check: check, V: V, f: f, obligation: obligation, pos: pos})
	return true
}

type logosSatisfyReadonlyDeferred struct {
	check      *Checker
	V          Type
	f          *Func
	obligation string
	pos        syntax.Pos
}

func (d *logosSatisfyReadonlyDeferred) cleanup() {
	check := d.check
	if check == nil || d.f == nil {
		return
	}
	if d.f.LogosReadonly() {
		return
	}
	if check.logosReceiverReadonly(d.f) {
		d.f.SetLogosReadonly(true)
		return
	}
	pos := d.pos
	if !pos.IsKnown() {
		pos = d.f.Pos()
	}
	var at poser = noposer{pos: pos}
	check.errorf(at, IncompatibleAssign, "%s %s does not implement readonly method %s (method %s is not readonly)", logosConstMutation, d.V, d.obligation, d.f.Name())
}

type noposer struct {
	pos syntax.Pos
}

func (n noposer) Pos() syntax.Pos { return n.pos }

func logosEmbedReadonlyConflict(a, b *Func) bool {
	if a == nil || b == nil {
		return false
	}
	return a.LogosReadonly() != b.LogosReadonly()
}
