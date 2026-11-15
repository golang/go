// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
	"maps"
	"slices"
)

func (check *Checker) logosAssertedStruct(e *syntax.AssertedStruct, def *TypeName) Type {
	typ := new(Struct)
	check.structType(typ, e.Struct)
	check.logosStructDefaults(e.Struct, typ)
	check.recordTypeAndValue(e.Struct, typexpr, typ, nil)

	if !check.conf.Logos {
		check.error(e, InvalidSyntaxTree, "invalid use of struct assertion outside logos mode")
		return typ
	}
	if def == nil {
		check.error(e, InvalidSyntaxTree, "struct assertion needs a type declaration")
		return typ
	}

	asserts := make([]Type, len(e.Asserts))
	for i, a := range e.Asserts {
		asserts[i] = check.typ(a)
	}
	if check.conf.Logos {
		check.logosRecordConforms(def, asserts)
	}
	check.later(func() {
		check.logosCheckAsserts(e, def, asserts)
	}).describef(e, "check struct assertions of %s", def.name)
	return typ
}

func LogosSelfInstance(named *Named) Type {
	list := named.TypeParams().list()
	if len(list) == 0 {
		return named
	}
	targs := make([]Type, len(list))
	for i, tpar := range list {
		targs[i] = tpar
	}
	return (*Checker)(nil).instance(nopos, named, targs, nil, NewContext())
}

func (check *Checker) logosCheckAsserts(e *syntax.AssertedStruct, def *TypeName, asserts []Type) {
	ptr := NewPointer(LogosSelfInstance(asNamed(def.Type())))
	var merged map[string]logosMergedProp
	propErr := false
	for i, T := range asserts {
		if !isValid(T) {
			continue
		}
		if !IsInterface(T) || isTypeParam(T) {
			check.errorf(e.Asserts[i], InvalidIfaceAssign, "%s %s is not an interface", logosAssertFail, T)
			propErr = true
			continue
		}
		if check.conf.Logos && logosPropertyInterface(T) != nil {
			if merged == nil {
				merged = make(map[string]logosMergedProp)
			}
			if check.logosCheckPropertyAssert(e, T, i, merged) {
				propErr = true
			}
			continue
		}
		var cause string
		if !check.implements(ptr, T, false, &cause) {
			check.errorf(e.Asserts[i], InvalidIfaceAssign, "%s %s", logosAssertFail, cause)
		}
	}
	if propErr || len(merged) == 0 {
		return
	}
	for _, name := range slices.Sorted(maps.Keys(merged)) {
		if check.logosCheckMergedProp(name, ptr, merged[name]) {
			return
		}
	}
	for i, T := range asserts {
		iface := logosPropertyInterface(T)
		if iface == nil {
			continue
		}
		if name := check.logosMissingOrdinary(ptr, iface, e.Asserts[i].Pos()); name != "" {
			check.errorf(e.Asserts[i], InvalidIfaceAssign, "%s %s does not implement %s (missing method %s)", logosAssertFail, ptr, T, name)
			return
		}
		if name := check.logosMutatingGetter(ptr, iface, e.Asserts[i].Pos()); name != "" {
			check.errorf(e.Asserts[i], InvalidIfaceAssign, "%s %s: the getter of const property %s is not proven readonly", logosConstMutation, ptr, name)
			return
		}
	}
}
