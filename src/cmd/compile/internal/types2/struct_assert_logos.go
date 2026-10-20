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

// LoGo/s M2-S6 rule for inline interface assertions (SPEC Sec 2.3,
// T-203.3): `type S struct : I, J { ... }` asserts at the declaration
// that *S implements I and J. typInternal calls in through one hook
// line. Lowering emits `var _ I = (*S)(nil)` for each assertion, so the
// backend sees ordinary Go.

// logosAssertedStruct type-checks the struct of the declaration def and
// schedules the implements check of every asserted interface. The
// interface types resolve now, in the scope of the declaration; the
// check runs after every method of the package is collected.
func (check *Checker) logosAssertedStruct(e *syntax.AssertedStruct, def *TypeName) Type {
	typ := new(Struct)
	check.structType(typ, e.Struct)
	check.logosStructDefaults(e.Struct, typ) // logos M6-S3 hook: `F T = const` defaults of asserted structs.
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

// logosCheckAsserts reports assertion_failed for every asserted type of
// e that is not an interface or that *def does not implement, with the
// cause upstream gives for the missing method. Asserted property
// interfaces (M4-S1) take the conformance path instead: their properties
// merge across the assert list (conflicts report property_conflict),
// then each merged property needs a field or method source on *def and
// each ordinary method must be implemented.
func (check *Checker) logosCheckAsserts(e *syntax.AssertedStruct, def *TypeName, asserts []Type) {
	ptr := NewPointer(def.Type())
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
		if name := check.logosMissingOrdinary(ptr, iface); name != "" {
			check.errorf(e.Asserts[i], InvalidIfaceAssign, "%s %s does not implement %s (missing method %s)", logosAssertFail, ptr, T, name)
			return
		}
	}
}
