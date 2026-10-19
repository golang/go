// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
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
	check.later(func() {
		check.logosCheckAsserts(e, def, asserts)
	}).describef(e, "check struct assertions of %s", def.name)
	return typ
}

// logosCheckAsserts reports assertion_failed for every asserted type of
// e that is not an interface or that *def does not implement, with the
// cause upstream gives for the missing method.
func (check *Checker) logosCheckAsserts(e *syntax.AssertedStruct, def *TypeName, asserts []Type) {
	ptr := NewPointer(def.Type())
	for i, T := range asserts {
		if !isValid(T) {
			continue
		}
		if !IsInterface(T) || isTypeParam(T) {
			check.errorf(e.Asserts[i], InvalidIfaceAssign, "%s %s is not an interface", logosAssertFail, T)
			continue
		}
		var cause string
		if !check.implements(ptr, T, false, &cause) {
			check.errorf(e.Asserts[i], InvalidIfaceAssign, "%s %s", logosAssertFail, cause)
		}
	}
}
