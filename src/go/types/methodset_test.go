// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types_test

import (
	"testing"

	. "go/types"
)

func TestNewMethodSet(t *testing.T) {
	type method struct {
		name     string
		index    []int
		indirect bool
	}

	// Tests are expressed src -> methods, for simplifying the composite literal.
	// Should be kept in sync with TestLookupFieldOrMethod.
	tests := map[string][]method{
		// Named types
		"var a T; type T struct{}; func (T) f() {}":   {{"f", []int{0}, false}},
		"var a *T; type T struct{}; func (T) f() {}":  {{"f", []int{0}, true}},
		"var a T; type T struct{}; func (*T) f() {}":  {},
		"var a *T; type T struct{}; func (*T) f() {}": {{"f", []int{0}, true}},

		// Interfaces
		"var a T; type T interface{ f() }":                           {{"f", []int{0}, true}},
		"var a T1; type ( T1 T2; T2 interface{ f() } )":              {{"f", []int{0}, true}},
		"var a T1; type ( T1 interface{ T2 }; T2 interface{ f() } )": {{"f", []int{0}, true}},

		// Embedding
		"var a struct{ E }; type E interface{ f() }":            {{"f", []int{0, 0}, true}},
		"var a *struct{ E }; type E interface{ f() }":           {{"f", []int{0, 0}, true}},
		"var a struct{ E }; type E struct{}; func (E) f() {}":   {{"f", []int{0, 0}, false}},
		"var a struct{ *E }; type E struct{}; func (E) f() {}":  {{"f", []int{0, 0}, true}},
		"var a struct{ E }; type E struct{}; func (*E) f() {}":  {},
		"var a struct{ *E }; type E struct{}; func (*E) f() {}": {{"f", []int{0, 0}, true}},

		// collisions
		"var a struct{ E1; *E2 }; type ( E1 interface{ f() }; E2 struct{ f int })":            {},
		"var a struct{ E1; *E2 }; type ( E1 struct{ f int }; E2 struct{} ); func (E2) f() {}": {},
	}

	genericTests := map[string][]method{
		// By convention, look up a in the scope of "g"
		"type C interface{ f() }; func g[T C](a T){}":               {{"f", []int{0}, true}},
		"type C interface{ f() }; func g[T C]() { var a T; _ = a }": {{"f", []int{0}, true}},

		// Issue #43621: We don't allow this anymore. Keep this code in case we
		// decide to revisit this decision.
		// "type C interface{ f() }; func g[T C]() { var a struct{T}; _ = a }": {{"f", []int{0, 0}, true}},

		// Issue #45639: We also don't allow this anymore.
		// "type C interface{ f() }; func g[T C]() { type Y T; var a Y; _ = a }": {},
	}

	check := func(src string, methods []method, generic bool) {
		pkgName := "p"
		if generic {
			// The generic_ prefix causes pkgFor to allow generic code.
			pkgName = "generic_p"
		}
		pkg, err := pkgFor("test", "package "+pkgName+";"+src, nil)
		if err != nil {
			t.Errorf("%s: incorrect test case: %s", src, err)
			return
		}

		scope := pkg.Scope()
		if generic {
			fn := pkg.Scope().Lookup("g").(*Func)
			scope = fn.Scope()
		}
		obj := scope.Lookup("a")
		if obj == nil {
			t.Errorf("%s: incorrect test case - no object a", src)
			return
		}

		ms := NewMethodSet(obj.Type())
		if got, want := ms.Len(), len(methods); got != want {
			t.Errorf("%s: got %d methods, want %d", src, got, want)
			return
		}
		for i, m := range methods {
			sel := ms.At(i)
			if got, want := sel.Obj().Name(), m.name; got != want {
				t.Errorf("%s [method %d]: got name = %q at, want %q", src, i, got, want)
			}
			if got, want := sel.Index(), m.index; !sameSlice(got, want) {
				t.Errorf("%s [method %d]: got index = %v, want %v", src, i, got, want)
			}
			if got, want := sel.Indirect(), m.indirect; got != want {
				t.Errorf("%s [method %d]: got indirect = %v, want %v", src, i, got, want)
			}
		}
	}

	for src, methods := range tests {
		check(src, methods, false)
	}

	for src, methods := range genericTests {
		check(src, methods, true)
	}
}
