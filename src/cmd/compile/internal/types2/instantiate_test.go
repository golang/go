// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package types2_test

import (
	. "cmd/compile/internal/types2"
	"strings"
	"testing"
)

func TestInstantiateEquality(t *testing.T) {
	const src = genericPkg + "p; type T[P any] int"
	pkg, err := pkgFor(".", src, nil)
	if err != nil {
		t.Fatal(err)
	}
	T := pkg.Scope().Lookup("T").Type().(*Named)
	// Instantiating the same type twice should result in pointer-equivalent
	// instances.
	ctxt := NewContext()
	res1, err := Instantiate(ctxt, T, []Type{Typ[Int]}, false)
	if err != nil {
		t.Fatal(err)
	}
	res2, err := Instantiate(ctxt, T, []Type{Typ[Int]}, false)
	if err != nil {
		t.Fatal(err)
	}
	if res1 != res2 {
		t.Errorf("first instance (%s) not pointer-equivalent to second instance (%s)", res1, res2)
	}
}
func TestInstantiateNonEquality(t *testing.T) {
	const src = genericPkg + "p; type T[P any] int"
	pkg1, err := pkgFor(".", src, nil)
	if err != nil {
		t.Fatal(err)
	}
	pkg2, err := pkgFor(".", src, nil)
	if err != nil {
		t.Fatal(err)
	}
	// We consider T1 and T2 to be distinct types, so their instances should not
	// be deduplicated by the context.
	T1 := pkg1.Scope().Lookup("T").Type().(*Named)
	T2 := pkg2.Scope().Lookup("T").Type().(*Named)
	ctxt := NewContext()
	res1, err := Instantiate(ctxt, T1, []Type{Typ[Int]}, false)
	if err != nil {
		t.Fatal(err)
	}
	res2, err := Instantiate(ctxt, T2, []Type{Typ[Int]}, false)
	if err != nil {
		t.Fatal(err)
	}
	if res1 == res2 {
		t.Errorf("instance from pkg1 (%s) is pointer-equivalent to instance from pkg2 (%s)", res1, res2)
	}
	if Identical(res1, res2) {
		t.Errorf("instance from pkg1 (%s) is identical to instance from pkg2 (%s)", res1, res2)
	}
}

func TestMethodInstantiation(t *testing.T) {
	const prefix = genericPkg + `p

type T[P any] struct{}

var X T[int]

`
	tests := []struct {
		decl string
		want string
	}{
		{"func (r T[P]) m() P", "func (T[int]).m() int"},
		{"func (r T[P]) m(P)", "func (T[int]).m(int)"},
		{"func (r *T[P]) m(P)", "func (*T[int]).m(int)"},
		{"func (r T[P]) m() T[P]", "func (T[int]).m() T[int]"},
		{"func (r T[P]) m(T[P])", "func (T[int]).m(T[int])"},
		{"func (r T[P]) m(T[P], P, string)", "func (T[int]).m(T[int], int, string)"},
		{"func (r T[P]) m(T[P], T[string], T[int])", "func (T[int]).m(T[int], T[string], T[int])"},
	}

	for _, test := range tests {
		src := prefix + test.decl
		pkg, err := pkgFor(".", src, nil)
		if err != nil {
			t.Fatal(err)
		}
		typ := NewPointer(pkg.Scope().Lookup("X").Type())
		obj, _, _ := LookupFieldOrMethod(typ, false, pkg, "m")
		m, _ := obj.(*Func)
		if m == nil {
			t.Fatalf(`LookupFieldOrMethod(%s, "m") = %v, want func m`, typ, obj)
		}
		if got := ObjectString(m, RelativeTo(pkg)); got != test.want {
			t.Errorf("instantiated %q, want %q", got, test.want)
		}
	}
}

func TestImmutableSignatures(t *testing.T) {
	const src = genericPkg + `p

type T[P any] struct{}

func (T[P]) m() {}

var _ T[int]
`
	pkg, err := pkgFor(".", src, nil)
	if err != nil {
		t.Fatal(err)
	}
	typ := pkg.Scope().Lookup("T").Type().(*Named)
	obj, _, _ := LookupFieldOrMethod(typ, false, pkg, "m")
	if obj == nil {
		t.Fatalf(`LookupFieldOrMethod(%s, "m") = %v, want func m`, typ, obj)
	}

	// Verify that the original method is not mutated by instantiating T (this
	// bug manifested when subst did not return a new signature).
	want := "func (T[P]).m()"
	if got := stripAnnotations(ObjectString(obj, RelativeTo(pkg))); got != want {
		t.Errorf("instantiated %q, want %q", got, want)
	}
}

// Copied from errors.go.
func stripAnnotations(s string) string {
	var b strings.Builder
	for _, r := range s {
		// strip #'s and subscript digits
		if r < '₀' || '₀'+10 <= r { // '₀' == U+2080
			b.WriteRune(r)
		}
	}
	if b.Len() < len(s) {
		return b.String()
	}
	return s
}
