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
	emptySignature := NewSignatureType(nil, nil, nil, nil, nil, false)
	tests := []struct {
		src       string
		name1     string
		targs1    []Type
		name2     string
		targs2    []Type
		wantEqual bool
	}{
		{
			"package basictype; type T[P any] int",
			"T", []Type{Typ[Int]},
			"T", []Type{Typ[Int]},
			true,
		},
		{
			"package differenttypeargs; type T[P any] int",
			"T", []Type{Typ[Int]},
			"T", []Type{Typ[String]},
			false,
		},
		{
			"package typeslice; type T[P any] int",
			"T", []Type{NewSlice(Typ[Int])},
			"T", []Type{NewSlice(Typ[Int])},
			true,
		},
		{
			// interface{interface{...}} is equivalent to interface{...}
			"package equivalentinterfaces; type T[P any] int",
			"T", []Type{
				NewInterfaceType([]*Func{NewFunc(nopos, nil, "M", emptySignature)}, nil),
			},
			"T", []Type{
				NewInterfaceType(
					nil,
					[]Type{
						NewInterfaceType([]*Func{NewFunc(nopos, nil, "M", emptySignature)}, nil),
					},
				),
			},
			true,
		},
		{
			// int|string is equivalent to string|int
			"package equivalenttypesets; type T[P any] int",
			"T", []Type{
				NewInterfaceType(nil, []Type{
					NewUnion([]*Term{NewTerm(false, Typ[Int]), NewTerm(false, Typ[String])}),
				}),
			},
			"T", []Type{
				NewInterfaceType(nil, []Type{
					NewUnion([]*Term{NewTerm(false, Typ[String]), NewTerm(false, Typ[Int])}),
				}),
			},
			true,
		},
		{
			"package basicfunc; func F[P any]() {}",
			"F", []Type{Typ[Int]},
			"F", []Type{Typ[Int]},
			true,
		},
		{
			"package funcslice; func F[P any]() {}",
			"F", []Type{NewSlice(Typ[Int])},
			"F", []Type{NewSlice(Typ[Int])},
			true,
		},
		{
			"package funcwithparams; func F[P any](x string) float64 { return 0 }",
			"F", []Type{Typ[Int]},
			"F", []Type{Typ[Int]},
			true,
		},
		{
			"package differentfuncargs; func F[P any](x string) float64 { return 0 }",
			"F", []Type{Typ[Int]},
			"F", []Type{Typ[String]},
			false,
		},
		{
			"package funcequality; func F1[P any](x int) {}; func F2[Q any](x int) {}",
			"F1", []Type{Typ[Int]},
			"F2", []Type{Typ[Int]},
			false,
		},
		{
			"package funcsymmetry; func F1[P any](x P) {}; func F2[Q any](x Q) {}",
			"F1", []Type{Typ[Int]},
			"F2", []Type{Typ[Int]},
			false,
		},
	}

	for _, test := range tests {
		pkg := mustTypecheck(test.src, nil, nil)

		t.Run(pkg.Name(), func { t ->
			ctxt := NewContext()

			T1 := pkg.Scope().Lookup(test.name1).Type()
			res1, err := Instantiate(ctxt, T1, test.targs1, false)
			if err != nil {
				t.Fatal(err)
			}

			T2 := pkg.Scope().Lookup(test.name2).Type()
			res2, err := Instantiate(ctxt, T2, test.targs2, false)
			if err != nil {
				t.Fatal(err)
			}

			if gotEqual := res1 == res2; gotEqual != test.wantEqual {
				t.Errorf("%s == %s: %t, want %t", res1, res2, gotEqual, test.wantEqual)
			}
		})
	}
}

func TestInstantiateNonEquality(t *testing.T) {
	const src = "package p; type T[P any] int"
	pkg1 := mustTypecheck(src, nil, nil)
	pkg2 := mustTypecheck(src, nil, nil)
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
	const prefix = `package p

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
		pkg := mustTypecheck(src, nil, nil)
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
	const src = `package p

type T[P any] struct{}

func (T[P]) m() {}

var _ T[int]
`
	pkg := mustTypecheck(src, nil, nil)
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
	var buf strings.Builder
	for _, r := range s {
		// strip #'s and subscript digits
		if r < '₀' || '₀'+10 <= r { // '₀' == U+2080
			buf.WriteRune(r)
		}
	}
	if buf.Len() < len(s) {
		return buf.String()
	}
	return s
}
