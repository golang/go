// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types_test

import (
	"strings"
	"testing"

	"go/ast"
	"go/parser"
	"go/token"
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

		// Generic named types
		"var a T[int]; type T[P any] struct{}; func (T[P]) f() {}":   {{"f", []int{0}, false}},
		"var a *T[int]; type T[P any] struct{}; func (T[P]) f() {}":  {{"f", []int{0}, true}},
		"var a T[int]; type T[P any] struct{}; func (*T[P]) f() {}":  {},
		"var a *T[int]; type T[P any] struct{}; func (*T[P]) f() {}": {{"f", []int{0}, true}},

		// Interfaces
		"var a T; type T interface{ f() }":                           {{"f", []int{0}, true}},
		"var a T1; type ( T1 T2; T2 interface{ f() } )":              {{"f", []int{0}, true}},
		"var a T1; type ( T1 interface{ T2 }; T2 interface{ f() } )": {{"f", []int{0}, true}},

		// Generic interfaces
		"var a T[int]; type T[P any] interface{ f() }":                                     {{"f", []int{0}, true}},
		"var a T1[int]; type ( T1[P any] T2[P]; T2[P any] interface{ f() } )":              {{"f", []int{0}, true}},
		"var a T1[int]; type ( T1[P any] interface{ T2[P] }; T2[P any] interface{ f() } )": {{"f", []int{0}, true}},

		// Embedding
		"var a struct{ E }; type E interface{ f() }":            {{"f", []int{0, 0}, true}},
		"var a *struct{ E }; type E interface{ f() }":           {{"f", []int{0, 0}, true}},
		"var a struct{ E }; type E struct{}; func (E) f() {}":   {{"f", []int{0, 0}, false}},
		"var a struct{ *E }; type E struct{}; func (E) f() {}":  {{"f", []int{0, 0}, true}},
		"var a struct{ E }; type E struct{}; func (*E) f() {}":  {},
		"var a struct{ *E }; type E struct{}; func (*E) f() {}": {{"f", []int{0, 0}, true}},

		// Embedding of generic types
		"var a struct{ E[int] }; type E[P any] interface{ f() }":               {{"f", []int{0, 0}, true}},
		"var a *struct{ E[int] }; type E[P any] interface{ f() }":              {{"f", []int{0, 0}, true}},
		"var a struct{ E[int] }; type E[P any] struct{}; func (E[P]) f() {}":   {{"f", []int{0, 0}, false}},
		"var a struct{ *E[int] }; type E[P any] struct{}; func (E[P]) f() {}":  {{"f", []int{0, 0}, true}},
		"var a struct{ E[int] }; type E[P any] struct{}; func (*E[P]) f() {}":  {},
		"var a struct{ *E[int] }; type E[P any] struct{}; func (*E[P]) f() {}": {{"f", []int{0, 0}, true}},

		// collisions
		"var a struct{ E1; *E2 }; type ( E1 interface{ f() }; E2 struct{ f int })":            {},
		"var a struct{ E1; *E2 }; type ( E1 struct{ f int }; E2 struct{} ); func (E2) f() {}": {},

		// recursive generic types; see go.dev/issue/52715
		"var a T[int]; type ( T[P any] struct { *N[P] }; N[P any] struct { *T[P] } ); func (N[P]) m() {}": {{"m", []int{0, 0}, true}},
		"var a T[int]; type ( T[P any] struct { *N[P] }; N[P any] struct { *T[P] } ); func (T[P]) m() {}": {{"m", []int{0}, false}},
	}

	tParamTests := map[string][]method{
		// By convention, look up a in the scope of "g"
		"type C interface{ f() }; func g[T C](a T){}":               {{"f", []int{0}, true}},
		"type C interface{ f() }; func g[T C]() { var a T; _ = a }": {{"f", []int{0}, true}},

		// go.dev/issue/43621: We don't allow this anymore. Keep this code in case we
		// decide to revisit this decision.
		// "type C interface{ f() }; func g[T C]() { var a struct{T}; _ = a }": {{"f", []int{0, 0}, true}},

		// go.dev/issue/45639: We also don't allow this anymore.
		// "type C interface{ f() }; func g[T C]() { type Y T; var a Y; _ = a }": {},
	}

	check := func(src string, methods []method, generic bool) {
		pkg := mustTypecheck("package p;"+src, nil, nil)

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

	for src, methods := range tParamTests {
		check(src, methods, true)
	}
}

// Test for go.dev/issue/52715
func TestNewMethodSet_RecursiveGeneric(t *testing.T) {
	const src = `
package pkg

type Tree[T any] struct {
	*Node[T]
}

type Node[T any] struct {
	*Tree[T]
}

type Instance = *Tree[int]
`

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "foo.go", src, 0)
	if err != nil {
		panic(err)
	}
	pkg := NewPackage("pkg", f.Name.Name)
	if err := NewChecker(nil, fset, pkg, nil).Files([]*ast.File{f}); err != nil {
		panic(err)
	}

	T := pkg.Scope().Lookup("Instance").Type()
	_ = NewMethodSet(T) // verify that NewMethodSet terminates
}

func TestIssue60634(t *testing.T) {
	const src = `
package p
type T *int
func (T) m() {} // expected error: invalid receiver type
`

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "p.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}

	var conf Config
	pkg, err := conf.Check("p", fset, []*ast.File{f}, nil)
	if err == nil || !strings.Contains(err.Error(), "invalid receiver type") {
		t.Fatalf("missing or unexpected error: %v", err)
	}

	// look up T.m and (*T).m
	T := pkg.Scope().Lookup("T").Type()
	name := "m"
	for _, recv := range []Type{T, NewPointer(T)} {
		// LookupFieldOrMethod and NewMethodSet must match:
		// either both find m or neither finds it.
		obj1, _, _ := LookupFieldOrMethod(recv, false, pkg, name)
		mset := NewMethodSet(recv)
		if (obj1 != nil) != (mset.Len() == 1) {
			t.Fatalf("lookup(%v.%s): got obj = %v, mset = %v", recv, name, obj1, mset)
		}
		// If the method exists, both must return the same object.
		if obj1 != nil {
			obj2 := mset.At(0).Obj()
			if obj1 != obj2 {
				t.Fatalf("%v != %v", obj1, obj2)
			}
		}
	}
}
