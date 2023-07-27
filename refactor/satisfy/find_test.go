// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package satisfy_test

import (
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"reflect"
	"sort"
	"testing"

	"golang.org/x/tools/internal/typeparams"
	"golang.org/x/tools/refactor/satisfy"
)

// This test exercises various operations on core types of type parameters.
// (It also provides pretty decent coverage of the non-generic operations.)
func TestGenericCoreOperations(t *testing.T) {
	if !typeparams.Enabled {
		t.Skip("!typeparams.Enabled")
	}

	const src = `package foo

import "unsafe"

type I interface { f() }

type impl struct{}
func (impl) f() {}

// A big pile of single-serving types that implement I.
type A struct{impl}
type B struct{impl}
type C struct{impl}
type D struct{impl}
type E struct{impl}
type F struct{impl}
type G struct{impl}
type H struct{impl}
type J struct{impl}
type K struct{impl}
type L struct{impl}
type M struct{impl}
type N struct{impl}
type O struct{impl}
type P struct{impl}
type Q struct{impl}
type R struct{impl}
type S struct{impl}
type T struct{impl}
type U struct{impl}
type V struct{impl}
type W struct{impl}
type X struct{impl}

type Generic[T any] struct{impl}
func (Generic[T]) g(T) {}

type GI[T any] interface{
	g(T)
}

func _[Slice interface{ []I }](s Slice) Slice {
	s[0] = L{} // I <- L
	return append(s, A{}) // I <- A
}

func _[Func interface{ func(I) B }](fn Func) {
	b := fn(C{}) // I <- C
	var _ I = b // I <- B
}

func _[Chan interface{ chan D }](ch Chan) {
	var i I
	for i = range ch {} // I <- D
	_ = i
}

func _[Chan interface{ chan E }](ch Chan) {
	var _ I = <-ch // I <- E
}

func _[Chan interface{ chan I }](ch Chan) {
	ch <- F{} // I <- F
}

func _[Map interface{ map[G]H }](m Map) {
	var k, v I
	for k, v = range m {} // I <- G, I <- H
	_, _ = k, v
}

func _[Map interface{ map[I]K }](m Map) {
	var _ I = m[J{}] // I <- J, I <- K
	delete(m, R{}) // I <- R
	_, _ = m[J{}]
}

func _[Array interface{ [1]I }](a Array) {
	a[0] = M{} // I <- M
}

func _[Array interface{ [1]N }](a Array) {
	var _ I = a[0] // I <- N
}

func _[Array interface{ [1]O }](a Array) {
	var v I
	for _, v = range a {} // I <- O
	_ = v
}

func _[ArrayPtr interface{ *[1]P }](a ArrayPtr) {
	var v I
	for _, v = range a {} // I <- P
	_ = v
}

func _[Slice interface{ []Q }](s Slice) {
	var v I
	for _, v = range s {} // I <- Q
	_ = v
}

func _[Func interface{ func() (S, bool) }](fn Func) {
	var i I
	i, _ = fn() // I <- S
	_ = i
}

func _() I {
	var _ I = T{} // I <- T
	var _ I = Generic[T]{} // I <- Generic[T]
	var _ I = Generic[string]{} // I <- Generic[string]
	return U{} // I <- U
}

var _ GI[string] = Generic[string]{} //  GI[string] <- Generic[string]

// universally quantified constraints:
// the type parameter may appear on the left, the right, or both sides.

func  _[T any](g Generic[T]) GI[T] {
	return g // GI[T] <- Generic[T]
}

func  _[T any]() {
	type GI2[T any] interface{ g(string) }
	var _ GI2[T] = Generic[string]{} // GI2[T] <- Generic[string]
}

type Gen2[T any] struct{}
func (f Gen2[T]) g(string) { global = f } // GI[string] <- Gen2[T]

var global GI[string]

func _() {
	var x [3]V
	// golang/go#56227: the finder should visit calls in the unsafe package.
	_ = unsafe.Slice(&x[0], func() int { var _ I = x[0]; return 3 }()) // I <- V
}

func _[P ~struct{F I}]() {
	_ = P{W{}}
	_ = P{F: X{}}
}
`
	got := constraints(t, src)
	want := []string{
		"p.GI2[T] <- p.Generic[string]", // implicitly "forall T" quantified
		"p.GI[T] <- p.Generic[T]",       // implicitly "forall T" quantified
		"p.GI[string] <- p.Gen2[T]",     // implicitly "forall T" quantified
		"p.GI[string] <- p.Generic[string]",
		"p.I <- p.A",
		"p.I <- p.B",
		"p.I <- p.C",
		"p.I <- p.D",
		"p.I <- p.E",
		"p.I <- p.F",
		"p.I <- p.G",
		"p.I <- p.Generic[p.T]",
		"p.I <- p.Generic[string]",
		"p.I <- p.H",
		"p.I <- p.J",
		"p.I <- p.K",
		"p.I <- p.L",
		"p.I <- p.M",
		"p.I <- p.N",
		"p.I <- p.O",
		"p.I <- p.P",
		"p.I <- p.Q",
		"p.I <- p.R",
		"p.I <- p.S",
		"p.I <- p.T",
		"p.I <- p.U",
		"p.I <- p.V",
		"p.I <- p.W",
		"p.I <- p.X",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("found unexpected constraints: got %s, want %s", got, want)
	}
}

func constraints(t *testing.T, src string) []string {
	// parse
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "p.go", src, 0)
	if err != nil {
		t.Fatal(err) // parse error
	}
	files := []*ast.File{f}

	// type-check
	info := &types.Info{
		Types:      make(map[ast.Expr]types.TypeAndValue),
		Defs:       make(map[*ast.Ident]types.Object),
		Uses:       make(map[*ast.Ident]types.Object),
		Implicits:  make(map[ast.Node]types.Object),
		Scopes:     make(map[ast.Node]*types.Scope),
		Selections: make(map[*ast.SelectorExpr]*types.Selection),
	}
	typeparams.InitInstanceInfo(info)
	conf := types.Config{
		Importer: importer.Default(),
	}
	if _, err := conf.Check("p", fset, files, info); err != nil {
		t.Fatal(err) // type error
	}

	// gather constraints
	var finder satisfy.Finder
	finder.Find(info, files)
	var constraints []string
	for c := range finder.Result {
		constraints = append(constraints, fmt.Sprintf("%v <- %v", c.LHS, c.RHS))
	}
	sort.Strings(constraints)
	return constraints
}
