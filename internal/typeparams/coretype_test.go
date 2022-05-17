// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeparams_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"testing"

	"golang.org/x/tools/internal/typeparams"
)

func TestCoreType(t *testing.T) {
	if !typeparams.Enabled {
		t.Skip("TestCoreType requires type parameters.")
	}

	const source = `
	package P

	type Named int

	type A any
	type B interface{~int}
	type C interface{int}
	type D interface{Named}
	type E interface{~int|interface{Named}}
	type F interface{~int|~float32}
	type G interface{chan int|interface{chan int}}
	type H interface{chan int|chan float32}
	type I interface{chan<- int|chan int}
	type J interface{chan int|chan<- int}
	type K interface{<-chan int|chan int}
	type L interface{chan int|<-chan int}
	type M interface{chan int|chan Named}
	type N interface{<-chan int|chan<- int}
	type O interface{chan int|bool}
	type P struct{ Named }
	type Q interface{ Foo() }
	type R interface{ Foo() ; Named }
	type S interface{ Foo() ; ~int }

	type T interface{chan int|interface{chan int}|<-chan int}
`

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "hello.go", source, 0)
	if err != nil {
		t.Fatal(err)
	}

	var conf types.Config
	pkg, err := conf.Check("P", fset, []*ast.File{f}, nil)
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range []struct {
		expr string // type expression of Named type
		want string // expected core type (or "<nil>" if none)
	}{
		{"Named", "int"},         // Underlying type is not interface.
		{"A", "<nil>"},           // Interface has no terms.
		{"B", "int"},             // Tilde term.
		{"C", "int"},             // Non-tilde term.
		{"D", "int"},             // Named term.
		{"E", "int"},             // Identical underlying types.
		{"F", "<nil>"},           // Differing underlying types.
		{"G", "chan int"},        // Identical Element types.
		{"H", "<nil>"},           // Element type int has differing underlying type to float32.
		{"I", "chan<- int"},      // SendRecv followed by SendOnly
		{"J", "chan<- int"},      // SendOnly followed by SendRecv
		{"K", "<-chan int"},      // RecvOnly followed by SendRecv
		{"L", "<-chan int"},      // SendRecv followed by RecvOnly
		{"M", "<nil>"},           // Element type int is not *identical* to Named.
		{"N", "<nil>"},           // Differing channel directions
		{"O", "<nil>"},           // A channel followed by a non-channel.
		{"P", "struct{P.Named}"}, // Embedded type.
		{"Q", "<nil>"},           // interface type with no terms and functions
		{"R", "int"},             // interface type with both terms and functions.
		{"S", "int"},             // interface type with a tilde term
		{"T", "<-chan int"},      // Prefix of 2 terms that are identical before switching to channel.
	} {
		// Eval() expr for its type.
		tv, err := types.Eval(fset, pkg, 0, test.expr)
		if err != nil {
			t.Fatalf("Eval(%s) failed: %v", test.expr, err)
		}

		ct := typeparams.CoreType(tv.Type)
		var got string
		if ct == nil {
			got = "<nil>"
		} else {
			got = ct.String()
		}
		if got != test.want {
			t.Errorf("coreType(%s) = %v, want %v", test.expr, got, test.want)
		}
	}
}
