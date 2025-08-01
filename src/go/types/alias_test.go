// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"testing"
)

func TestIssue74181(t *testing.T) {
	t.Setenv("GODEBUG", "gotypesalias=1")

	src := `package p

type AB = A[B]

type _ struct {
	_ AB
}

type B struct {
	f *AB
}

type A[T any] struct{}`

	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "p.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("could not parse: %v", err)
	}

	conf := types.Config{}
	pkg, err := conf.Check(file.Name.Name, fset, []*ast.File{file}, &types.Info{})
	if err != nil {
		t.Fatalf("could not type check: %v", err)
	}

	b := pkg.Scope().Lookup("B").Type()
	if n, ok := b.(*types.Named); ok {
		if s, ok := n.Underlying().(*types.Struct); ok {
			got := s.Field(0).Type()
			want := types.NewPointer(pkg.Scope().Lookup("AB").Type())
			if !types.Identical(got, want) {
				t.Errorf("wrong type for f: got %v, want %v", got, want)
			}
			return
		}
	}
	t.Errorf("unexpected type for B: %v", b)
}

func TestPartialTypeCheckUndeclaredAliasPanic(t *testing.T) {
	t.Setenv("GODEBUG", "gotypesalias=1")

	src := `package p

type A = B // undeclared`

	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "p.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("could not parse: %v", err)
	}

	conf := types.Config{} // no error handler, panic
	pkg, _ := conf.Check(file.Name.Name, fset, []*ast.File{file}, &types.Info{})
	a := pkg.Scope().Lookup("A").Type()

	if alias, ok := a.(*types.Alias); ok {
		got := alias.Rhs()
		want := types.Typ[types.Invalid]

		if !types.Identical(got, want) {
			t.Errorf("wrong type for B: got %v, want %v", got, want)
		}
		return
	}
	t.Errorf("unexpected type for A: %v", a)
}
