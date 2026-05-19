// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2_test

import (
	"cmd/compile/internal/types2"
	"testing"
)

func TestIssue74181(t *testing.T) {
	src := `package p

type AB = A[B]

type _ struct {
	_ AB
}

type B struct {
	f *AB
}

type A[T any] struct{}
`

	pkg := mustTypecheck(src, nil, nil)
	b := pkg.Scope().Lookup("B").Type()
	if n, ok := b.(*types2.Named); ok {
		if s, ok := n.Underlying().(*types2.Struct); ok {
			got := s.Field(0).Type()
			want := types2.NewPointer(pkg.Scope().Lookup("AB").Type())
			if !types2.Identical(got, want) {
				t.Errorf("wrong type for f: got %v, want %v", got, want)
			}
			return
		}
	}
	t.Errorf("unexpected type for B: %v", b)
}

func TestPartialTypeCheckUndeclaredAliasPanic(t *testing.T) {
	src := `package p

type A = B // undeclared
`

	pkg, _ := typecheck(src, nil, nil) // don't panic on error
	a := pkg.Scope().Lookup("A").Type()
	if alias, ok := a.(*types2.Alias); ok {
		got := alias.Rhs()
		want := types2.Typ[types2.Invalid]

		if !types2.Identical(got, want) {
			t.Errorf("wrong type for B: got %v, want %v", got, want)
		}
		return
	}
	t.Errorf("unexpected type for A: %v", a)
}
