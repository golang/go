// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2_test

import (
	"cmd/compile/internal/types2"
	"sync"
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

func TestIssue81138(t *testing.T) {
	// type A[T any] = func(T)
	T := types2.NewTypeParam(types2.NewTypeName(nopos, nil, "T", nil), types2.Universe.Lookup("any").Type())
	rhs := types2.NewSignatureType(nil, nil, nil,
		types2.NewTuple(types2.NewParam(nopos, nil, "", T)), nil, false)
	A := types2.NewAlias(types2.NewTypeName(nopos, nil, "A", nil), rhs)
	A.SetTypeParams([]*types2.TypeParam{T})

	// Check calling Unalias concurrently on a newly instantiated alias
	// does not race on Alias.actual or observe a partially written interface
	// value (a non-nil itab with a nil data pointer, i.e. (*Signature)(nil)).
	for i := range 1000 {
		inst, err := types2.Instantiate(types2.NewContext(), A, []types2.Type{types2.Typ[types2.Int]}, false)
		if err != nil {
			t.Fatal(err)
		}

		ch := make(chan types2.Type, 1)
		go func() { ch <- types2.Unalias(inst) }()
		for _, u := range []types2.Type{types2.Unalias(inst), <-ch} {
			if sig, ok := u.(*types2.Signature); ok && sig == nil {
				t.Fatalf("round %d: Unalias returned (*Signature)(nil)", i)
			}
		}
	}

	// Incomplete aliases must not race or overwrite actual when Unalias is called concurrently.
	incomplete := types2.NewAlias(types2.NewTypeName(nopos, nil, "Incomplete", nil), nil)
	var wg sync.WaitGroup
	for range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if got := types2.Unalias(incomplete); got != nil {
				t.Errorf("Unalias(incomplete) = %v, want nil", got)
			}
		}()
	}
	wg.Wait()
}
