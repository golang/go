// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types_test

import (
	"errors"
	"fmt"
	"go/importer"
	"go/types"
	"strings"
	"testing"
)

func checkMono(t *testing.T, body string) error {
	src := "package x; import `unsafe`; var _ unsafe.Pointer;\n" + body

	var buf strings.Builder
	conf := types.Config{
		Error:    func(err error) { fmt.Fprintln(&buf, err) },
		Importer: importer.Default(),
	}
	typecheck(src, &conf, nil)
	if buf.Len() == 0 {
		return nil
	}
	return errors.New(strings.TrimRight(buf.String(), "\n"))
}

func TestMonoGood(t *testing.T) {
	for i, good := range goods {
		if err := checkMono(t, good); err != nil {
			t.Errorf("%d: unexpected failure: %v", i, err)
		}
	}
}

func TestMonoBad(t *testing.T) {
	for i, bad := range bads {
		if err := checkMono(t, bad); err == nil {
			t.Errorf("%d: unexpected success", i)
		} else {
			t.Log(err)
		}
	}
}

var goods = []string{
	"func F[T any](x T) { F(x) }",
	"func F[T, U, V any]() { F[U, V, T](); F[V, T, U]() }",
	"type Ring[A, B, C any] struct { L *Ring[B, C, A]; R *Ring[C, A, B] }",
	"func F[T any]() { type U[T any] [unsafe.Sizeof(F[*T])]byte }",
	"func F[T any]() { type U[T any] [unsafe.Sizeof(F[*T])]byte; var _ U[int] }",
	"type U[T any] [unsafe.Sizeof(F[*T])]byte; func F[T any]() { var _ U[U[int]] }",
	"func F[T any]() { type A = int; F[A]() }",
}

// TODO(mdempsky): Validate specific error messages and positioning.

var bads = []string{
	"func F[T any](x T) { F(&x) }",
	"func F[T any]() { F[*T]() }",
	"func F[T any]() { F[[]T]() }",
	"func F[T any]() { F[[1]T]() }",
	"func F[T any]() { F[chan T]() }",
	"func F[T any]() { F[map[*T]int]() }",
	"func F[T any]() { F[map[error]T]() }",
	"func F[T any]() { F[func(T)]() }",
	"func F[T any]() { F[func() T]() }",
	"func F[T any]() { F[struct{ t T }]() }",
	"func F[T any]() { F[interface{ t() T }]() }",
	"type U[_ any] int; func F[T any]() { F[U[T]]() }",
	"func F[T any]() { type U int; F[U]() }",
	"func F[T any]() { type U int; F[*U]() }",
	"type U[T any] int; func (U[T]) m() { var _ U[*T] }",
	"type U[T any] int; func (*U[T]) m() { var _ U[*T] }",
	"type U[T1 any] [unsafe.Sizeof(F[*T1])]byte; func F[T2 any]() { var _ U[T2] }",
	"func F[A, B, C, D, E any]() { F[B, C, D, E, *A]() }",
	"type U[_ any] int; const X = unsafe.Sizeof(func() { type A[T any] U[A[*T]] })",
	"func F[T any]() { type A = *T; F[A]() }",
	"type A[T any] struct { _ A[*T] }",
}
