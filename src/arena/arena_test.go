// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.arenas

package arena_test

import (
	"arena"
	"testing"
)

type T1 struct {
	n int
}
type T2 [1 << 20]byte // 1MiB

func TestSmoke(t *testing.T) {
	a := arena.NewArena()
	defer a.Free()

	tt := arena.New[T1](a)
	tt.n = 1

	ts := arena.MakeSlice[T1](a, 99, 100)
	if len(ts) != 99 {
		t.Errorf("Slice() len = %d, want 99", len(ts))
	}
	if cap(ts) != 100 {
		t.Errorf("Slice() cap = %d, want 100", cap(ts))
	}
	ts[1].n = 42
}

func TestSmokeLarge(t *testing.T) {
	a := arena.NewArena()
	defer a.Free()
	for i := 0; i < 10*64; i++ {
		_ = arena.New[T2](a)
	}
}
