// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "runtime"

type T struct {
	pad [3]int
	x   [4096]uint64
}

var sink *T

//go:noinline
func call(t *T) { runtime.GC() }

//go:noinline
func zero(t *T) {
	t.x = [4096]uint64{}
	call(t)
	t.x = [4096]uint64{}
}

//go:noinline
func move(t *T, a, b *[4096]uint64) {
	*a = t.x
	call(t)
	*b = t.x
}

func main() {
	for i := 0; i < 20; i++ {
		t := new(T)
		sink = t
		zero(t)
		move(t, new([4096]uint64), new([4096]uint64))
	}
}
