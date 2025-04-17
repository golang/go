// run

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "runtime"

const C = 16

type T [C * C]byte

func main() {
	var ts []*T

	for i := 0; i < 100; i++ {
		t := new(T)
		// Save every even object.
		if i%2 == 0 {
			ts = append(ts, t)
		}
	}
	// Make sure the odd objects are collected.
	runtime.GC()

	for _, t := range ts {
		f(t, C, C)
	}
}

//go:noinline
func f(t *T, i, j uint) {
	if i == 0 || i > C || j == 0 || j > C {
		return // gets rid of bounds check below (via prove pass)
	}
	p := &t[i*j-1]
	*p = 0
	runtime.GC()
	*p = 0

	// This goes badly if compiled to
	//   q := &t[i*j]
	//   *(q-1) = 0
	//   runtime.GC()
	//   *(q-1) = 0
	// as at the GC call, q is an invalid pointer
	// (it points past the end of t's allocation).
}
