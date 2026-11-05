// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 82865: unsafe.Pointer arithmetic guarded by a nil check must
// not be lifted out of its loop. For a nil base, the lifted value is a
// bad pointer that is live across the calls in the loop.

package main

import "unsafe"

type T struct{ a, b, c, d int }

type U struct {
	T
	s []int
}

// sOf returns the s field of the U that t is part of.
func sOf(t *T) *[]int {
	return (*[]int)(unsafe.Add(unsafe.Pointer(t), unsafe.Sizeof(T{})))
}

//go:noinline
func first(s *[]int) int { return (*s)[0] }

// grow has a large frame, so calling it on a new goroutine grows
// the stack, which checks the pointers in its callers' frames.
//
//go:noinline
func grow(i int) int {
	var buf [16 << 10]byte
	return use(buf[:], i)
}

//go:noinline
func use(b []byte, i int) int { return len(b) + i&1 }

//go:noinline
func sum(t *T, n int) int {
	r := 0
	for i := 0; i < n; i++ {
		if t != nil {
			r += first(sOf(t))
		}
		r += grow(i)
	}
	return r
}

func main() {
	u := &U{s: []int{1}}
	for range 100 {
		done := make(chan int)
		go func() { done <- sum(nil, 2) + sum(&u.T, 2) }()
		<-done
	}
}
