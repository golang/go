// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:noinline
func id[T any](x T) T { return x }

//go:noinline
func stale[T any](p *T, v T) T {
	q := id[*T](p)

	*q = v

	var zero T
	*p = zero

	return *q
}

var a, b = 7, 9

func main() {
	p := &a
	r := stale[*int](&p, &b)
	if p != nil {
		panic("p is not nil")
	}
	if r != nil {
		panic("r is not nil")
	}
}
