// run

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure address calculations don't float up before
// the corresponding nil check.

package main

type T struct {
	a, b int
}

//go:noinline
func f(x *T, p *bool, n int) {
	*p = n != 0
	useStack(1000)
	g(&x.b)
}

//go:noinline
func g(p *int) {
}

func useStack(n int) {
	if n == 0 {
		return
	}
	useStack(n - 1)
}

func main() {
	mustPanic(func() {
		var b bool
		f(nil, &b, 3)
	})
}

func mustPanic(f func()) {
	defer func() {
		if recover() == nil {
			panic("expected panic, got nil")
		}
	}()
	f()
}
