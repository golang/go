// run

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	mustPanic(func() {
		f1(1)
	})
	f2(1, 0) // must not panic
	mustPanic(func() {
		f2(1, 2)
	})
}

var v []func()

//go:noinline
func f1(i int) {
	v = make([]func(), -2|i)
}

//go:noinline
func f2(i, j int) {
	if j > 0 {
		v = make([]func(), -2|i)
	}
}

func mustPanic(f func()) {
	defer func() {
		r := recover()
		if r == nil {
			panic("didn't panic")
		}
	}()
	f()
}
