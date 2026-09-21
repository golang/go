// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:noinline
func a1(i int) {
	var a [1]struct{}
	a[i] = struct{}{}
}

//go:noinline
func a2(i int) {
	var a [1][0]int
	a[i] = [0]int{}
}

//go:noinline
func a3(i int) {
	var a [1]struct{ x [0]int }
	a[i] = struct{ x [0]int }{}
}

func wantPanic(name string, f func()) {
	defer func() {
		if r := recover(); r != nil {
			return
		}
		panic(name + ": no panic (bug)")
	}()
	f()
}

func main() {
	wantPanic("a1", func() { a1(5) })
	wantPanic("a2", func() { a2(5) })
	wantPanic("a3", func() { a3(5) })
}
