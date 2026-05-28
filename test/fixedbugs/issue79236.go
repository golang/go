// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:noinline
func a1(i int) {
	var a [2][0]int
	a[i] = [0]int{}
}

//go:noinline
func a2(i int) int {
	var a [0][2]int
	return a[i][0]
}

//go:noinline
func a3(i int) {
	var a [0][2]int
	a[i][0] = 1
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
	wantPanic("a2", func() { _ = a2(5) })
	wantPanic("a3", func() { a3(5) })
}
