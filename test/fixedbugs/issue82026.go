// run -gcflags=-clobberdead

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:noinline
func use(string) {}

//go:noinline
func f(i, j int) string {
	var a [2][2][2]string
	a[1][1][0], a[1][1][1] = "x", "y"
	use(a[1][i][j])
	return a[1][i][j]
}

//go:noinline
func g(a [2][2]string, i, j int) string {
	use(a[i][j])
	return a[i][j]
}

func main() {
	if s := f(1, 1); s != "y" {
		panic("f: " + s)
	}
	if s := g([2][2]string{{"w", "x"}, {"y", "z"}}, 1, 1); s != "z" {
		panic("g: " + s)
	}
}
