// run

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Smoke test for reverse type inference.
// The type checker has more expansive tests.

package main

func main() {
	var f1 func(int) int
	f1 = g1
	if f1(1) != g1(1) {
		panic(1)
	}

	var f2 func(int) string = g2
	if f2(2) != "" {
		panic(2)
	}

	if g3(g1, 3) != g1(3) {
		panic(3)
	}

	if g4(g2, 4) != "" {
		panic(4)
	}
}

func g1[P any](x P) P    { return x }
func g2[P, Q any](x P) Q { var q Q; return q }

func g3(f1 func(int) int, x int) int       { return f1(x) }
func g4(f2 func(int) string, x int) string { return f2(x) }
