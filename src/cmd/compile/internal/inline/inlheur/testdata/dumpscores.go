// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dumpscores

var G int

func inlinable(x int, f func(int) int) int {
	if x != 0 {
		return 1
	}
	G += noninl(x)
	return f(x)
}

func inlinable2(x int) int {
	return noninl(-x)
}

//go:noinline
func noninl(x int) int {
	return x + 1
}

func tooLargeToInline(x int) int {
	if x > 101 {
		// Drive up the cost of inlining this func over the
		// regular threshold.
		return big(big(big(big(big(G + x)))))
	}
	if x < 100 {
		// make sure this callsite is scored properly
		G += inlinable(101, inlinable2)
		if G == 101 {
			return 0
		}
		panic(inlinable2(3))
	}
	return G
}

func big(q int) int {
	return noninl(q) + noninl(-q)
}
