// run

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 8620. Used to fail with -race.

package main

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func test(s1, s2 []struct{}) {
	n := min(len(s1), len(s2))
	if copy(s1, s2) != n {
		panic("bad copy result")
	}
}

func main() {
	var b [100]struct{}
	test(b[:], b[:])
	test(b[1:], b[:])
	test(b[:], b[2:])
}
