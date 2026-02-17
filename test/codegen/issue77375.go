// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func Range(n int) []int {
	m := make([]int, n)

	for i := 0; i < n; i++ {
		m[i] = i
	}

	for i := range n {
		m[i] = i
	}

	for i := range len(m) {
		m[i] = i
	}

	for i := range m {
		m[i] = i
	}

	return m
}

func F(size int) {
	// amd64:-`.*panicBounds`
	// arm64:-`.*panicBounds`
	Range(size)
}

func main() {
	F(-1)
}
