// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const (
	intSize = 32 << (^uint(0) >> 63)
	minInt  = -1 << (intSize - 1)
)

func main() {
	f()
}

func f() {
	for i := 0; true; i += minInt {
		if i < 0 {
			return
		}
	}
	panic("unreachable")
}
