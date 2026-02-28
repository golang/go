// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 20811: slice-in-bound check is lowered incorrectly on
// amd64p32.

package main

func main() {
	i := g()
	_ = "x"[int32(i)]
	j := g()
	_ = "x"[:int32(j)]
}

//go:noinline
func g() int64 {
	return 4398046511104
}

