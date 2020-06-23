// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:noinline
func div(x, y uint32) uint32 {
	return x / y
}

func main() {
	a := div(97, 11)
	if a != 8 {
		panic("FAIL")
	}
}
