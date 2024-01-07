// run

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

//go:noinline
func f(x []byte) bool {
	return unsafe.SliceData(x) != nil
}

//go:noinline
func g(x string) bool {
	return unsafe.StringData(x) != nil
}

func main() {
	if f(nil) {
		panic("bad f")
	}
	if g("") {
		panic("bad g")
	}
}
