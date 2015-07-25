// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests arithmetic expressions

package main

func test64BitConstMult(a, b int64) {
	want := 34359738369*a + b*34359738370
	if got := test64BitConstMult_ssa(a, b); want != got {
		println("test64BitConstMult failed, wanted", want, "got", got)
		failed = true
	}
}
func test64BitConstMult_ssa(a, b int64) int64 {
	switch { // prevent inlining
	}
	return 34359738369*a + b*34359738370
}

func test64BitConstAdd(a, b int64) {
	want := a + 575815584948629622 + b + 2991856197886747025
	if got := test64BitConstAdd_ssa(a, b); want != got {
		println("test64BitConstAdd failed, wanted", want, "got", got)
		failed = true
	}
}
func test64BitConstAdd_ssa(a, b int64) int64 {
	switch {
	}
	return a + 575815584948629622 + b + 2991856197886747025
}

var failed = false

func main() {

	test64BitConstMult(1, 2)
	test64BitConstAdd(1, 2)

	if failed {
		panic("failed")
	}
}
