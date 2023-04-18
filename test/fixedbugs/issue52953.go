// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 52953: miscompilation for composite literal assignment
// when LHS is address-taken.

package main

type T struct {
	Field1 bool
}

func main() {
	var ret T
	ret.Field1 = true
	var v *bool = &ret.Field1
	ret = T{Field1: *v}
	check(ret.Field1)
}

//go:noinline
func check(b bool) {
	if !b {
		panic("FAIL")
	}
}
