// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 24503: Handle == and != of addresses taken of symbols consistently.

package main

func test() string {
	type test struct{}
	o1 := test{}
	o2 := test{}
	if &o1 == &o2 {
		return "equal"
	}
	if &o1 != &o2 {
		return "unequal"
	}
	return "failed"
}

func main() {
	if test() == "failed" {
		panic("expected either 'equal' or 'unequal'")
	}
}
