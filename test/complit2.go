// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that composite literals using selectors for
// embedded fields are assembled correctly.

package main

import "fmt"

type A struct {
	a int
	B
}

type B struct {
	b string
	C
}

type C struct {
	c any
}

func main() {
	eq(A{1, B{b: "foo"}}, A{a: 1, b: "foo"})
	eq(A{B: B{C: C{c: "foo"}}}, A{c: "foo"})
}

func eq(x, y any) {
	if x != y {
		panic(fmt.Sprintf("%v != %v", x, y))
	}
}
