// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type A struct {
	a int
	B
}

type B struct {
	b int
	C
}

type C struct {
	c int
}

func main() {
	_ = A{a: 1, b: 2}
	_ = A{a: 1, c: 3}
	_ = A{a: 1, b: 2, c: 3} // don't panic during compilation
}
