// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T struct{}

func (T) B() int {
	return 1
}

func main() {
	a := T{}
	x := a
	.B()
	println(x)
}
