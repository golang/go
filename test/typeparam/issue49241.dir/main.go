// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"b"
	"c"
)

func main() {
	if b.G() != c.G() {
		println(b.G(), c.G())
		panic("bad")
	}
	if b.F() != c.F() {
		println(b.F(), c.F())
		panic("bad")
	}
}
