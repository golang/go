// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// "Push" linknames are ok.

package main

import (
	"cmd/link/testdata/linkname/p"
	_ "unsafe"
)

// Push f1 to p.
//
//go:linkname f1 cmd/link/testdata/linkname/p.f1
func f1() { f2() }

// f2 is pushed from p.
//
//go:linkname f2
func f2()

func main() {
	p.F()
}
