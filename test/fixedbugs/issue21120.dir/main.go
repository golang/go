// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"

	"./a"
	"./b"
)

func main() {
	// Make sure the reflect information for a.S is in the executable.
	_ = a.V()

	b1 := b.F1()
	b2 := b.F2()
	if b1 != b2 {
		fmt.Printf("%q (from b.F1()) != %q (from b.F2())\n", b1, b2)
		os.Exit(1)
	}
}
