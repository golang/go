// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure a pointer variable and a zero-sized variable
// aren't allocated to the same stack slot.
// See issue 24993.

package codegen

func zeroSize() {
	c := make(chan struct{})
	// amd64:`MOVQ\t\$0, ""\.s\+56\(SP\)`
	var s *int
	// force s to be a stack object, also use some (fixed) stack space
	g(&s, 1, 2, 3, 4, 5)

	// amd64:`LEAQ\t""\..*\+55\(SP\)`
	c <- struct{}{}
}

//go:noinline
func g(**int, int, int, int, int, int) {}
