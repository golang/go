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
	// amd64:`MOVQ\t\$0, ""\.s\+32\(SP\)`
	var s *int
	g(&s) // force s to be a stack object

	// amd64:`LEAQ\t""\..*\+31\(SP\)`
	c <- struct{}{}
}

//go:noinline
func g(p **int) {
}
