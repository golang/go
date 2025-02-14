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
	// amd64:`MOVQ\t\$0, command-line-arguments\.s\+56\(SP\)`
	var s *int
	// force s to be a stack object, also use some (fixed) stack space
	g(&s, 1, 2, 3, 4, 5)

	// amd64:`LEAQ\tcommand-line-arguments\..*\+55\(SP\)`
	c <- noliteral(struct{}{})
}

// Like zeroSize, but without hiding the zero-sized struct.
func zeroSize2() {
	c := make(chan struct{})
	// amd64:`MOVQ\t\$0, command-line-arguments\.s\+48\(SP\)`
	var s *int
	// force s to be a stack object, also use some (fixed) stack space
	g(&s, 1, 2, 3, 4, 5)

	// amd64:`LEAQ\tcommand-line-arguments\..*stmp_\d+\(SB\)`
	c <- struct{}{}
}

//go:noinline
func g(**int, int, int, int, int, int) {}

// noliteral prevents the compiler from recognizing a literal value.
//
//go:noinline
func noliteral[T any](t T) T {
	return t
}
