// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package abi

const (
	// We have three different sequences for stack bounds checks, depending on
	// whether the stack frame of a function is small, big, or huge.

	// After a stack split check the SP is allowed to be StackSmall bytes below
	// the stack guard.
	//
	// Functions that need frames <= StackSmall can perform the stack check
	// using a single comparison directly between the stack guard and the SP
	// because we ensure that StackSmall bytes of stack space are available
	// beyond the stack guard.
	StackSmall = 128

	// Functions that need frames <= StackBig can assume that neither
	// SP-framesize nor stackGuard-StackSmall will underflow, and thus use a
	// more efficient check. In order to ensure this, StackBig must be <= the
	// size of the unmapped space at zero.
	StackBig = 4096
)
