// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import _ "unsafe" // for go:linkname

type stdFunction *byte

//go:linkname os_sigpipe os.sigpipe
func os_sigpipe() {
	throw("too many writes on closed pipe")
}

func sigpanic() {
	g := getg()
	if !canpanic(g) {
		throw("unexpected signal during runtime execution")
	}

	switch uint32(g.sig) {
	case _EXCEPTION_ACCESS_VIOLATION:
		if g.sigcode1 < 0x1000 || g.paniconfault {
			panicmem()
		}
		print("unexpected fault address ", hex(g.sigcode1), "\n")
		throw("fault")
	case _EXCEPTION_INT_DIVIDE_BY_ZERO:
		panicdivide()
	case _EXCEPTION_INT_OVERFLOW:
		panicoverflow()
	case _EXCEPTION_FLT_DENORMAL_OPERAND,
		_EXCEPTION_FLT_DIVIDE_BY_ZERO,
		_EXCEPTION_FLT_INEXACT_RESULT,
		_EXCEPTION_FLT_OVERFLOW,
		_EXCEPTION_FLT_UNDERFLOW:
		panicfloat()
	}
	throw("fault")
}
