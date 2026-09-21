// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"unsafe"

	"internal/abi"
	"internal/runtime/sys"
)

// adjust Gobuf as if it executed a call to fn with context ctxt
// and then did an immediate Gosave.
func gostartcall(buf *gobuf, fn, ctxt unsafe.Pointer) {
	if buf.lr != 0 {
		throw("invalid use of gostartcall")
	}
	// Use double the PC quantum on riscv64, so that we retain
	// four byte alignment and use non-compressed instructions.
	buf.lr = abi.FuncPCABI0(goexit) + sys.PCQuantum*2
	buf.pc = uintptr(fn)
	buf.ctxt = ctxt
}
