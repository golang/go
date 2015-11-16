// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64 amd64p32 386

package runtime

import (
	"runtime/internal/sys"
	"unsafe"
)

// adjust Gobuf as it if executed a call to fn with context ctxt
// and then did an immediate gosave.
func gostartcall(buf *gobuf, fn, ctxt unsafe.Pointer) {
	sp := buf.sp
	if sys.RegSize > sys.PtrSize {
		sp -= sys.PtrSize
		*(*uintptr)(unsafe.Pointer(sp)) = 0
	}
	sp -= sys.PtrSize
	*(*uintptr)(unsafe.Pointer(sp)) = buf.pc
	buf.sp = sp
	buf.pc = uintptr(fn)
	buf.ctxt = ctxt
}

// Called to rewind context saved during morestack back to beginning of function.
// To help us, the linker emits a jmp back to the beginning right after the
// call to morestack. We just have to decode and apply that jump.
func rewindmorestack(buf *gobuf) {
	pc := (*[8]byte)(unsafe.Pointer(buf.pc))
	if pc[0] == 0xe9 { // jmp 4-byte offset
		buf.pc = buf.pc + 5 + uintptr(int64(*(*int32)(unsafe.Pointer(&pc[1]))))
		return
	}
	if pc[0] == 0xeb { // jmp 1-byte offset
		buf.pc = buf.pc + 2 + uintptr(int64(*(*int8)(unsafe.Pointer(&pc[1]))))
		return
	}
	if pc[0] == 0xcc {
		// This is a breakpoint inserted by gdb.  We could use
		// runtime·findfunc to find the function.  But if we
		// do that, then we will continue execution at the
		// function entry point, and we will not hit the gdb
		// breakpoint.  So for this case we don't change
		// buf.pc, so that when we return we will execute
		// the jump instruction and carry on.  This means that
		// stack unwinding may not work entirely correctly
		// (https://golang.org/issue/5723) but the user is
		// running under gdb anyhow.
		return
	}
	print("runtime: pc=", pc, " ", hex(pc[0]), " ", hex(pc[1]), " ", hex(pc[2]), " ", hex(pc[3]), " ", hex(pc[4]), "\n")
	throw("runtime: misuse of rewindmorestack")
}
