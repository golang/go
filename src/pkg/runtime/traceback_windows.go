// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// sigtrampPC is the PC at the beginning of the jmpdefer assembly function.
// The traceback needs to recognize it on link register architectures.
var sigtrampPC uintptr

func sigtramp()

func init() {
	sigtrampPC = funcPC(sigtramp)
	systraceback = traceback_windows
}

func traceback_windows(f *_func, frame *stkframe, gp *g, printing bool, callback func(*stkframe, unsafe.Pointer) bool, v unsafe.Pointer) (changed, aborted bool) {
	// The main traceback thinks it has found a function. Check this.

	// Windows exception handlers run on the actual g stack (there is room
	// dedicated to this below the usual "bottom of stack"), not on a separate
	// stack. As a result, we have to be able to unwind past the exception
	// handler when called to unwind during stack growth inside the handler.
	// Recognize the frame at the call to sighandler in sigtramp and unwind
	// using the context argument passed to the call. This is awful.
	if f != nil && f.entry == sigtrampPC && frame.pc > f.entry {
		var r *context
		// Invoke callback so that stack copier sees an uncopyable frame.
		if callback != nil {
			frame.continpc = frame.pc
			frame.argp = 0
			frame.arglen = 0
			if !callback(frame, v) {
				aborted = true
				return
			}
		}
		r = (*context)(unsafe.Pointer(frame.sp + ptrSize))
		frame.pc = contextPC(r)
		frame.sp = contextSP(r)
		frame.lr = 0
		frame.fp = 0
		frame.fn = nil
		if printing && showframe(nil, gp) {
			print("----- exception handler -----\n")
		}
		f = findfunc(frame.pc)
		if f == nil {
			print("runtime: unknown pc ", hex(frame.pc), " after exception handler\n")
			if callback != nil {
				gothrow("unknown pc")
			}
		}
		frame.fn = f
		changed = true
		return
	}

	return
}
