// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/goarch"
	"internal/runtime/sys"
	"unsafe"
)

type m0Stack struct {
	_ [8192 * sys.StackGuardMultiplier]byte
}

var wasmStack m0Stack

func wasmDiv()

func wasmTruncS()
func wasmTruncU()

//go:wasmimport gojs runtime.wasmExit
func wasmExit(code int32)

// adjust Gobuf as it if executed a call to fn with context ctxt
// and then stopped before the first instruction in fn.
func gostartcall(buf *gobuf, fn, ctxt unsafe.Pointer) {
	sp := buf.sp
	sp -= goarch.PtrSize
	*(*uintptr)(unsafe.Pointer(sp)) = buf.pc
	buf.sp = sp
	buf.pc = uintptr(fn)
	buf.ctxt = ctxt
}

func notInitialized() // defined in assembly, call notInitialized1

// Called if a wasmexport function is called before runtime initialization
//
//go:nosplit
func notInitialized1() {
	writeErrStr("runtime: wasmexport function called before runtime initialization\n")
	if isarchive || islibrary {
		writeErrStr("\tcall _initialize first\n")
	} else {
		writeErrStr("\tcall _start first\n")
	}
}
