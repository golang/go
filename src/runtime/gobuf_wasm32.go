// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasm32

package runtime

import (
	"internal/abi"
	"internal/runtime/sys"
	"unsafe"
)

func wasmFuncEntryAddr() uintptr
func wasmFuncCount() uintptr
func wasmGoexitHandle() uintptr

//go:linkname reflect_wasm32FuncHandleToPC reflect.wasm32FuncHandleToPC
func reflect_wasm32FuncHandleToPC(handle uintptr) uintptr {
	return funcValuePC(handle)
}

//go:linkname reflect_wasm32FuncPCToHandle reflect.wasm32FuncPCToHandle
func reflect_wasm32FuncPCToHandle(pc uintptr) uintptr {
	return funcHandle(pc)
}

// funcValuePC returns the logical entry PC for a call_indirect handle by
// indexing runtime.wasmFuncEntry, a linker-generated table of uint32 entry PCs.
// Handle 0 is nil and maps to entry 0, which is zero.
//
//go:nosplit
func funcValuePC(handle uintptr) uintptr {
	if handle > wasmFuncCount() {
		throw("invalid wasm32 function handle")
	}
	base := wasmFuncEntryAddr()
	return uintptr(*(*uint32)(unsafe.Pointer(base + handle*4)))
}

// funcHandle returns the call_indirect handle for the function containing pc.
// PC 0 maps to handle 0. Otherwise, funcHandle calls findfunc to get the
// containing function's entry PC and binary-searches runtime.wasmFuncEntry,
// which is ordered by entry PC.
//
//go:nosplit
func funcHandle(pc uintptr) uintptr {
	if pc == 0 {
		return 0
	}
	f := findfunc(pc)
	if !f.valid() {
		throw("invalid wasm32 function PC")
	}
	entry := f.entry()

	count := wasmFuncCount()
	base := wasmFuncEntryAddr()
	lo, hi := uintptr(1), count+1
	for lo < hi {
		mid := lo + (hi-lo)/2
		midPC := uintptr(*(*uint32)(unsafe.Pointer(base + mid*4)))
		if midPC < entry {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo > count || uintptr(*(*uint32)(unsafe.Pointer(base + lo*4))) != entry {
		throw("missing wasm32 function handle")
	}
	return lo
}

// gobufSetPC sets a wasm32 continuation. gobuf.lr is the PC_F handle paired
// with gobuf.pc, not a link register. Assembly paths that copy an existing
// continuation copy both fields directly.
//
//go:nosplit
func gobufSetPC(buf *gobuf, pc uintptr) {
	buf.pc = pc
	buf.lr = funcHandle(pc)
}

// gobufSetGoexit leaves PC-1 within goexit for stack unwinding and uses
// the static handle instead of looking up a known PC.
//
//go:nosplit
func gobufSetGoexit(buf *gobuf) {
	buf.pc = abi.FuncPCABI0(goexit) + sys.PCQuantum
	buf.lr = wasmGoexitHandle()
}
