// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import "unsafe"

// scratch keeps every live scratch buffer reachable from the Go heap until
// the host resets it, so the collector never frees a buffer that the host
// still addresses by number. It stays nil until the first allocation
// rather than getting a package initializer: a Go main package that only
// gets this package through the implicit import has no init dependency on
// it, so no initializer of this package would run.
var scratch map[int32][]byte

// address returns the linear-memory address of the first byte of b.
func address(b []byte) int32 {
	return int32(uintptr(unsafe.Pointer(unsafe.SliceData(b))))
}

// scratchAlloc allocates n zeroed bytes of scratch memory and returns
// their address for the host loader to fill (SPEC Sec 14.2). Growing the
// heap may detach the host's memory views, so the loader reads
// memory.buffer again after this call.
//
//go:wasmexport _logos_scratch_alloc
func scratchAlloc(n int32) int32 {
	b := make([]byte, n)
	p := address(b)
	if scratch == nil {
		scratch = make(map[int32][]byte)
	}
	scratch[p] = b
	return p
}

// scratchReset releases the scratch buffer at p, which scratchAlloc
// returned. An address that no live buffer holds is ignored.
//
//go:wasmexport _logos_scratch_reset
func scratchReset(p int32) {
	delete(scratch, p)
}

// errmsgPtr returns the address of the message that WasmError recorded
// last.
//
//go:wasmexport _logos_errmsg_ptr
func errmsgPtr() int32 {
	return address(errmsg)
}

// errmsgLen returns the byte length of the message that WasmError recorded
// last.
//
//go:wasmexport _logos_errmsg_len
func errmsgLen() int32 {
	return int32(len(errmsg))
}
