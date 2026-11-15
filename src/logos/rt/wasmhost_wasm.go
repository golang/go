// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import "unsafe"

var scratch map[int32][]byte

func address(b []byte) int32 {
	return int32(uintptr(unsafe.Pointer(unsafe.SliceData(b))))
}

const (
	maxScratchBytes = 256 << 20
	scratchRefused  = -1
)

//go:wasmexport _logos_scratch_alloc
func scratchAlloc(n int32) int32 {
	if n < 0 || n > maxScratchBytes {
		return scratchRefused
	}
	b := make([]byte, n)
	p := address(b)
	if scratch == nil {
		scratch = make(map[int32][]byte)
	}
	scratch[p] = b
	return p
}

//go:wasmexport _logos_scratch_reset
func scratchReset(p int32) {
	delete(scratch, p)
}

//go:wasmexport _logos_errmsg_ptr
func errmsgPtr() int32 {
	return address(errmsg)
}

//go:wasmexport _logos_errmsg_len
func errmsgLen() int32 {
	return int32(len(errmsg))
}
