// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && linux && goexperiment.regabiargs

package runtime

import "internal/abi"

// storeRegArgs sets up argument registers in the signal
// context state from an abi.RegArgs.
//
// Both src and dst must be non-nil.
func storeRegArgs(dst *sigcontext, src *abi.RegArgs) {
	dst.rax = uint64(src.Ints[0])
	dst.rbx = uint64(src.Ints[1])
	dst.rcx = uint64(src.Ints[2])
	dst.rdi = uint64(src.Ints[3])
	dst.rsi = uint64(src.Ints[4])
	dst.r8 = uint64(src.Ints[5])
	dst.r9 = uint64(src.Ints[6])
	dst.r10 = uint64(src.Ints[7])
	dst.r11 = uint64(src.Ints[8])
	for i := range src.Floats {
		dst.fpstate._xmm[i].element[0] = uint32(src.Floats[i] >> 0)
		dst.fpstate._xmm[i].element[1] = uint32(src.Floats[i] >> 32)
	}
}

func loadRegArgs(dst *abi.RegArgs, src *sigcontext) {
	dst.Ints[0] = uintptr(src.rax)
	dst.Ints[1] = uintptr(src.rbx)
	dst.Ints[2] = uintptr(src.rcx)
	dst.Ints[3] = uintptr(src.rdi)
	dst.Ints[4] = uintptr(src.rsi)
	dst.Ints[5] = uintptr(src.r8)
	dst.Ints[6] = uintptr(src.r9)
	dst.Ints[7] = uintptr(src.r10)
	dst.Ints[8] = uintptr(src.r11)
	for i := range dst.Floats {
		dst.Floats[i] = uint64(src.fpstate._xmm[i].element[0]) << 0
		dst.Floats[i] |= uint64(src.fpstate._xmm[i].element[1]) << 32
	}
}
