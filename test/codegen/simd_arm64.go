// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// These tests check ARM64 SIMD code generation and peephole optimizations.

//go:build goexperiment.simd && arm64

package codegen

import (
	"simd/archsimd"
)

//go:noinline
func forceSpill() {}

func spillAroundCall(a archsimd.Int8x16) archsimd.Int8x16 {
	forceSpill()
	// arm64:`FMOVQ` `FMOVQ`
	return a
}

var (
	sinkU8 archsimd.Uint8x16
	sinkI8 archsimd.Int8x16
)

func broadcastConstImmFold(k int) {
	switch k {
	case 0:
		// arm64:`VMOVI [$]0,` -`VDUP`
		sinkU8 = archsimd.BroadcastUint8x16(0)
	case 1:
		// arm64:`VMOVI [$]1,` -`VDUP`
		sinkU8 = archsimd.BroadcastUint8x16(1)
	case 127:
		// arm64:`VMOVI [$]127,` -`VDUP`
		sinkI8 = archsimd.BroadcastInt8x16(127)
	case 128:
		// arm64:`VMOVI [$]128,` -`VDUP`
		sinkU8 = archsimd.BroadcastUint8x16(128)
	case -128:
		// arm64:`VMOVI [$]128,` -`VDUP`
		sinkI8 = archsimd.BroadcastInt8x16(-128)
	case 255:
		// arm64:`VMOVI [$]255,` -`VDUP`
		sinkU8 = archsimd.BroadcastUint8x16(255)
	case -1:
		// arm64:`VMOVI [$]255,` -`VDUP`
		sinkI8 = archsimd.BroadcastInt8x16(-1)
	case -2:
		// arm64:`VMOVI [$]254,` -`VDUP`
		sinkI8 = archsimd.BroadcastInt8x16(-2)
	default:
		// arm64:`VMOV R0, V\d+.B\[0\]` `VDUP`
		sinkI8 = archsimd.BroadcastInt8x16(int8(k))
	}
}
