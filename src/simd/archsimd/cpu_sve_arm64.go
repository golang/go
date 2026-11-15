// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package archsimd

import "internal/cpu"

// vl reports the SVE hardware vector length in bytes. It is an intrinsic,
// lowered to the RDVL instruction.
func vl() int

// SVE returns whether the CPU supports the SVE extension with a hardware
// vector length of at most 256 bits. The scalable vector types require SVE.
//
// SVE is defined on all GOARCHes, but will only return true on
// GOARCH arm64.
func (ARM64Features) SVE() bool {
	return cpu.ARM64.HasSVE && vl() <= 32
}

// SVE2 returns whether the CPU supports the SVE2 extension with a hardware
// vector length of at most 256 bits.
//
// SVE2 is defined on all GOARCHes, but will only return true on
// GOARCH arm64.
func (ARM64Features) SVE2() bool {
	return cpu.ARM64.HasSVE2 && vl() <= 32
}
