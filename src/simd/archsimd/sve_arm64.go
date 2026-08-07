// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package archsimd

// vl reports the SVE hardware vector length in bytes. It is an intrinsic,
// lowered to the RDVL instruction.
func vl() int

func init() {
	if !ARM64.SVE() {
		// No SVE so no need to check, unsafe accesses are not reachable.
		return
	}
	// Go supports VL up to 32 bytes, that's because the stack allocation
	// for scalable vectors have to be a fixed size, and 32 bytes is what
	// we currently support, which we believe should cover most hardware.
	// TODO: when the support for dynamic stack is ready, we can reconsider
	// this design choice.
	// TODO: another idea is to make SVE() return false if vl() > 32.
	// But the user can still write code without checking SVE(), then
	// it makes out-of-bound memory access possible.
	if vl() > 32 {
		panic("SVE vector length > 32 bytes not supported")
	}
}
