// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package archsimd

// vsve is a tag type for SVE vectors.
type vsve struct {
	_sve [0]func() // uncomparable
}

// psve is a tag type for SVE predicates.
type psve struct {
	_sve [0]func() // uncomparable
}

// Int8s is an SVE vector of int8s.
type Int8s struct {
	int8s vsve
	vals  [32]int8
}

// Mask8s is an SVE predicate for 8-bit elements.
type Mask8s struct {
	mask8s psve
	vals   uint32
}

func (v *Int8s) Len() int {
	return vl()
}

func (m *Mask8s) Len() int {
	return vl()
}

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
