// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package archsimd

// LoadInt8sPart loads an Int8s vector from memory pointed to by s and returns the number of
// loaded elements.
// Asm: Emulated, CPU Feature: SVE
func LoadInt8sPart(s []int8) (Int8s, int) {
	return loadInt8sMasked(&s[0], Mask8sFromCount(len(s))), min(len(s), vl())
}

// Asm: ZLD1B,
func loadInt8sMasked(s *int8, m Mask8s) Int8s

// StorePart stores x to memory pointed to by s and returns the number of
// stored elements.
// Asm: Emulated, CPU Feature: SVE
func (x Int8s) StorePart(s []int8) int {
	x.storeMasked(&s[0], Mask8sFromCount(len(s)))
	return min(len(s), vl())
}

// Asm: ZST1B, CPU Feature: SVE
func (x Int8s) storeMasked(s *int8, m Mask8s)

// Mask8sFromCount creates a Mask8s that enables elements from index 0 up to x, exclusive.
// Asm: PWHILELT, CPU Feature: SVE
func Mask8sFromCount(x int) Mask8s

// Greater returns a mask whose elements indicate whether x > y.
// Asm: ZCMPGT, CPU Feature: SVE
func (x Int8s) Greater(y Int8s) Mask8s

// Masked returns the result vector of x masked by m, where m is active,
// or 0 otherwise.
// Asm: Emulated, CPU Feature: SVE
func (x Int8s) Masked(m Mask8s) Int8s {
	var zero Int8s
	return x.IfElse(zero, m)
}

// IfElse returns x but with elements set to y where mask is false.
// Asm: ZSEL, CPU Feature: SVE
func (x Int8s) IfElse(y Int8s, m Mask8s) Int8s

// Add adds two Int8s vectors element-wise.
// Asm: ZADD, CPU Feature: SVE
func (x Int8s) Add(y Int8s) Int8s
