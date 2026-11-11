// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"simd/archsimd"
	"testing"
)

//go:noinline
func f(a, b, c archsimd.Int32x16) archsimd.Int32x16 {
	return a.AndNot(b).Xor(c) // a &^ b ^ c, fused into one VPTERNLOGD
}

func TestTernLogAndnot(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("needs AVX-512")
		return
	}
	A := int32(0b1100)
	B := int32(0b1010)
	C := int32(0)
	a := archsimd.BroadcastInt32x16(A)
	b := archsimd.BroadcastInt32x16(B)
	c := archsimd.BroadcastInt32x16(C)
	var r [16]int32
	f(a, b, c).StoreArray(&r)
	want := A &^ B
	got := r[0]
	if want != got {
		t.Errorf("got %04b, want %04b", got, want)
	}
}
