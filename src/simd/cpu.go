// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

// The build condition == if the experiment is not on, cmd/api TestCheck will see this and complain
// see also go/doc/comment, where "simd" is inserted to the package list of the experiment is not on.

package simd

import "internal/cpu"

// HasAVX checks AVX CPU feature.
func HasAVX() bool {
	return cpu.X86.HasAVX
}

// HasAVXVNNI checks AVX CPU feature VNNI.
func HasAVXVNNI() bool {
	return cpu.X86.HasAVXVNNI
}

// HasAVX2 checks AVX2 CPU feature.
func HasAVX2() bool {
	return cpu.X86.HasAVX2
}

// HasAVX512 checks AVX512 CPU feature F+CD+BW+DQ+VL.
func HasAVX512() bool {
	return cpu.X86.HasAVX512
}

// HasAVX512GFNI checks AVX512 CPU feature GFNI.
func HasAVX512GFNI() bool {
	return cpu.X86.HasAVX512GFNI
}

// HasAVX512VBMI checks AVX512 CPU feature VBMI
func HasAVX512VBMI() bool {
	return cpu.X86.HasAVX512VBMI
}

// HasAVX512VBMI2 checks AVX512 CPU feature VBMI2
func HasAVX512VBMI2() bool {
	return cpu.X86.HasAVX512VBMI2
}

// HasAVX512VNNI checks AVX512 CPU feature VNNI
func HasAVX512VNNI() bool {
	return cpu.X86.HasAVX512VNNI
}

// HasAVX512VPOPCNTDQ checks AVX512 CPU feature VPOPCNTDQ
func HasAVX512VPOPCNTDQ() bool {
	return cpu.X86.HasAVX512VPOPCNTDQ
}

// HasAVX512BITALG checks AVX512 CPU feature BITALG
func HasAVX512BITALG() bool {
	return cpu.X86.HasAVX512BITALG
}
