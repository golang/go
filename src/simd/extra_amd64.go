// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd

// ClearAVXUpperBits clears the high bits of Y0-Y15 and Z0-Z15 registers.
// It is intended for transitioning from AVX to SSE, eliminating the
// performance penalties caused by false dependencies.
//
// Note: in the future the compiler may automatically generate the
// instruction, making this function unnecessary.
//
// Asm: VZEROUPPER, CPU Feature: AVX
func ClearAVXUpperBits()
