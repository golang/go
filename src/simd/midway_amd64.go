// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd

import (
	"fmt"
	"os"
	"simd/archsimd"
	"strconv"
)

var maxVectorSize int

func init() {
	actualMax := archMaxVectorSize()
	if gosimd := os.Getenv("GOSIMD"); gosimd != "" {
		val, err := strconv.Atoi(gosimd)
		if err != nil {
			panic(fmt.Errorf("Could not parse GOSIMD(='%s') as a decimal number, %v", gosimd, err))
		}
		if val > actualMax {
			panic(fmt.Errorf("Requested GOSIMD(='%d') is larger than the simd length (%d) supported on this cpu ", val, actualMax))
		}
		if val < 0 {
			panic(fmt.Errorf("Requested GOSIMD(='%d') is negative", val))
		}
		maxVectorSize = val
		return
	}
	maxVectorSize = actualMax
}

// VectorBitSize returns the bit length of the longest vector available
// on the current hardware.  For amd64, this is 128, 256, or 512, depending
// on the hardware.  It can be artificially reduced by setting the
// GOSIMD environment variable before running a program.
func VectorBitSize() int {
	return maxVectorSize
}

// Emulated returns whether simd operations are emulated or
// running on actual vector hardware.
func Emulated() bool {
	return false
}

func archMaxVectorSize() int {
	if archsimd.X86.AVX512() {
		return 512
	}
	if archsimd.X86.AVX2() {
		return 256
	}
	// AVX has 256 bit float ops but only 128-bit integer ops
	// therefore it is 128.
	if archsimd.X86.AVX() {
		return 128
	}
	return 0
}
