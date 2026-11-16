// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (amd64 || arm64 || wasm)

package simd

import (
	"fmt"
	"internal/godebug"
	"strconv"
)

// The `simd` package provides an architecture and vector-length agnostic API
// for single-instruction-multiple-data "SIMD" vectors and operations. The
// functions and methods in this package are those that can be mostly supported
// in hardware, combined with an emulation for those platforms that are not yet
// supported.
//
// Users can also control emulation and vector length with the 'simd' GODEBUG
// setting.  GODEBUG=simd=0 requests emulation, not hardware SIMD, even if
// hardware is available.  On platforms that may support multiple vector
// lengths, GODEBUG=simd=N (N=128, 256, or 512) requests a specific vector
// length.  If the request cannot be satisfied, the simd package panics
// informatively.  GODEBUG=simd=1 requests the default behavior, the same as
// leaving it unset.
//
// The package only uses SIMD hardware that has a baseline of features, and
// emulates every operation otherwise.  On amd64, the baseline is AVX and FMA.
//
// Every operation in this package is available at every vector length the
// hardware supports, so the default is the longest supported vector size.  A
// few operations, such as carryless multiply, correspond to an instruction
// that some CPUs lack, either at their longest vector length or entirely.
// The archsimd package implements those operations using narrower
// instructions or software on such CPUs.  That is slower than the
// instruction, but it doesn't affect the vector length or any other
// operation.

var simd = godebug.New("#simd")

var maxVectorSize int
var emulated = false
var hwClmul = true

func init() {
	maxVectorSize, emulated = configure(archMaxVectorSize(), simd.Value())
	hwClmul = !emulated && archHasHwClmul(maxVectorSize)
}

type fromArchError string

func (e fromArchError) Error() string {
	if emulated {
		return string(e) + " when simd.Emulated() is true"
	}
	return string(e) + " when simd.VectorBits() is " + strconv.Itoa(maxVectorSize)
}

// configure returns the vector size to use and whether to emulate SIMD
// operations, given the largest vector size the hardware supports (0 if it
// doesn't support SIMD) and the value of GODEBUG=simd.
func configure(hwMax int, gosimd string) (maxVectorSize int, emulated bool) {
	// No SIMD, must emulate
	if hwMax == 0 {
		return 128, true
	}

	var val int
	switch gosimd {
	case "", "1":
		// Use the largest vector size the hardware supports.
		return hwMax, false
	case "0":
		// user-requested emulation
		return 128, true
	case "128":
		val = 128
	case "256":
		val = 256
	case "512":
		val = 512
	default:
		panic(fmt.Errorf("requested GODEBUG=simd=%s is not supported (want 0, 1, 128, 256, or 512)", gosimd))
	}
	if val > hwMax {
		panic(fmt.Errorf("requested GODEBUG=simd=%d is larger than the SIMD length (%d) supported on this CPU", val, hwMax))
	}
	return val, false
}
