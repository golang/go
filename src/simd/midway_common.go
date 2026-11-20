// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simdstable && (amd64 || arm64 || wasm)

package simd

import (
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
// lengths, GODEBUG=simd=N (N=128, 256, or 512) limits the vector length to
// at most N.  If N is larger than the hardware supports, the simd package
// uses the longest length the hardware supports.  GODEBUG=simd=1 requests the
// default behavior, the same as leaving it unset, as does any value not
// listed here.
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

// clmulImpl names the archsimd carryless multiply that the CarrylessMul
// methods use at maxVectorSize: an archsimd type, such as "Uint64x2", whose
// CarrylessMul methods are a single instruction, or "software". If it is
// narrower than maxVectorSize, the methods apply it to each part.
var clmulImpl = "software"

func init() {
	maxVectorSize, emulated = configure(archMaxVectorSize(), simd.Value())
	hwClmul = false
	if !emulated {
		clmulImpl, hwClmul = archClmulImpl(maxVectorSize)
	}
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

	switch gosimd {
	case "0":
		// user-requested emulation
		return 128, true
	case "128":
		return min(128, hwMax), false
	case "256":
		return min(256, hwMax), false
	case "512":
		return min(512, hwMax), false
	}
	// By default, including for "1" and for values this doesn't recognize,
	// use the largest vector size the hardware supports.
	return hwMax, false
}
