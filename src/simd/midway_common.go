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
// informatively.
//
// Some platforms may support vectors of a particular length, but not all of the
// expected operations (those appearing in this package) are available at that
// length.  In that case, the default is to automatically downgrade to a length
// where the operations are supported, perhaps even to emulated-only
// (size=0).  If a size is requested that is not compatible with the available
// features, the simd package will panic (and note the reason).  To override
// the feature check, in the case that the user knows that the missing
// operations will not be used, prefix the size request with a '+', for
// example "GODEBUG=simd=+256".  A plain '+' will override the feature check at
// whatever the hardware's default vector size happens to be.

var simd = godebug.New("#simd")

var maxVectorSize int
var emulated = false
var hwClmul = true

func init() {
	actualMax, allFeatureSize := archMaxVectorSize() // zero == no simd, zero == features unavailable
	gosimd := simd.Value()
	explicitRequest := false

	// No SIMD, must emulate
	if actualMax == 0 {
		maxVectorSize = 128
		emulated = true
		hwClmul = false
		return
	}

	maxVectorSize = actualMax

	// If gosimd begins with a '+' or is a single '1' then override
	// any hardware feature check disabling of hardware SIMD.
	// The '+' may be followed by a size, expected to be 0, 128, 256, 512.
	// If it is zero (e.g., "0" or +0") then hardware SIMD is still disabled.
	if len(gosimd) > 0 && gosimd[0] == '+' {
		// override feature reduction
		// keep maxVectorSize
		// emulated remains false
		// note if features missing.
		hwClmul = allFeatureSize < actualMax
		gosimd = gosimd[1:]
		explicitRequest = true

	} else if allFeatureSize < actualMax {
		if allFeatureSize > 0 {
			maxVectorSize = allFeatureSize
			hwClmul = true
			emulated = false
		} else {
			maxVectorSize = 128
			hwClmul = false
			emulated = true
		}
	}

	if gosimd == "" {
		return
	}

	// possible adjustment to chosen size
	val, err := strconv.Atoi(gosimd)
	if err != nil {
		panic(fmt.Errorf("Could not parse GODEBUG=gosimd='%s' as a decimal number, %v", gosimd, err))
	}
	if val > actualMax {
		panic(fmt.Errorf("Requested GODEBUG=gosimd=%d is larger than the simd length (%d) supported on this cpu ", val, actualMax))
	}
	if !explicitRequest && val > allFeatureSize {
		panic(fmt.Errorf("Requested GODEBUG=gosimd=%d is larger than the simd length required for expected features (%d) on this cpu. GODEBUG=gosimd='+%d' will skip this check.", val, allFeatureSize, val))
	}
	if val < 0 {
		panic(fmt.Errorf("Requested GODEBUG=gosimd=%d is negative", val))
	}
	// user-requested emulation
	if val == 0 {
		maxVectorSize = 128
		hwClmul = false
		emulated = true
		return
	}

	hwClmul = allFeatureSize >= val
	maxVectorSize = val
	emulated = false
	return
}

// VectorBitSize returns the bit length of the longest vector available
// on the current hardware.  It can be artificially reduced by setting
// GODEBUG=simd=<smaller size> environment variable before running a program.
func VectorBitSize() int {
	return maxVectorSize
}

// Emulated returns whether simd operations are emulated or
// running on actual vector hardware.
func Emulated() bool {
	return emulated
}

// HasHardwareCarrylessMultiply returns whether this platform
// as a hardware-implemented version of carryless multiply.
// With default GODEBUG=simd settings, if this is false,
// it is emulated and merely slow, but with non-default settings
// this can indicate the possibility of a missing instruction
// that will fail ("SIGILL") if it is executed.
func HasHardwareCarrylessMultiply() bool {
	return hwClmul && archHasHwClmul
}
