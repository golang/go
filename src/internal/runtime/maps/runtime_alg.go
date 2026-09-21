// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package maps

import (
	"internal/byteorder"
	"internal/cpu"
	"internal/goarch"
	"unsafe"
)

// MinAeshashSize is the smallest key size hashed with the AES-based
// implementation. Selecting hash is a size comparison against this value.
// Setting this to MaxUintptr disables AES altogether.
//
// When support is detected, the threshold is lowered to select between the
// scalar-based fallback hash and the vector-based AES hash. This value is
// selected on a per-platform basis based on what value produces the best
// benchmark results.
//
// Scalar hashes are faster on small values because it avoids taking a trip
// into the vector unit, which hurts latency (and for very small values,
// throughput).
var MinAeshashSize uintptr = ^uintptr(0)

// AeshashEnabled reports whether this machine hashes any sizes with AES.
//
// Test-only; compare against MinAeshashSize in non-test code to fuse this
// comparison with the MinAeshashSize check.
func AeshashEnabled() bool {
	return MinAeshashSize != ^uintptr(0)
}

const hashRandomBytes = goarch.PtrSize / 4 * 64

// used to seed the hash function
var aeskeysched [hashRandomBytes]byte

// used in hash{32,64}.go to seed the hash function
var hashkey [4]uintptr

// Pre-computed comparisons against MinAeshashSize, which reduces a
// load-and-compare-and-branch to a load-and-branch.
var (
	useAeshash32 bool // = MinAeshashSize <= 4
	useAeshash64 bool // = MinAeshashSize <= 8
)

func AlgInit() {
	// Always intialize hashkey.
	//
	// See #78073
	for i := range hashkey {
		hashkey[i] = uintptr(bootstrapRand())
	}

	// Install AES hash algorithms if the instructions needed are present.
	if (goarch.GOARCH == "386" || goarch.GOARCH == "amd64") &&
		cpu.X86.HasAES && // AESENC
		cpu.X86.HasSSSE3 && // PSHUFB
		cpu.X86.HasSSE41 { // PINSR{D,Q}

		// In memHashAES we have global variables that should be properly aligned.
		//
		// See #12415
		if !checkMasksAndShiftsAlignment() {
			fatal("maps: global variables for AES hashing are not properly aligned!")
		}
		initAlgAES()

		if memHashUsesVAES && !cpu.X86.HasAVX {
			// We are using intrinsics hash implementation.
			// Override the UseAeshash in this case, since it uses VAES (AVX) instructions.
			// While assembly implementation used AES-NI instructions,
			// simd intrinsics only provide access to AVX ones.
			MinAeshashSize = ^uintptr(0)
		}
	} else if goarch.GOARCH == "arm64" && cpu.ARM64.HasAES {
		initAlgAES()
	}

	useAeshash32 = 4 >= MinAeshashSize
	useAeshash64 = 8 >= MinAeshashSize
}

func initAlgAES() {
	// TODO(mcy): investigate cutoffs on a per-uarch basis.
	// See memhash_bench_test.go.
	switch goarch.ArchFamily {
	case goarch.AMD64:
		// Measured on AMD Ryzen Threadripper PRO 7995WX (Zen4).
		MinAeshashSize = 9
	case goarch.ARM64:
		// Measured on Apple M1.
		// Latency crossover is at 192.
		MinAeshashSize = 16
	default:
		MinAeshashSize = 0
	}

	// Initialize with random data so hash collisions will be hard to engineer.
	key := (*[hashRandomBytes / 8]uint64)(unsafe.Pointer(&aeskeysched))
	for i := range key {
		key[i] = bootstrapRand()
	}
}

//go:nosplit
func add(p unsafe.Pointer, x uintptr) unsafe.Pointer {
	return unsafe.Pointer(uintptr(p) + x)
}

// Note: These routines perform the read with a native endianness.
func readUnaligned32(p unsafe.Pointer) uint32 {
	q := (*[4]byte)(p)
	if goarch.BigEndian {
		return byteorder.BEUint32(q[:])
	}
	return byteorder.LEUint32(q[:])
}

func readUnaligned64(p unsafe.Pointer) uint64 {
	q := (*[8]byte)(p)
	if goarch.BigEndian {
		return byteorder.BEUint64(q[:])
	}
	return byteorder.LEUint64(q[:])
}
