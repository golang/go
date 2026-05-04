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

// runtime variable to check if the processor we're running on
// actually supports the instructions used by the AES-based
// hash implementation.
var UseAeshash bool

const hashRandomBytes = goarch.PtrSize / 4 * 64

// used to seed the hash function
var aeskeysched [hashRandomBytes]byte

// used in hash{32,64}.go to seed the hash function
var hashkey [4]uintptr

func AlgInit() {
	// Install AES hash algorithms if the instructions needed are present.
	if (goarch.GOARCH == "386" || goarch.GOARCH == "amd64") &&
		cpu.X86.HasAES && // AESENC
		cpu.X86.HasSSSE3 && // PSHUFB
		cpu.X86.HasSSE41 { // PINSR{D,Q}

		// In aeshashbody (that is used by memhash & strhash)
		// we have global variables that should be properly aligned.
		//
		// See #12415
		if !checkMasksAndShiftsAlignment() {
			fatal("maps: global variables for AES hashing are not properly aligned!")
		}
		initAlgAES()
		return
	}
	if goarch.GOARCH == "arm64" && cpu.ARM64.HasAES {
		initAlgAES()
		return
	}
	for i := range hashkey {
		hashkey[i] = uintptr(bootstrapRand())
	}
}

func initAlgAES() {
	UseAeshash = true
	// Initialize with random data so hash collisions will be hard to engineer.
	key := (*[hashRandomBytes / 8]uint64)(unsafe.Pointer(&aeskeysched))
	for i := range key {
		key[i] = bootstrapRand()
	}
}

func strHashFallback(a unsafe.Pointer, h uintptr) uintptr {
	type stringStruct struct {
		str unsafe.Pointer
		len int
	}
	x := (*stringStruct)(a)
	return memHashFallback(x.str, h, uintptr(x.len))
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
