// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Hashing algorithm inspired by
//   xxhash: https://code.google.com/p/xxhash/
// cityhash: https://code.google.com/p/cityhash/

// +build amd64 amd64p32 arm64 mips64 mips64le ppc64 ppc64le

package runtime

import "unsafe"

const (
	// Constants for multiplication: four random odd 64-bit numbers.
	m1 = 16877499708836156737
	m2 = 2820277070424839065
	m3 = 9497967016996688599
	m4 = 15839092249703872147
)

func memhash(p unsafe.Pointer, seed, s uintptr) uintptr {
	if GOARCH == "amd64" && GOOS != "nacl" && useAeshash {
		return aeshash(p, seed, s)
	}
	h := uint64(seed + s*hashkey[0])
tail:
	switch {
	case s == 0:
	case s < 4:
		h ^= uint64(*(*byte)(p))
		h ^= uint64(*(*byte)(add(p, s>>1))) << 8
		h ^= uint64(*(*byte)(add(p, s-1))) << 16
		h = rotl_31(h*m1) * m2
	case s <= 8:
		h ^= uint64(readUnaligned32(p))
		h ^= uint64(readUnaligned32(add(p, s-4))) << 32
		h = rotl_31(h*m1) * m2
	case s <= 16:
		h ^= readUnaligned64(p)
		h = rotl_31(h*m1) * m2
		h ^= readUnaligned64(add(p, s-8))
		h = rotl_31(h*m1) * m2
	case s <= 32:
		h ^= readUnaligned64(p)
		h = rotl_31(h*m1) * m2
		h ^= readUnaligned64(add(p, 8))
		h = rotl_31(h*m1) * m2
		h ^= readUnaligned64(add(p, s-16))
		h = rotl_31(h*m1) * m2
		h ^= readUnaligned64(add(p, s-8))
		h = rotl_31(h*m1) * m2
	default:
		v1 := h
		v2 := uint64(seed * hashkey[1])
		v3 := uint64(seed * hashkey[2])
		v4 := uint64(seed * hashkey[3])
		for s >= 32 {
			v1 ^= readUnaligned64(p)
			v1 = rotl_31(v1*m1) * m2
			p = add(p, 8)
			v2 ^= readUnaligned64(p)
			v2 = rotl_31(v2*m2) * m3
			p = add(p, 8)
			v3 ^= readUnaligned64(p)
			v3 = rotl_31(v3*m3) * m4
			p = add(p, 8)
			v4 ^= readUnaligned64(p)
			v4 = rotl_31(v4*m4) * m1
			p = add(p, 8)
			s -= 32
		}
		h = v1 ^ v2 ^ v3 ^ v4
		goto tail
	}

	h ^= h >> 29
	h *= m3
	h ^= h >> 32
	return uintptr(h)
}

// Note: in order to get the compiler to issue rotl instructions, we
// need to constant fold the shift amount by hand.
// TODO: convince the compiler to issue rotl instructions after inlining.
func rotl_31(x uint64) uint64 {
	return (x << 31) | (x >> (64 - 31))
}
