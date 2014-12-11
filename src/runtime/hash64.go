// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Hashing algorithm inspired by
//   xxhash: https://code.google.com/p/xxhash/
// cityhash: https://code.google.com/p/cityhash/

// +build amd64 amd64p32 ppc64 ppc64le

package runtime

import "unsafe"

const (
	// Constants for multiplication: four random odd 64-bit numbers.
	m1 = 16877499708836156737
	m2 = 2820277070424839065
	m3 = 9497967016996688599
	m4 = 15839092249703872147
)

func memhash(p unsafe.Pointer, s, seed uintptr) uintptr {
	if GOARCH == "amd64" && GOOS != "nacl" && useAeshash {
		return aeshash(p, s, seed)
	}
	h := uint64(seed + s)
tail:
	switch {
	case s == 0:
	case s < 4:
		w := uint64(*(*byte)(p))
		w += uint64(*(*byte)(add(p, s>>1))) << 8
		w += uint64(*(*byte)(add(p, s-1))) << 16
		h ^= w * m1
	case s <= 8:
		w := uint64(readUnaligned32(p))
		w += uint64(readUnaligned32(add(p, s-4))) << 32
		h ^= w * m1
	case s <= 16:
		h ^= readUnaligned64(p) * m1
		h = rotl_31(h) * m2
		h = rotl_27(h)
		h ^= readUnaligned64(add(p, s-8)) * m1
	case s <= 32:
		h ^= readUnaligned64(p) * m1
		h = rotl_31(h) * m2
		h = rotl_27(h)
		h ^= readUnaligned64(add(p, 8)) * m1
		h = rotl_31(h) * m2
		h = rotl_27(h)
		h ^= readUnaligned64(add(p, s-16)) * m1
		h = rotl_31(h) * m2
		h = rotl_27(h)
		h ^= readUnaligned64(add(p, s-8)) * m1
	default:
		v1 := h
		v2 := h + m1
		v3 := h + m2
		v4 := h + m3
		for s >= 32 {
			v1 ^= readUnaligned64(p) * m1
			v1 = rotl_31(v1) * m2
			p = add(p, 8)
			v2 ^= readUnaligned64(p) * m1
			v2 = rotl_31(v2) * m2
			p = add(p, 8)
			v3 ^= readUnaligned64(p) * m1
			v3 = rotl_31(v3) * m2
			p = add(p, 8)
			v4 ^= readUnaligned64(p) * m1
			v4 = rotl_31(v4) * m2
			p = add(p, 8)
			s -= 32
		}
		h = rotl_27(v1)*m1 + rotl_27(v2)*m2 + rotl_27(v3)*m3 + rotl_27(v4)*m4
		goto tail
	}

	h ^= h >> 33
	h *= m2
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
func rotl_27(x uint64) uint64 {
	return (x << 27) | (x >> (64 - 27))
}
