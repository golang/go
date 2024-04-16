// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Hashing algorithm inspired by
// wyhash: https://github.com/wangyi-fudan/wyhash

//go:build amd64 || arm64 || loong64 || mips64 || mips64le || ppc64 || ppc64le || riscv64 || s390x || wasm

package runtime

import (
	"runtime/internal/math"
	"unsafe"
)

const (
	m1 = 0xa0761d6478bd642f
	m2 = 0xe7037ed1a0b428db
	m3 = 0x8ebc6af09c88c6e3
	m4 = 0x589965cc75374cc3
	m5 = 0x1d8e4e27c47d124f
)

func memhashFallback(p unsafe.Pointer, seed, s uintptr) uintptr {
	var a, b uintptr
	seed ^= hashkey[0] ^ m1
	switch {
	case s == 0:
		return seed
	case s < 4:
		a = uintptr(*(*byte)(p))
		a |= uintptr(*(*byte)(add(p, s>>1))) << 8
		a |= uintptr(*(*byte)(add(p, s-1))) << 16
	case s == 4:
		a = r4(p)
		b = a
	case s < 8:
		a = r4(p)
		b = r4(add(p, s-4))
	case s == 8:
		a = r8(p)
		b = a
	case s <= 16:
		a = r8(p)
		b = r8(add(p, s-8))
	default:
		l := s
		if l > 48 {
			seed1 := seed
			seed2 := seed
			for ; l > 48; l -= 48 {
				seed = mix(r8(p)^hashkey[1]^m2, r8(add(p, 8))^seed)
				seed1 = mix(r8(add(p, 16))^hashkey[2]^m3, r8(add(p, 24))^seed1)
				seed2 = mix(r8(add(p, 32))^hashkey[3]^m4, r8(add(p, 40))^seed2)
				p = add(p, 48)
			}
			seed ^= seed1 ^ seed2
		}
		for ; l > 16; l -= 16 {
			seed = mix(r8(p)^hashkey[1]^m2, r8(add(p, 8))^seed)
			p = add(p, 16)
		}
		a = r8(add(p, l-16))
		b = r8(add(p, l-8))
	}

	return mix(m5^s, mix(a^hashkey[1]^m2, b^seed))
}

func memhash32Fallback(p unsafe.Pointer, seed uintptr) uintptr {
	a := r4(p)
	return mix(m5^4, mix(a^hashkey[1]^m2, a^seed^hashkey[0]^m1))
}

func memhash64Fallback(p unsafe.Pointer, seed uintptr) uintptr {
	a := r8(p)
	return mix(m5^8, mix(a^hashkey[1]^m2, a^seed^hashkey[0]^m1))
}

func mix(a, b uintptr) uintptr {
	hi, lo := math.Mul64(uint64(a), uint64(b))
	return uintptr(hi ^ lo)
}

func r4(p unsafe.Pointer) uintptr {
	return uintptr(readUnaligned32(p))
}

func r8(p unsafe.Pointer) uintptr {
	return uintptr(readUnaligned64(p))
}
