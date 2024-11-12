// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build purego

package maphash

import (
	"crypto/rand"
	"internal/byteorder"
	"math/bits"
	"reflect"
)

var hashkey [4]uint64

func init() {
	for i := range hashkey {
		hashkey[i] = randUint64()
	}
}

func rthash(buf []byte, seed uint64) uint64 {
	if len(buf) == 0 {
		return seed
	}
	return wyhash(buf, seed, uint64(len(buf)))
}

func rthashString(s string, state uint64) uint64 {
	return rthash([]byte(s), state)
}

func randUint64() uint64 {
	buf := make([]byte, 8)
	_, _ = rand.Read(buf)
	return byteorder.LEUint64(buf)
}

// This is a port of wyhash implementation in runtime/hash64.go,
// without using unsafe for purego.

const m5 = 0x1d8e4e27c47d124f

func wyhash(key []byte, seed, len uint64) uint64 {
	p := key
	i := len
	var a, b uint64
	seed ^= hashkey[0]

	if i > 16 {
		if i > 48 {
			seed1 := seed
			seed2 := seed
			for ; i > 48; i -= 48 {
				seed = mix(r8(p)^hashkey[1], r8(p[8:])^seed)
				seed1 = mix(r8(p[16:])^hashkey[2], r8(p[24:])^seed1)
				seed2 = mix(r8(p[32:])^hashkey[3], r8(p[40:])^seed2)
				p = p[48:]
			}
			seed ^= seed1 ^ seed2
		}
		for ; i > 16; i -= 16 {
			seed = mix(r8(p)^hashkey[1], r8(p[8:])^seed)
			p = p[16:]
		}
	}
	switch {
	case i == 0:
		return seed
	case i < 4:
		a = r3(p, i)
	default:
		n := (i >> 3) << 2
		a = r4(p)<<32 | r4(p[n:])
		b = r4(p[i-4:])<<32 | r4(p[i-4-n:])
	}
	return mix(m5^len, mix(a^hashkey[1], b^seed))
}

func r3(p []byte, k uint64) uint64 {
	return (uint64(p[0]) << 16) | (uint64(p[k>>1]) << 8) | uint64(p[k-1])
}

func r4(p []byte) uint64 {
	return uint64(byteorder.LEUint32(p))
}

func r8(p []byte) uint64 {
	return byteorder.LEUint64(p)
}

func mix(a, b uint64) uint64 {
	hi, lo := bits.Mul64(a, b)
	return hi ^ lo
}

func comparableF[T comparable](h *Hash, v T) {
	vv := reflect.ValueOf(v)
	appendT(h, vv)
}
