// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build purego

package maphash

import (
	"crypto/rand"
	"internal/abi"
	"internal/byteorder"
	"math/bits"
	"reflect"
)

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
	return byteorder.LeUint64(buf)
}

// This is a port of wyhash implementation in runtime/hash64.go,
// without using unsafe for purego.

const (
	m1 = 0xa0761d6478bd642f
	m2 = 0xe7037ed1a0b428db
	m3 = 0x8ebc6af09c88c6e3
	m4 = 0x589965cc75374cc3
	m5 = 0x1d8e4e27c47d124f
)

func wyhash(key []byte, seed, len uint64) uint64 {
	p := key
	i := len
	var a, b uint64
	seed ^= m1

	if i > 16 {
		if i > 48 {
			seed1 := seed
			seed2 := seed
			for ; i > 48; i -= 48 {
				seed = mix(r8(p)^m2, r8(p[8:])^seed)
				seed1 = mix(r8(p[16:])^m3, r8(p[24:])^seed1)
				seed2 = mix(r8(p[32:])^m4, r8(p[40:])^seed2)
				p = p[48:]
			}
			seed ^= seed1 ^ seed2
		}
		for ; i > 16; i -= 16 {
			seed = mix(r8(p)^m2, r8(p[8:])^seed)
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
	return mix(m5^len, mix(a^m2, b^seed))
}

func r3(p []byte, k uint64) uint64 {
	return (uint64(p[0]) << 16) | (uint64(p[k>>1]) << 8) | uint64(p[k-1])
}

func r4(p []byte) uint64 {
	return uint64(byteorder.LeUint32(p))
}

func r8(p []byte) uint64 {
	return byteorder.LeUint64(p)
}

func mix(a, b uint64) uint64 {
	hi, lo := bits.Mul64(a, b)
	return hi ^ lo
}

var byteSliceTyp = reflect.TypeFor[[]byte]()

var strSliceTyp = reflect.TypeFor[string]()

func comparableF[T comparable](seed uint64, v T, t *abi.Type) uint64 {
	vv := reflect.ValueOf(v)
	typ := vv.Type()
	if typ == byteSliceTyp {
		return wyhash(vv.Bytes(), seed, uint64(vv.Len()))
	}
	if typ == strSliceTyp {
		return rthashString(vv.String(), seed)
	}
	buf := make([]byte, 0, typ.Size())
	buf = appendT(buf, vv)
	return wyhash(buf, seed, uint64(len(buf)))
}

func appendT(buf []byte, v reflect.Value) []byte {
	switch v.Kind() {
	case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Int:
		return byteorder.LeAppendUint64(buf, uint64(v.Int()))
	case reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uint, reflect.Uintptr:
		return byteorder.LeAppendUint64(buf, v.Uint())
	case reflect.Array, reflect.Slice:
		for i := range v.Len() {
			buf = appendT(buf, v.Index(i))
		}
		return buf
	case reflect.String:
		return append(buf, v.String()...)
	case reflect.Struct:
		for i := range v.NumField() {
			buf = appendT(buf, v.Field(i))
		}
		return buf
	case reflect.Complex64, reflect.Complex128:
		c := v.Complex()
		buf = byteorder.LeAppendUint64(buf, uint64(real(c)))
		return byteorder.LeAppendUint64(buf, uint64(imag(c)))
	case reflect.Float32, reflect.Float64:
		return byteorder.LeAppendUint64(buf, uint64(v.Float()))
	case reflect.Bool:
		return byteorder.LeAppendUint16(buf, btoi(v.Bool()))
	case reflect.UnsafePointer, reflect.Pointer:
		return byteorder.LeAppendUint64(buf, uint64(v.Pointer()))
	case reflect.Interface:
		a := v.InterfaceData()
		buf = byteorder.LeAppendUint64(buf, uint64(a[0]))
		return byteorder.LeAppendUint64(buf, uint64(a[1]))
	}
	panic("unreachable")
}

func btoi(b bool) uint16 {
	if b {
		return 1
	}
	return 0
}
