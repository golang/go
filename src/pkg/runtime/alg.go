// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

const (
	c0 = uintptr((8-uint64(ptrSize))/4*2860486313 + (uint64(ptrSize)-4)/4*33054211828000289)
	c1 = uintptr((8-uint64(ptrSize))/4*3267000013 + (uint64(ptrSize)-4)/4*23344194077549503)
)

const (
	alg_MEM = iota
	alg_MEM0
	alg_MEM8
	alg_MEM16
	alg_MEM32
	alg_MEM64
	alg_MEM128
	alg_NOEQ
	alg_NOEQ0
	alg_NOEQ8
	alg_NOEQ16
	alg_NOEQ32
	alg_NOEQ64
	alg_NOEQ128
	alg_STRING
	alg_INTER
	alg_NILINTER
	alg_SLICE
	alg_FLOAT32
	alg_FLOAT64
	alg_CPLX64
	alg_CPLX128
	alg_max
)

const nacl = GOOS == "nacl"

var use_aeshash bool

// in asm_*.s
func aeshash(p unsafe.Pointer, s uintptr, h uintptr) uintptr

func memhash(p unsafe.Pointer, s uintptr, h uintptr) uintptr {
	if !nacl && use_aeshash {
		return aeshash(p, s, h)
	}

	h ^= c0
	for s > 0 {
		h = (h ^ uintptr(*(*byte)(p))) * c1
		p = add(p, 1)
		s--
	}
	return h
}

func strhash(a *string, s uintptr, h uintptr) uintptr {
	return memhash((*stringStruct)(unsafe.Pointer(a)).str, uintptr(len(*a)), h)
}

// NOTE: Because NaN != NaN, a map can contain any
// number of (mostly useless) entries keyed with NaNs.
// To avoid long hash chains, we assign a random number
// as the hash value for a NaN.

func f32hash(a *float32, s uintptr, h uintptr) uintptr {
	f := *a
	switch {
	case f == 0:
		return c1 * (c0 ^ h) // +0, -0
	case f != f:
		return c1 * (c0 ^ h ^ uintptr(fastrand2())) // any kind of NaN
	default:
		return c1 * (c0 ^ h ^ uintptr(*(*uint32)(unsafe.Pointer(a))))
	}
}

func f64hash(a *float64, s uintptr, h uintptr) uintptr {
	f := *a
	switch {
	case f == 0:
		return c1 * (c0 ^ h) // +0, -0
	case f != f:
		return c1 * (c0 ^ h ^ uintptr(fastrand2())) // any kind of NaN
	case ptrSize == 4:
		x := (*[2]uintptr)(unsafe.Pointer(a))
		return c1 * (c0 ^ h ^ (x[1] * c1) ^ x[0])
	default:
		return c1 * (c0 ^ h ^ *(*uintptr)(unsafe.Pointer(a)))
	}
}

func c64hash(a *complex64, s uintptr, h uintptr) uintptr {
	x := (*[2]float32)(unsafe.Pointer(a))
	return f32hash(&x[1], 4, f32hash(&x[0], 4, h))
}

func c128hash(a *complex128, s uintptr, h uintptr) uintptr {
	x := (*[2]float64)(unsafe.Pointer(a))
	return f64hash(&x[1], 4, f64hash(&x[0], 4, h))
}

func nohash(a unsafe.Pointer, s uintptr, h uintptr) uintptr {
	panic(errorString("hash of unhashable type"))
}

func interhash(a *interface {
	f()
}, s uintptr, h uintptr) uintptr {
	tab := (*iface)(unsafe.Pointer(a)).tab
	if tab == nil {
		return h
	}
	t := tab._type
	fn := goalg(t.alg).hash
	if **(**uintptr)(unsafe.Pointer(&fn)) == nohashcode {
		// calling nohash will panic too,
		// but we can print a better error.
		panic(errorString("hash of unhashable type " + *t._string))
	}
	if uintptr(t.size) <= ptrSize {
		return c1 * fn(unsafe.Pointer(&(*eface)(unsafe.Pointer(a)).data), uintptr(t.size), h^c0)
	} else {
		return c1 * fn((*eface)(unsafe.Pointer(a)).data, uintptr(t.size), h^c0)
	}
}

func nilinterhash(a *interface{}, s uintptr, h uintptr) uintptr {
	t := (*eface)(unsafe.Pointer(a))._type
	if t == nil {
		return h
	}
	fn := goalg(t.alg).hash
	if **(**uintptr)(unsafe.Pointer(&fn)) == nohashcode {
		// calling nohash will panic too,
		// but we can print a better error.
		panic(errorString("hash of unhashable type " + *t._string))
	}
	if uintptr(t.size) <= ptrSize {
		return c1 * fn(unsafe.Pointer(&(*eface)(unsafe.Pointer(a)).data), uintptr(t.size), h^c0)
	} else {
		return c1 * fn((*eface)(unsafe.Pointer(a)).data, uintptr(t.size), h^c0)
	}
}

// Testing adapters for hash quality tests (see hash_test.go)
func haveGoodHash() bool {
	return use_aeshash
}

func stringHash(s string, seed uintptr) uintptr {
	return goalg(&algarray[alg_STRING]).hash(noescape(unsafe.Pointer(&s)), unsafe.Sizeof(s), seed)
}

func bytesHash(b []byte, seed uintptr) uintptr {
	// TODO: use sliceStruct
	return goalg(&algarray[alg_MEM]).hash(*(*unsafe.Pointer)(unsafe.Pointer(&b)), uintptr(len(b)), seed)
}

func int32Hash(i uint32, seed uintptr) uintptr {
	return goalg(&algarray[alg_MEM32]).hash(noescape(unsafe.Pointer(&i)), 4, seed)
}

func int64Hash(i uint64, seed uintptr) uintptr {
	return goalg(&algarray[alg_MEM64]).hash(noescape(unsafe.Pointer(&i)), 8, seed)
}
