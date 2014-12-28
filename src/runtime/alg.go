// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

const (
	c0 = uintptr((8-ptrSize)/4*2860486313 + (ptrSize-4)/4*33054211828000289)
	c1 = uintptr((8-ptrSize)/4*3267000013 + (ptrSize-4)/4*23344194077549503)
)

// type algorithms - known to compiler
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

type typeAlg struct {
	// function for hashing objects of this type
	// (ptr to object, size, seed) -> hash
	hash func(unsafe.Pointer, uintptr, uintptr) uintptr
	// function for comparing objects of this type
	// (ptr to object A, ptr to object B, size) -> ==?
	equal func(unsafe.Pointer, unsafe.Pointer, uintptr) bool
}

var algarray = [alg_max]typeAlg{
	alg_MEM:      {memhash, memequal},
	alg_MEM0:     {memhash, memequal0},
	alg_MEM8:     {memhash, memequal8},
	alg_MEM16:    {memhash, memequal16},
	alg_MEM32:    {memhash, memequal32},
	alg_MEM64:    {memhash, memequal64},
	alg_MEM128:   {memhash, memequal128},
	alg_NOEQ:     {nil, nil},
	alg_NOEQ0:    {nil, nil},
	alg_NOEQ8:    {nil, nil},
	alg_NOEQ16:   {nil, nil},
	alg_NOEQ32:   {nil, nil},
	alg_NOEQ64:   {nil, nil},
	alg_NOEQ128:  {nil, nil},
	alg_STRING:   {strhash, strequal},
	alg_INTER:    {interhash, interequal},
	alg_NILINTER: {nilinterhash, nilinterequal},
	alg_SLICE:    {nil, nil},
	alg_FLOAT32:  {f32hash, f32equal},
	alg_FLOAT64:  {f64hash, f64equal},
	alg_CPLX64:   {c64hash, c64equal},
	alg_CPLX128:  {c128hash, c128equal},
}

var useAeshash bool

// in asm_*.s
func aeshash(p unsafe.Pointer, s, h uintptr) uintptr
func aeshash32(p unsafe.Pointer, s, h uintptr) uintptr
func aeshash64(p unsafe.Pointer, s, h uintptr) uintptr
func aeshashstr(p unsafe.Pointer, s, h uintptr) uintptr

func strhash(a unsafe.Pointer, s, h uintptr) uintptr {
	x := (*stringStruct)(a)
	return memhash(x.str, uintptr(x.len), h)
}

// NOTE: Because NaN != NaN, a map can contain any
// number of (mostly useless) entries keyed with NaNs.
// To avoid long hash chains, we assign a random number
// as the hash value for a NaN.

func f32hash(p unsafe.Pointer, s, h uintptr) uintptr {
	f := *(*float32)(p)
	switch {
	case f == 0:
		return c1 * (c0 ^ h) // +0, -0
	case f != f:
		return c1 * (c0 ^ h ^ uintptr(fastrand1())) // any kind of NaN
	default:
		return memhash(p, 4, h)
	}
}

func f64hash(p unsafe.Pointer, s, h uintptr) uintptr {
	f := *(*float64)(p)
	switch {
	case f == 0:
		return c1 * (c0 ^ h) // +0, -0
	case f != f:
		return c1 * (c0 ^ h ^ uintptr(fastrand1())) // any kind of NaN
	default:
		return memhash(p, 8, h)
	}
}

func c64hash(p unsafe.Pointer, s, h uintptr) uintptr {
	x := (*[2]float32)(p)
	return f32hash(unsafe.Pointer(&x[1]), 4, f32hash(unsafe.Pointer(&x[0]), 4, h))
}

func c128hash(p unsafe.Pointer, s, h uintptr) uintptr {
	x := (*[2]float64)(p)
	return f64hash(unsafe.Pointer(&x[1]), 8, f64hash(unsafe.Pointer(&x[0]), 8, h))
}

func interhash(p unsafe.Pointer, s, h uintptr) uintptr {
	a := (*iface)(p)
	tab := a.tab
	if tab == nil {
		return h
	}
	t := tab._type
	fn := t.alg.hash
	if fn == nil {
		panic(errorString("hash of unhashable type " + *t._string))
	}
	if isDirectIface(t) {
		return c1 * fn(unsafe.Pointer(&a.data), uintptr(t.size), h^c0)
	} else {
		return c1 * fn(a.data, uintptr(t.size), h^c0)
	}
}

func nilinterhash(p unsafe.Pointer, s, h uintptr) uintptr {
	a := (*eface)(p)
	t := a._type
	if t == nil {
		return h
	}
	fn := t.alg.hash
	if fn == nil {
		panic(errorString("hash of unhashable type " + *t._string))
	}
	if isDirectIface(t) {
		return c1 * fn(unsafe.Pointer(&a.data), uintptr(t.size), h^c0)
	} else {
		return c1 * fn(a.data, uintptr(t.size), h^c0)
	}
}

func memequal(p, q unsafe.Pointer, size uintptr) bool {
	if p == q {
		return true
	}
	return memeq(p, q, size)
}

func memequal0(p, q unsafe.Pointer, size uintptr) bool {
	return true
}
func memequal8(p, q unsafe.Pointer, size uintptr) bool {
	return *(*int8)(p) == *(*int8)(q)
}
func memequal16(p, q unsafe.Pointer, size uintptr) bool {
	return *(*int16)(p) == *(*int16)(q)
}
func memequal32(p, q unsafe.Pointer, size uintptr) bool {
	return *(*int32)(p) == *(*int32)(q)
}
func memequal64(p, q unsafe.Pointer, size uintptr) bool {
	return *(*int64)(p) == *(*int64)(q)
}
func memequal128(p, q unsafe.Pointer, size uintptr) bool {
	return *(*[2]int64)(p) == *(*[2]int64)(q)
}
func f32equal(p, q unsafe.Pointer, size uintptr) bool {
	return *(*float32)(p) == *(*float32)(q)
}
func f64equal(p, q unsafe.Pointer, size uintptr) bool {
	return *(*float64)(p) == *(*float64)(q)
}
func c64equal(p, q unsafe.Pointer, size uintptr) bool {
	return *(*complex64)(p) == *(*complex64)(q)
}
func c128equal(p, q unsafe.Pointer, size uintptr) bool {
	return *(*complex128)(p) == *(*complex128)(q)
}
func strequal(p, q unsafe.Pointer, size uintptr) bool {
	return *(*string)(p) == *(*string)(q)
}
func interequal(p, q unsafe.Pointer, size uintptr) bool {
	return ifaceeq(*(*interface {
		f()
	})(p), *(*interface {
		f()
	})(q))
}
func nilinterequal(p, q unsafe.Pointer, size uintptr) bool {
	return efaceeq(*(*interface{})(p), *(*interface{})(q))
}
func efaceeq(p, q interface{}) bool {
	x := (*eface)(unsafe.Pointer(&p))
	y := (*eface)(unsafe.Pointer(&q))
	t := x._type
	if t != y._type {
		return false
	}
	if t == nil {
		return true
	}
	eq := t.alg.equal
	if eq == nil {
		panic(errorString("comparing uncomparable type " + *t._string))
	}
	if isDirectIface(t) {
		return eq(noescape(unsafe.Pointer(&x.data)), noescape(unsafe.Pointer(&y.data)), uintptr(t.size))
	}
	return eq(x.data, y.data, uintptr(t.size))
}
func ifaceeq(p, q interface {
	f()
}) bool {
	x := (*iface)(unsafe.Pointer(&p))
	y := (*iface)(unsafe.Pointer(&q))
	xtab := x.tab
	if xtab != y.tab {
		return false
	}
	if xtab == nil {
		return true
	}
	t := xtab._type
	eq := t.alg.equal
	if eq == nil {
		panic(errorString("comparing uncomparable type " + *t._string))
	}
	if isDirectIface(t) {
		return eq(noescape(unsafe.Pointer(&x.data)), noescape(unsafe.Pointer(&y.data)), uintptr(t.size))
	}
	return eq(x.data, y.data, uintptr(t.size))
}

// Testing adapters for hash quality tests (see hash_test.go)
func stringHash(s string, seed uintptr) uintptr {
	return algarray[alg_STRING].hash(noescape(unsafe.Pointer(&s)), unsafe.Sizeof(s), seed)
}

func bytesHash(b []byte, seed uintptr) uintptr {
	s := (*sliceStruct)(unsafe.Pointer(&b))
	return algarray[alg_MEM].hash(s.array, uintptr(s.len), seed)
}

func int32Hash(i uint32, seed uintptr) uintptr {
	return algarray[alg_MEM32].hash(noescape(unsafe.Pointer(&i)), 4, seed)
}

func int64Hash(i uint64, seed uintptr) uintptr {
	return algarray[alg_MEM64].hash(noescape(unsafe.Pointer(&i)), 8, seed)
}

func efaceHash(i interface{}, seed uintptr) uintptr {
	return algarray[alg_NILINTER].hash(noescape(unsafe.Pointer(&i)), unsafe.Sizeof(i), seed)
}

func ifaceHash(i interface {
	F()
}, seed uintptr) uintptr {
	return algarray[alg_INTER].hash(noescape(unsafe.Pointer(&i)), unsafe.Sizeof(i), seed)
}

// Testing adapter for memclr
func memclrBytes(b []byte) {
	s := (*sliceStruct)(unsafe.Pointer(&b))
	memclr(s.array, uintptr(s.len))
}

// used in asm_{386,amd64}.s
const hashRandomBytes = ptrSize / 4 * 64

var aeskeysched [hashRandomBytes]byte

func init() {
	if GOOS == "nacl" {
		return
	}

	// Install aes hash algorithm if we have the instructions we need
	if (cpuid_ecx&(1<<25)) != 0 && // aes (aesenc)
		(cpuid_ecx&(1<<9)) != 0 && // sse3 (pshufb)
		(cpuid_ecx&(1<<19)) != 0 { // sse4.1 (pinsr{d,q})
		useAeshash = true
		algarray[alg_MEM].hash = aeshash
		algarray[alg_MEM8].hash = aeshash
		algarray[alg_MEM16].hash = aeshash
		algarray[alg_MEM32].hash = aeshash32
		algarray[alg_MEM64].hash = aeshash64
		algarray[alg_MEM128].hash = aeshash
		algarray[alg_STRING].hash = aeshashstr
		// Initialize with random data so hash collisions will be hard to engineer.
		getRandomData(aeskeysched[:])
	}
}
