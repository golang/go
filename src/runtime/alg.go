// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/sys"
	"unsafe"
)

const (
	c0 = uintptr((8-sys.PtrSize)/4*2860486313 + (sys.PtrSize-4)/4*33054211828000289)
	c1 = uintptr((8-sys.PtrSize)/4*3267000013 + (sys.PtrSize-4)/4*23344194077549503)
)

// type algorithms - known to compiler
const (
	alg_NOEQ = iota
	alg_MEM0
	alg_MEM8
	alg_MEM16
	alg_MEM32
	alg_MEM64
	alg_MEM128
	alg_STRING
	alg_INTER
	alg_NILINTER
	alg_FLOAT32
	alg_FLOAT64
	alg_CPLX64
	alg_CPLX128
	alg_max
)

// typeAlg is also copied/used in reflect/type.go.
// keep them in sync.
type typeAlg struct {
	// function for hashing objects of this type
	// (ptr to object, seed) -> hash
	hash func(unsafe.Pointer, uintptr) uintptr
	// function for comparing objects of this type
	// (ptr to object A, ptr to object B) -> ==?
	equal func(unsafe.Pointer, unsafe.Pointer) bool
}

func memhash0(p unsafe.Pointer, h uintptr) uintptr {
	return h
}
func memhash8(p unsafe.Pointer, h uintptr) uintptr {
	return memhash(p, h, 1)
}
func memhash16(p unsafe.Pointer, h uintptr) uintptr {
	return memhash(p, h, 2)
}
func memhash32(p unsafe.Pointer, h uintptr) uintptr {
	return memhash(p, h, 4)
}
func memhash64(p unsafe.Pointer, h uintptr) uintptr {
	return memhash(p, h, 8)
}
func memhash128(p unsafe.Pointer, h uintptr) uintptr {
	return memhash(p, h, 16)
}

// memhash_varlen is defined in assembly because it needs access
// to the closure. It appears here to provide an argument
// signature for the assembly routine.
func memhash_varlen(p unsafe.Pointer, h uintptr) uintptr

var algarray = [alg_max]typeAlg{
	alg_NOEQ:     {nil, nil},
	alg_MEM0:     {memhash0, memequal0},
	alg_MEM8:     {memhash8, memequal8},
	alg_MEM16:    {memhash16, memequal16},
	alg_MEM32:    {memhash32, memequal32},
	alg_MEM64:    {memhash64, memequal64},
	alg_MEM128:   {memhash128, memequal128},
	alg_STRING:   {strhash, strequal},
	alg_INTER:    {interhash, interequal},
	alg_NILINTER: {nilinterhash, nilinterequal},
	alg_FLOAT32:  {f32hash, f32equal},
	alg_FLOAT64:  {f64hash, f64equal},
	alg_CPLX64:   {c64hash, c64equal},
	alg_CPLX128:  {c128hash, c128equal},
}

var useAeshash bool

// in asm_*.s
func aeshash(p unsafe.Pointer, h, s uintptr) uintptr
func aeshash32(p unsafe.Pointer, h uintptr) uintptr
func aeshash64(p unsafe.Pointer, h uintptr) uintptr
func aeshashstr(p unsafe.Pointer, h uintptr) uintptr

func strhash(a unsafe.Pointer, h uintptr) uintptr {
	x := (*stringStruct)(a)
	return memhash(x.str, h, uintptr(x.len))
}

// NOTE: Because NaN != NaN, a map can contain any
// number of (mostly useless) entries keyed with NaNs.
// To avoid long hash chains, we assign a random number
// as the hash value for a NaN.

func f32hash(p unsafe.Pointer, h uintptr) uintptr {
	f := *(*float32)(p)
	switch {
	case f == 0:
		return c1 * (c0 ^ h) // +0, -0
	case f != f:
		return c1 * (c0 ^ h ^ uintptr(fastrand())) // any kind of NaN
	default:
		return memhash(p, h, 4)
	}
}

func f64hash(p unsafe.Pointer, h uintptr) uintptr {
	f := *(*float64)(p)
	switch {
	case f == 0:
		return c1 * (c0 ^ h) // +0, -0
	case f != f:
		return c1 * (c0 ^ h ^ uintptr(fastrand())) // any kind of NaN
	default:
		return memhash(p, h, 8)
	}
}

func c64hash(p unsafe.Pointer, h uintptr) uintptr {
	x := (*[2]float32)(p)
	return f32hash(unsafe.Pointer(&x[1]), f32hash(unsafe.Pointer(&x[0]), h))
}

func c128hash(p unsafe.Pointer, h uintptr) uintptr {
	x := (*[2]float64)(p)
	return f64hash(unsafe.Pointer(&x[1]), f64hash(unsafe.Pointer(&x[0]), h))
}

func interhash(p unsafe.Pointer, h uintptr) uintptr {
	a := (*iface)(p)
	tab := a.tab
	if tab == nil {
		return h
	}
	t := tab._type
	fn := t.alg.hash
	if fn == nil {
		panic(errorString("hash of unhashable type " + t.string()))
	}
	if isDirectIface(t) {
		return c1 * fn(unsafe.Pointer(&a.data), h^c0)
	} else {
		return c1 * fn(a.data, h^c0)
	}
}

func nilinterhash(p unsafe.Pointer, h uintptr) uintptr {
	a := (*eface)(p)
	t := a._type
	if t == nil {
		return h
	}
	fn := t.alg.hash
	if fn == nil {
		panic(errorString("hash of unhashable type " + t.string()))
	}
	if isDirectIface(t) {
		return c1 * fn(unsafe.Pointer(&a.data), h^c0)
	} else {
		return c1 * fn(a.data, h^c0)
	}
}

func memequal0(p, q unsafe.Pointer) bool {
	return true
}
func memequal8(p, q unsafe.Pointer) bool {
	return *(*int8)(p) == *(*int8)(q)
}
func memequal16(p, q unsafe.Pointer) bool {
	return *(*int16)(p) == *(*int16)(q)
}
func memequal32(p, q unsafe.Pointer) bool {
	return *(*int32)(p) == *(*int32)(q)
}
func memequal64(p, q unsafe.Pointer) bool {
	return *(*int64)(p) == *(*int64)(q)
}
func memequal128(p, q unsafe.Pointer) bool {
	return *(*[2]int64)(p) == *(*[2]int64)(q)
}
func f32equal(p, q unsafe.Pointer) bool {
	return *(*float32)(p) == *(*float32)(q)
}
func f64equal(p, q unsafe.Pointer) bool {
	return *(*float64)(p) == *(*float64)(q)
}
func c64equal(p, q unsafe.Pointer) bool {
	return *(*complex64)(p) == *(*complex64)(q)
}
func c128equal(p, q unsafe.Pointer) bool {
	return *(*complex128)(p) == *(*complex128)(q)
}
func strequal(p, q unsafe.Pointer) bool {
	return *(*string)(p) == *(*string)(q)
}
func interequal(p, q unsafe.Pointer) bool {
	return ifaceeq(*(*iface)(p), *(*iface)(q))
}
func nilinterequal(p, q unsafe.Pointer) bool {
	return efaceeq(*(*eface)(p), *(*eface)(q))
}
func efaceeq(x, y eface) bool {
	t := x._type
	if t != y._type {
		return false
	}
	if t == nil {
		return true
	}
	eq := t.alg.equal
	if eq == nil {
		panic(errorString("comparing uncomparable type " + t.string()))
	}
	if isDirectIface(t) {
		return eq(noescape(unsafe.Pointer(&x.data)), noescape(unsafe.Pointer(&y.data)))
	}
	return eq(x.data, y.data)
}
func ifaceeq(x, y iface) bool {
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
		panic(errorString("comparing uncomparable type " + t.string()))
	}
	if isDirectIface(t) {
		return eq(noescape(unsafe.Pointer(&x.data)), noescape(unsafe.Pointer(&y.data)))
	}
	return eq(x.data, y.data)
}

// Testing adapters for hash quality tests (see hash_test.go)
func stringHash(s string, seed uintptr) uintptr {
	return algarray[alg_STRING].hash(noescape(unsafe.Pointer(&s)), seed)
}

func bytesHash(b []byte, seed uintptr) uintptr {
	s := (*slice)(unsafe.Pointer(&b))
	return memhash(s.array, seed, uintptr(s.len))
}

func int32Hash(i uint32, seed uintptr) uintptr {
	return algarray[alg_MEM32].hash(noescape(unsafe.Pointer(&i)), seed)
}

func int64Hash(i uint64, seed uintptr) uintptr {
	return algarray[alg_MEM64].hash(noescape(unsafe.Pointer(&i)), seed)
}

func efaceHash(i interface{}, seed uintptr) uintptr {
	return algarray[alg_NILINTER].hash(noescape(unsafe.Pointer(&i)), seed)
}

func ifaceHash(i interface {
	F()
}, seed uintptr) uintptr {
	return algarray[alg_INTER].hash(noescape(unsafe.Pointer(&i)), seed)
}

const hashRandomBytes = sys.PtrSize / 4 * 64

// used in asm_{386,amd64}.s to seed the hash function
var aeskeysched [hashRandomBytes]byte

// used in hash{32,64}.go to seed the hash function
var hashkey [4]uintptr

func alginit() {
	// Install aes hash algorithm if we have the instructions we need
	if (GOARCH == "386" || GOARCH == "amd64") &&
		GOOS != "nacl" &&
		cpuid_ecx&(1<<25) != 0 && // aes (aesenc)
		cpuid_ecx&(1<<9) != 0 && // sse3 (pshufb)
		cpuid_ecx&(1<<19) != 0 { // sse4.1 (pinsr{d,q})
		useAeshash = true
		algarray[alg_MEM32].hash = aeshash32
		algarray[alg_MEM64].hash = aeshash64
		algarray[alg_STRING].hash = aeshashstr
		// Initialize with random data so hash collisions will be hard to engineer.
		getRandomData(aeskeysched[:])
		return
	}
	getRandomData((*[len(hashkey) * sys.PtrSize]byte)(unsafe.Pointer(&hashkey))[:])
	hashkey[0] |= 1 // make sure these numbers are odd
	hashkey[1] |= 1
	hashkey[2] |= 1
	hashkey[3] |= 1
}
