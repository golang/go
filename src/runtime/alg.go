// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"internal/cpu"
	"internal/goarch"
	"unsafe"
)

const (
	c0 = uintptr((8-goarch.PtrSize)/4*2860486313 + (goarch.PtrSize-4)/4*33054211828000289)
	c1 = uintptr((8-goarch.PtrSize)/4*3267000013 + (goarch.PtrSize-4)/4*23344194077549503)
)

func memhash0(p unsafe.Pointer, h uintptr) uintptr {
	return h
}

func memhash8(p unsafe.Pointer, h uintptr) uintptr {
	return memhash(p, h, 1)
}

func memhash16(p unsafe.Pointer, h uintptr) uintptr {
	return memhash(p, h, 2)
}

func memhash128(p unsafe.Pointer, h uintptr) uintptr {
	return memhash(p, h, 16)
}

//go:nosplit
func memhash_varlen(p unsafe.Pointer, h uintptr) uintptr {
	ptr := getclosureptr()
	size := *(*uintptr)(unsafe.Pointer(ptr + unsafe.Sizeof(h)))
	return memhash(p, h, size)
}

// runtime variable to check if the processor we're running on
// actually supports the instructions used by the AES-based
// hash implementation.
var useAeshash bool

// in asm_*.s

// memhash should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/aacfactory/fns
//   - github.com/dgraph-io/ristretto
//   - github.com/minio/simdjson-go
//   - github.com/nbd-wtf/go-nostr
//   - github.com/outcaste-io/ristretto
//   - github.com/puzpuzpuz/xsync/v2
//   - github.com/puzpuzpuz/xsync/v3
//   - github.com/segmentio/parquet-go
//   - github.com/parquet-go/parquet-go
//   - github.com/authzed/spicedb
//   - github.com/pingcap/badger
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname memhash
func memhash(p unsafe.Pointer, h, s uintptr) uintptr

// memhash32 should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/segmentio/parquet-go
//   - github.com/parquet-go/parquet-go
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname memhash32
func memhash32(p unsafe.Pointer, h uintptr) uintptr

// memhash64 should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/segmentio/parquet-go
//   - github.com/parquet-go/parquet-go
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname memhash64
func memhash64(p unsafe.Pointer, h uintptr) uintptr

// strhash should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/aristanetworks/goarista
//   - github.com/bytedance/sonic
//   - github.com/bytedance/go-tagexpr/v2
//   - github.com/cloudwego/dynamicgo
//   - github.com/v2fly/v2ray-core/v5
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname strhash
func strhash(p unsafe.Pointer, h uintptr) uintptr

func strhashFallback(a unsafe.Pointer, h uintptr) uintptr {
	x := (*stringStruct)(a)
	return memhashFallback(x.str, h, uintptr(x.len))
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
		return c1 * (c0 ^ h ^ uintptr(rand())) // any kind of NaN
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
		return c1 * (c0 ^ h ^ uintptr(rand())) // any kind of NaN
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
	t := tab.Type
	if t.Equal == nil {
		// Check hashability here. We could do this check inside
		// typehash, but we want to report the topmost type in
		// the error text (e.g. in a struct with a field of slice type
		// we want to report the struct, not the slice).
		panic(errorString("hash of unhashable type " + toRType(t).string()))
	}
	if isDirectIface(t) {
		return c1 * typehash(t, unsafe.Pointer(&a.data), h^c0)
	} else {
		return c1 * typehash(t, a.data, h^c0)
	}
}

// nilinterhash should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/anacrolix/stm
//   - github.com/aristanetworks/goarista
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname nilinterhash
func nilinterhash(p unsafe.Pointer, h uintptr) uintptr {
	a := (*eface)(p)
	t := a._type
	if t == nil {
		return h
	}
	if t.Equal == nil {
		// See comment in interhash above.
		panic(errorString("hash of unhashable type " + toRType(t).string()))
	}
	if isDirectIface(t) {
		return c1 * typehash(t, unsafe.Pointer(&a.data), h^c0)
	} else {
		return c1 * typehash(t, a.data, h^c0)
	}
}

// typehash computes the hash of the object of type t at address p.
// h is the seed.
// This function is seldom used. Most maps use for hashing either
// fixed functions (e.g. f32hash) or compiler-generated functions
// (e.g. for a type like struct { x, y string }). This implementation
// is slower but more general and is used for hashing interface types
// (called from interhash or nilinterhash, above) or for hashing in
// maps generated by reflect.MapOf (reflect_typehash, below).
// Note: this function must match the compiler generated
// functions exactly. See issue 37716.
//
// typehash should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/puzpuzpuz/xsync/v2
//   - github.com/puzpuzpuz/xsync/v3
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname typehash
func typehash(t *_type, p unsafe.Pointer, h uintptr) uintptr {
	if t.TFlag&abi.TFlagRegularMemory != 0 {
		// Handle ptr sizes specially, see issue 37086.
		switch t.Size_ {
		case 4:
			return memhash32(p, h)
		case 8:
			return memhash64(p, h)
		default:
			return memhash(p, h, t.Size_)
		}
	}
	switch t.Kind_ & abi.KindMask {
	case abi.Float32:
		return f32hash(p, h)
	case abi.Float64:
		return f64hash(p, h)
	case abi.Complex64:
		return c64hash(p, h)
	case abi.Complex128:
		return c128hash(p, h)
	case abi.String:
		return strhash(p, h)
	case abi.Interface:
		i := (*interfacetype)(unsafe.Pointer(t))
		if len(i.Methods) == 0 {
			return nilinterhash(p, h)
		}
		return interhash(p, h)
	case abi.Array:
		a := (*arraytype)(unsafe.Pointer(t))
		for i := uintptr(0); i < a.Len; i++ {
			h = typehash(a.Elem, add(p, i*a.Elem.Size_), h)
		}
		return h
	case abi.Struct:
		s := (*structtype)(unsafe.Pointer(t))
		for _, f := range s.Fields {
			if f.Name.IsBlank() {
				continue
			}
			h = typehash(f.Typ, add(p, f.Offset), h)
		}
		return h
	default:
		// Should never happen, as typehash should only be called
		// with comparable types.
		panic(errorString("hash of unhashable type " + toRType(t).string()))
	}
}

func mapKeyError(t *maptype, p unsafe.Pointer) error {
	if !t.HashMightPanic() {
		return nil
	}
	return mapKeyError2(t.Key, p)
}

func mapKeyError2(t *_type, p unsafe.Pointer) error {
	if t.TFlag&abi.TFlagRegularMemory != 0 {
		return nil
	}
	switch t.Kind_ & abi.KindMask {
	case abi.Float32, abi.Float64, abi.Complex64, abi.Complex128, abi.String:
		return nil
	case abi.Interface:
		i := (*interfacetype)(unsafe.Pointer(t))
		var t *_type
		var pdata *unsafe.Pointer
		if len(i.Methods) == 0 {
			a := (*eface)(p)
			t = a._type
			if t == nil {
				return nil
			}
			pdata = &a.data
		} else {
			a := (*iface)(p)
			if a.tab == nil {
				return nil
			}
			t = a.tab.Type
			pdata = &a.data
		}

		if t.Equal == nil {
			return errorString("hash of unhashable type " + toRType(t).string())
		}

		if isDirectIface(t) {
			return mapKeyError2(t, unsafe.Pointer(pdata))
		} else {
			return mapKeyError2(t, *pdata)
		}
	case abi.Array:
		a := (*arraytype)(unsafe.Pointer(t))
		for i := uintptr(0); i < a.Len; i++ {
			if err := mapKeyError2(a.Elem, add(p, i*a.Elem.Size_)); err != nil {
				return err
			}
		}
		return nil
	case abi.Struct:
		s := (*structtype)(unsafe.Pointer(t))
		for _, f := range s.Fields {
			if f.Name.IsBlank() {
				continue
			}
			if err := mapKeyError2(f.Typ, add(p, f.Offset)); err != nil {
				return err
			}
		}
		return nil
	default:
		// Should never happen, keep this case for robustness.
		return errorString("hash of unhashable type " + toRType(t).string())
	}
}

//go:linkname reflect_typehash reflect.typehash
func reflect_typehash(t *_type, p unsafe.Pointer, h uintptr) uintptr {
	return typehash(t, p, h)
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
	x := *(*iface)(p)
	y := *(*iface)(q)
	return x.tab == y.tab && ifaceeq(x.tab, x.data, y.data)
}
func nilinterequal(p, q unsafe.Pointer) bool {
	x := *(*eface)(p)
	y := *(*eface)(q)
	return x._type == y._type && efaceeq(x._type, x.data, y.data)
}
func efaceeq(t *_type, x, y unsafe.Pointer) bool {
	if t == nil {
		return true
	}
	eq := t.Equal
	if eq == nil {
		panic(errorString("comparing uncomparable type " + toRType(t).string()))
	}
	if isDirectIface(t) {
		// Direct interface types are ptr, chan, map, func, and single-element structs/arrays thereof.
		// Maps and funcs are not comparable, so they can't reach here.
		// Ptrs, chans, and single-element items can be compared directly using ==.
		return x == y
	}
	return eq(x, y)
}
func ifaceeq(tab *itab, x, y unsafe.Pointer) bool {
	if tab == nil {
		return true
	}
	t := tab.Type
	eq := t.Equal
	if eq == nil {
		panic(errorString("comparing uncomparable type " + toRType(t).string()))
	}
	if isDirectIface(t) {
		// See comment in efaceeq.
		return x == y
	}
	return eq(x, y)
}

// Testing adapters for hash quality tests (see hash_test.go)
//
// stringHash should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/k14s/starlark-go
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname stringHash
func stringHash(s string, seed uintptr) uintptr {
	return strhash(noescape(unsafe.Pointer(&s)), seed)
}

func bytesHash(b []byte, seed uintptr) uintptr {
	s := (*slice)(unsafe.Pointer(&b))
	return memhash(s.array, seed, uintptr(s.len))
}

func int32Hash(i uint32, seed uintptr) uintptr {
	return memhash32(noescape(unsafe.Pointer(&i)), seed)
}

func int64Hash(i uint64, seed uintptr) uintptr {
	return memhash64(noescape(unsafe.Pointer(&i)), seed)
}

func efaceHash(i any, seed uintptr) uintptr {
	return nilinterhash(noescape(unsafe.Pointer(&i)), seed)
}

func ifaceHash(i interface {
	F()
}, seed uintptr) uintptr {
	return interhash(noescape(unsafe.Pointer(&i)), seed)
}

const hashRandomBytes = goarch.PtrSize / 4 * 64

// used in asm_{386,amd64,arm64}.s to seed the hash function
var aeskeysched [hashRandomBytes]byte

// used in hash{32,64}.go to seed the hash function
var hashkey [4]uintptr

func alginit() {
	// Install AES hash algorithms if the instructions needed are present.
	if (GOARCH == "386" || GOARCH == "amd64") &&
		cpu.X86.HasAES && // AESENC
		cpu.X86.HasSSSE3 && // PSHUFB
		cpu.X86.HasSSE41 { // PINSR{D,Q}
		initAlgAES()
		return
	}
	if GOARCH == "arm64" && cpu.ARM64.HasAES {
		initAlgAES()
		return
	}
	for i := range hashkey {
		hashkey[i] = uintptr(bootstrapRand())
	}
}

func initAlgAES() {
	useAeshash = true
	// Initialize with random data so hash collisions will be hard to engineer.
	key := (*[hashRandomBytes / 8]uint64)(unsafe.Pointer(&aeskeysched))
	for i := range key {
		key[i] = bootstrapRand()
	}
}

// Note: These routines perform the read with a native endianness.
func readUnaligned32(p unsafe.Pointer) uint32 {
	q := (*[4]byte)(p)
	if goarch.BigEndian {
		return uint32(q[3]) | uint32(q[2])<<8 | uint32(q[1])<<16 | uint32(q[0])<<24
	}
	return uint32(q[0]) | uint32(q[1])<<8 | uint32(q[2])<<16 | uint32(q[3])<<24
}

func readUnaligned64(p unsafe.Pointer) uint64 {
	q := (*[8]byte)(p)
	if goarch.BigEndian {
		return uint64(q[7]) | uint64(q[6])<<8 | uint64(q[5])<<16 | uint64(q[4])<<24 |
			uint64(q[3])<<32 | uint64(q[2])<<40 | uint64(q[1])<<48 | uint64(q[0])<<56
	}
	return uint64(q[0]) | uint64(q[1])<<8 | uint64(q[2])<<16 | uint64(q[3])<<24 | uint64(q[4])<<32 | uint64(q[5])<<40 | uint64(q[6])<<48 | uint64(q[7])<<56
}
