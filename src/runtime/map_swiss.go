// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.swissmap

package runtime

import (
	"internal/abi"
	"internal/runtime/maps"
	"internal/runtime/math"
	"internal/runtime/sys"
	"unsafe"
)

const (
	// TODO: remove? These are used by tests but not the actual map
	loadFactorNum = 7
	loadFactorDen = 8
)

type maptype = abi.SwissMapType

func makemap64(t *abi.SwissMapType, hint int64, m *maps.Map) *maps.Map {
	if int64(int(hint)) != hint {
		hint = 0
	}
	return makemap(t, int(hint), m)
}

// makemap_small implements Go map creation for make(map[k]v) and
// make(map[k]v, hint) when hint is known to be at most bucketCnt
// at compile time and the map needs to be allocated on the heap.
func makemap_small() *maps.Map {
	panic("unimplemented")
}

// checkHint verifies that hint is reasonable, adjusting as necessary.
func checkHint(t *abi.SwissMapType, hint int) uint64 {
	if hint <= 0 {
		return 0
	}

	capacity := uint64(hint)

	// Ensure a groups allocation for a capacity this high doesn't exceed
	// the maximum allocation size.
	//
	// TODO(prattmic): Once we split tables, a large hint will result in
	// splitting the tables up front, which will use smaller individual
	// allocations.
	//
	// TODO(prattmic): This logic is largely duplicated from maps.newTable
	// / maps.(*table).reset.
	capacity, overflow := alignUpPow2(capacity)
	if !overflow {
		groupCount := capacity / abi.SwissMapGroupSlots
		mem, overflow := math.MulUintptr(uintptr(groupCount), t.Group.Size_)
		if overflow || mem > maxAlloc {
			return 0
		}
	} else {
		return 0
	}

	return capacity
}

// makemap implements Go map creation for make(map[k]v, hint).
// If the compiler has determined that the map or the first bucket
// can be created on the stack, h and/or bucket may be non-nil.
// If h != nil, the map can be created directly in h.
// If h.buckets != nil, bucket pointed to can be used as the first bucket.
func makemap(t *abi.SwissMapType, hint int, m *maps.Map) *maps.Map {
	capacity := checkHint(t, hint)

	// TODO: use existing m
	return maps.NewTable(t, capacity)
}

// alignUpPow2 rounds n up to the next power of 2.
//
// Returns true if round up causes overflow.
//
// TODO(prattmic): deduplicate from internal/runtime/maps.
func alignUpPow2(n uint64) (uint64, bool) {
	if n == 0 {
		return 0, false
	}
	v := (uint64(1) << sys.Len64(n-1))
	if v == 0 {
		return 0, true
	}
	return v, false
}

// mapaccess1 returns a pointer to h[key].  Never returns nil, instead
// it will return a reference to the zero object for the elem type if
// the key is not in the map.
// NOTE: The returned pointer may keep the whole map live, so don't
// hold onto it for very long.
func mapaccess1(t *abi.SwissMapType, m *maps.Map, key unsafe.Pointer) unsafe.Pointer {
	// TODO: concurrent checks.
	if raceenabled && m != nil {
		callerpc := sys.GetCallerPC()
		pc := abi.FuncPCABIInternal(mapaccess1)
		racereadpc(unsafe.Pointer(m), callerpc, pc)
		raceReadObjectPC(t.Key, key, callerpc, pc)
	}
	if msanenabled && m != nil {
		msanread(key, t.Key.Size_)
	}
	if asanenabled && m != nil {
		asanread(key, t.Key.Size_)
	}

	if m == nil || m.Used() == 0 {
		if err := mapKeyError(t, key); err != nil {
			panic(err) // see issue 23734
		}
		return unsafe.Pointer(&zeroVal[0])
	}

	elem, ok := m.Get(key)
	if !ok {
		return unsafe.Pointer(&zeroVal[0])
	}
	return elem
}

func mapaccess2(t *abi.SwissMapType, m *maps.Map, key unsafe.Pointer) (unsafe.Pointer, bool) {
	// TODO: concurrent checks.
	if raceenabled && m != nil {
		callerpc := sys.GetCallerPC()
		pc := abi.FuncPCABIInternal(mapaccess2)
		racereadpc(unsafe.Pointer(m), callerpc, pc)
		raceReadObjectPC(t.Key, key, callerpc, pc)
	}
	if msanenabled && m != nil {
		msanread(key, t.Key.Size_)
	}
	if asanenabled && m != nil {
		asanread(key, t.Key.Size_)
	}

	if m == nil || m.Used() == 0 {
		if err := mapKeyError(t, key); err != nil {
			panic(err) // see issue 23734
		}
		return unsafe.Pointer(&zeroVal[0]), false
	}

	elem, ok := m.Get(key)
	if !ok {
		return unsafe.Pointer(&zeroVal[0]), false
	}
	return elem, true
}

func mapaccess1_fat(t *abi.SwissMapType, m *maps.Map, key, zero unsafe.Pointer) unsafe.Pointer {
	e := mapaccess1(t, m, key)
	if e == unsafe.Pointer(&zeroVal[0]) {
		return zero
	}
	return e
}

func mapaccess2_fat(t *abi.SwissMapType, m *maps.Map, key, zero unsafe.Pointer) (unsafe.Pointer, bool) {
	e := mapaccess1(t, m, key)
	if e == unsafe.Pointer(&zeroVal[0]) {
		return zero, false
	}
	return e, true
}

func mapassign(t *abi.SwissMapType, m *maps.Map, key unsafe.Pointer) unsafe.Pointer {
	// TODO: concurrent checks.
	if m == nil {
		panic(plainError("assignment to entry in nil map"))
	}
	if raceenabled {
		callerpc := sys.GetCallerPC()
		pc := abi.FuncPCABIInternal(mapassign)
		racewritepc(unsafe.Pointer(m), callerpc, pc)
		raceReadObjectPC(t.Key, key, callerpc, pc)
	}
	if msanenabled {
		msanread(key, t.Key.Size_)
	}
	if asanenabled {
		asanread(key, t.Key.Size_)
	}

	return m.PutSlot(key)
}

func mapdelete(t *abi.SwissMapType, m *maps.Map, key unsafe.Pointer) {
	// TODO: concurrent checks.
	if raceenabled && m != nil {
		callerpc := sys.GetCallerPC()
		pc := abi.FuncPCABIInternal(mapdelete)
		racewritepc(unsafe.Pointer(m), callerpc, pc)
		raceReadObjectPC(t.Key, key, callerpc, pc)
	}
	if msanenabled && m != nil {
		msanread(key, t.Key.Size_)
	}
	if asanenabled && m != nil {
		asanread(key, t.Key.Size_)
	}

	if m == nil || m.Used() == 0 {
		if err := mapKeyError(t, key); err != nil {
			panic(err) // see issue 23734
		}
		return
	}

	m.Delete(key)
}

// mapiterinit initializes the Iter struct used for ranging over maps.
// The Iter struct pointed to by 'it' is allocated on the stack
// by the compilers order pass or on the heap by reflect_mapiterinit.
// Both need to have zeroed hiter since the struct contains pointers.
func mapiterinit(t *abi.SwissMapType, m *maps.Map, it *maps.Iter) {
	if raceenabled && m != nil {
		callerpc := sys.GetCallerPC()
		racereadpc(unsafe.Pointer(m), callerpc, abi.FuncPCABIInternal(mapiterinit))
	}

	it.Init(t, m)
	it.Next()
}

func mapiternext(it *maps.Iter) {
	// TODO: concurrent checks.
	if raceenabled {
		callerpc := sys.GetCallerPC()
		racereadpc(unsafe.Pointer(it.Map()), callerpc, abi.FuncPCABIInternal(mapiternext))
	}

	it.Next()
}

// mapclear deletes all keys from a map.
func mapclear(t *abi.SwissMapType, m *maps.Map) {
	// TODO: concurrent checks.
	if raceenabled && m != nil {
		callerpc := sys.GetCallerPC()
		pc := abi.FuncPCABIInternal(mapclear)
		racewritepc(unsafe.Pointer(m), callerpc, pc)
	}

	if m == nil || m.Used() == 0 {
		return
	}

	m.Clear()
}

// Reflect stubs. Called from ../reflect/asm_*.s

//go:linkname reflect_makemap reflect.makemap
func reflect_makemap(t *abi.SwissMapType, cap int) *maps.Map {
	// Check invariants and reflects math.
	if t.Key.Equal == nil {
		throw("runtime.reflect_makemap: unsupported map key type")
	}
	// TODO: other checks

	return makemap(t, cap, nil)
}

//go:linkname reflect_mapaccess reflect.mapaccess
func reflect_mapaccess(t *abi.SwissMapType, m *maps.Map, key unsafe.Pointer) unsafe.Pointer {
	elem, ok := mapaccess2(t, m, key)
	if !ok {
		// reflect wants nil for a missing element
		elem = nil
	}
	return elem
}

//go:linkname reflect_mapaccess_faststr reflect.mapaccess_faststr
func reflect_mapaccess_faststr(t *abi.SwissMapType, m *maps.Map, key string) unsafe.Pointer {
	elem, ok := mapaccess2_faststr(t, m, key)
	if !ok {
		// reflect wants nil for a missing element
		elem = nil
	}
	return elem
}

//go:linkname reflect_mapassign reflect.mapassign0
func reflect_mapassign(t *abi.SwissMapType, m *maps.Map, key unsafe.Pointer, elem unsafe.Pointer) {
	p := mapassign(t, m, key)
	typedmemmove(t.Elem, p, elem)
}

//go:linkname reflect_mapassign_faststr reflect.mapassign_faststr0
func reflect_mapassign_faststr(t *abi.SwissMapType, m *maps.Map, key string, elem unsafe.Pointer) {
	p := mapassign_faststr(t, m, key)
	typedmemmove(t.Elem, p, elem)
}

//go:linkname reflect_mapdelete reflect.mapdelete
func reflect_mapdelete(t *abi.SwissMapType, m *maps.Map, key unsafe.Pointer) {
	mapdelete(t, m, key)
}

//go:linkname reflect_mapdelete_faststr reflect.mapdelete_faststr
func reflect_mapdelete_faststr(t *abi.SwissMapType, m *maps.Map, key string) {
	mapdelete_faststr(t, m, key)
}

//go:linkname reflect_mapiterinit reflect.mapiterinit
func reflect_mapiterinit(t *abi.SwissMapType, m *maps.Map, it *maps.Iter) {
	mapiterinit(t, m, it)
}

//go:linkname reflect_mapiternext reflect.mapiternext
func reflect_mapiternext(it *maps.Iter) {
	mapiternext(it)
}

//go:linkname reflect_mapiterkey reflect.mapiterkey
func reflect_mapiterkey(it *maps.Iter) unsafe.Pointer {
	return it.Key()
}

//go:linkname reflect_mapiterelem reflect.mapiterelem
func reflect_mapiterelem(it *maps.Iter) unsafe.Pointer {
	return it.Elem()
}

//go:linkname reflect_maplen reflect.maplen
func reflect_maplen(m *maps.Map) int {
	if m == nil {
		return 0
	}
	if raceenabled {
		callerpc := sys.GetCallerPC()
		racereadpc(unsafe.Pointer(m), callerpc, abi.FuncPCABIInternal(reflect_maplen))
	}
	return int(m.Used())
}

//go:linkname reflect_mapclear reflect.mapclear
func reflect_mapclear(t *abi.SwissMapType, m *maps.Map) {
	mapclear(t, m)
}

//go:linkname reflectlite_maplen internal/reflectlite.maplen
func reflectlite_maplen(m *maps.Map) int {
	if m == nil {
		return 0
	}
	if raceenabled {
		callerpc := sys.GetCallerPC()
		racereadpc(unsafe.Pointer(m), callerpc, abi.FuncPCABIInternal(reflect_maplen))
	}
	return int(m.Used())
}

// mapinitnoop is a no-op function known the Go linker; if a given global
// map (of the right size) is determined to be dead, the linker will
// rewrite the relocation (from the package init func) from the outlined
// map init function to this symbol. Defined in assembly so as to avoid
// complications with instrumentation (coverage, etc).
func mapinitnoop()

// mapclone for implementing maps.Clone
//
//go:linkname mapclone maps.clone
func mapclone(m any) any {
	e := efaceOf(&m)
	e.data = unsafe.Pointer(mapclone2((*abi.SwissMapType)(unsafe.Pointer(e._type)), (*maps.Map)(e.data)))
	return m
}

func mapclone2(t *abi.SwissMapType, src *maps.Map) *maps.Map {
	dst := makemap(t, int(src.Used()), nil)

	var iter maps.Iter
	iter.Init(t, src)
	for iter.Next(); iter.Key() != nil; iter.Next() {
		dst.Put(iter.Key(), iter.Elem())
	}

	return dst
}

// keys for implementing maps.keys
//
//go:linkname keys maps.keys
func keys(m any, p unsafe.Pointer) {
	// Currently unused in the maps package.
	panic("unimplemented")
}

// values for implementing maps.values
//
//go:linkname values maps.values
func values(m any, p unsafe.Pointer) {
	// Currently unused in the maps package.
	panic("unimplemented")
}
