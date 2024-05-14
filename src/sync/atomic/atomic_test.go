// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package atomic_test

import (
	"fmt"
	"reflect"
	"runtime"
	"runtime/debug"
	"strings"
	. "sync/atomic"
	"testing"
	"unsafe"
)

// Tests of correct behavior, without contention.
// (Does the function work as advertised?)
//
// Test that the Add functions add correctly.
// Test that the CompareAndSwap functions actually
// do the comparison and the swap correctly.
//
// The loop over power-of-two values is meant to
// ensure that the operations apply to the full word size.
// The struct fields x.before and x.after check that the
// operations do not extend past the full word size.

const (
	magic32 = 0xdedbeef
	magic64 = 0xdeddeadbeefbeef
)

func TestSwapInt32(t *testing.T) {
	var x struct {
		before int32
		i      int32
		after  int32
	}
	x.before = magic32
	x.after = magic32
	var j int32
	for delta := int32(1); delta+delta > delta; delta += delta {
		k := SwapInt32(&x.i, delta)
		if x.i != delta || k != j {
			t.Fatalf("delta=%d i=%d j=%d k=%d", delta, x.i, j, k)
		}
		j = delta
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestSwapInt32Method(t *testing.T) {
	var x struct {
		before int32
		i      Int32
		after  int32
	}
	x.before = magic32
	x.after = magic32
	var j int32
	for delta := int32(1); delta+delta > delta; delta += delta {
		k := x.i.Swap(delta)
		if x.i.Load() != delta || k != j {
			t.Fatalf("delta=%d i=%d j=%d k=%d", delta, x.i.Load(), j, k)
		}
		j = delta
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestSwapUint32(t *testing.T) {
	var x struct {
		before uint32
		i      uint32
		after  uint32
	}
	x.before = magic32
	x.after = magic32
	var j uint32
	for delta := uint32(1); delta+delta > delta; delta += delta {
		k := SwapUint32(&x.i, delta)
		if x.i != delta || k != j {
			t.Fatalf("delta=%d i=%d j=%d k=%d", delta, x.i, j, k)
		}
		j = delta
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestSwapUint32Method(t *testing.T) {
	var x struct {
		before uint32
		i      Uint32
		after  uint32
	}
	x.before = magic32
	x.after = magic32
	var j uint32
	for delta := uint32(1); delta+delta > delta; delta += delta {
		k := x.i.Swap(delta)
		if x.i.Load() != delta || k != j {
			t.Fatalf("delta=%d i=%d j=%d k=%d", delta, x.i.Load(), j, k)
		}
		j = delta
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestSwapInt64(t *testing.T) {
	var x struct {
		before int64
		i      int64
		after  int64
	}
	magic64 := int64(magic64)
	x.before = magic64
	x.after = magic64
	var j int64
	for delta := int64(1); delta+delta > delta; delta += delta {
		k := SwapInt64(&x.i, delta)
		if x.i != delta || k != j {
			t.Fatalf("delta=%d i=%d j=%d k=%d", delta, x.i, j, k)
		}
		j = delta
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestSwapInt64Method(t *testing.T) {
	var x struct {
		before int64
		i      Int64
		after  int64
	}
	magic64 := int64(magic64)
	x.before = magic64
	x.after = magic64
	var j int64
	for delta := int64(1); delta+delta > delta; delta += delta {
		k := x.i.Swap(delta)
		if x.i.Load() != delta || k != j {
			t.Fatalf("delta=%d i=%d j=%d k=%d", delta, x.i.Load(), j, k)
		}
		j = delta
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestSwapUint64(t *testing.T) {
	var x struct {
		before uint64
		i      uint64
		after  uint64
	}
	magic64 := uint64(magic64)
	x.before = magic64
	x.after = magic64
	var j uint64
	for delta := uint64(1); delta+delta > delta; delta += delta {
		k := SwapUint64(&x.i, delta)
		if x.i != delta || k != j {
			t.Fatalf("delta=%d i=%d j=%d k=%d", delta, x.i, j, k)
		}
		j = delta
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestSwapUint64Method(t *testing.T) {
	var x struct {
		before uint64
		i      Uint64
		after  uint64
	}
	magic64 := uint64(magic64)
	x.before = magic64
	x.after = magic64
	var j uint64
	for delta := uint64(1); delta+delta > delta; delta += delta {
		k := x.i.Swap(delta)
		if x.i.Load() != delta || k != j {
			t.Fatalf("delta=%d i=%d j=%d k=%d", delta, x.i.Load(), j, k)
		}
		j = delta
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestSwapUintptr(t *testing.T) {
	var x struct {
		before uintptr
		i      uintptr
		after  uintptr
	}
	var m uint64 = magic64
	magicptr := uintptr(m)
	x.before = magicptr
	x.after = magicptr
	var j uintptr
	for delta := uintptr(1); delta+delta > delta; delta += delta {
		k := SwapUintptr(&x.i, delta)
		if x.i != delta || k != j {
			t.Fatalf("delta=%d i=%d j=%d k=%d", delta, x.i, j, k)
		}
		j = delta
	}
	if x.before != magicptr || x.after != magicptr {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magicptr, magicptr)
	}
}

func TestSwapUintptrMethod(t *testing.T) {
	var x struct {
		before uintptr
		i      Uintptr
		after  uintptr
	}
	var m uint64 = magic64
	magicptr := uintptr(m)
	x.before = magicptr
	x.after = magicptr
	var j uintptr
	for delta := uintptr(1); delta+delta > delta; delta += delta {
		k := x.i.Swap(delta)
		if x.i.Load() != delta || k != j {
			t.Fatalf("delta=%d i=%d j=%d k=%d", delta, x.i.Load(), j, k)
		}
		j = delta
	}
	if x.before != magicptr || x.after != magicptr {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magicptr, magicptr)
	}
}

var global [1024]byte

func testPointers() []unsafe.Pointer {
	var pointers []unsafe.Pointer
	// globals
	for i := 0; i < 10; i++ {
		pointers = append(pointers, unsafe.Pointer(&global[1<<i-1]))
	}
	// heap
	pointers = append(pointers, unsafe.Pointer(new(byte)))
	// nil
	pointers = append(pointers, nil)
	return pointers
}

func TestSwapPointer(t *testing.T) {
	var x struct {
		before uintptr
		i      unsafe.Pointer
		after  uintptr
	}
	var m uint64 = magic64
	magicptr := uintptr(m)
	x.before = magicptr
	x.after = magicptr
	var j unsafe.Pointer

	for _, p := range testPointers() {
		k := SwapPointer(&x.i, p)
		if x.i != p || k != j {
			t.Fatalf("p=%p i=%p j=%p k=%p", p, x.i, j, k)
		}
		j = p
	}
	if x.before != magicptr || x.after != magicptr {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magicptr, magicptr)
	}
}

func TestSwapPointerMethod(t *testing.T) {
	var x struct {
		before uintptr
		i      Pointer[byte]
		after  uintptr
	}
	var m uint64 = magic64
	magicptr := uintptr(m)
	x.before = magicptr
	x.after = magicptr
	var j *byte
	for _, p := range testPointers() {
		p := (*byte)(p)
		k := x.i.Swap(p)
		if x.i.Load() != p || k != j {
			t.Fatalf("p=%p i=%p j=%p k=%p", p, x.i.Load(), j, k)
		}
		j = p
	}
	if x.before != magicptr || x.after != magicptr {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magicptr, magicptr)
	}
}

func TestAddInt32(t *testing.T) {
	var x struct {
		before int32
		i      int32
		after  int32
	}
	x.before = magic32
	x.after = magic32
	var j int32
	for delta := int32(1); delta+delta > delta; delta += delta {
		k := AddInt32(&x.i, delta)
		j += delta
		if x.i != j || k != j {
			t.Fatalf("delta=%d i=%d j=%d k=%d", delta, x.i, j, k)
		}
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestAddInt32Method(t *testing.T) {
	var x struct {
		before int32
		i      Int32
		after  int32
	}
	x.before = magic32
	x.after = magic32
	var j int32
	for delta := int32(1); delta+delta > delta; delta += delta {
		k := x.i.Add(delta)
		j += delta
		if x.i.Load() != j || k != j {
			t.Fatalf("delta=%d i=%d j=%d k=%d", delta, x.i.Load(), j, k)
		}
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestAddUint32(t *testing.T) {
	var x struct {
		before uint32
		i      uint32
		after  uint32
	}
	x.before = magic32
	x.after = magic32
	var j uint32
	for delta := uint32(1); delta+delta > delta; delta += delta {
		k := AddUint32(&x.i, delta)
		j += delta
		if x.i != j || k != j {
			t.Fatalf("delta=%d i=%d j=%d k=%d", delta, x.i, j, k)
		}
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestAddUint32Method(t *testing.T) {
	var x struct {
		before uint32
		i      Uint32
		after  uint32
	}
	x.before = magic32
	x.after = magic32
	var j uint32
	for delta := uint32(1); delta+delta > delta; delta += delta {
		k := x.i.Add(delta)
		j += delta
		if x.i.Load() != j || k != j {
			t.Fatalf("delta=%d i=%d j=%d k=%d", delta, x.i.Load(), j, k)
		}
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestAddInt64(t *testing.T) {
	var x struct {
		before int64
		i      int64
		after  int64
	}
	magic64 := int64(magic64)
	x.before = magic64
	x.after = magic64
	var j int64
	for delta := int64(1); delta+delta > delta; delta += delta {
		k := AddInt64(&x.i, delta)
		j += delta
		if x.i != j || k != j {
			t.Fatalf("delta=%d i=%d j=%d k=%d", delta, x.i, j, k)
		}
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestAddInt64Method(t *testing.T) {
	var x struct {
		before int64
		i      Int64
		after  int64
	}
	magic64 := int64(magic64)
	x.before = magic64
	x.after = magic64
	var j int64
	for delta := int64(1); delta+delta > delta; delta += delta {
		k := x.i.Add(delta)
		j += delta
		if x.i.Load() != j || k != j {
			t.Fatalf("delta=%d i=%d j=%d k=%d", delta, x.i.Load(), j, k)
		}
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestAddUint64(t *testing.T) {
	var x struct {
		before uint64
		i      uint64
		after  uint64
	}
	magic64 := uint64(magic64)
	x.before = magic64
	x.after = magic64
	var j uint64
	for delta := uint64(1); delta+delta > delta; delta += delta {
		k := AddUint64(&x.i, delta)
		j += delta
		if x.i != j || k != j {
			t.Fatalf("delta=%d i=%d j=%d k=%d", delta, x.i, j, k)
		}
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestAddUint64Method(t *testing.T) {
	var x struct {
		before uint64
		i      Uint64
		after  uint64
	}
	magic64 := uint64(magic64)
	x.before = magic64
	x.after = magic64
	var j uint64
	for delta := uint64(1); delta+delta > delta; delta += delta {
		k := x.i.Add(delta)
		j += delta
		if x.i.Load() != j || k != j {
			t.Fatalf("delta=%d i=%d j=%d k=%d", delta, x.i.Load(), j, k)
		}
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestAddUintptr(t *testing.T) {
	var x struct {
		before uintptr
		i      uintptr
		after  uintptr
	}
	var m uint64 = magic64
	magicptr := uintptr(m)
	x.before = magicptr
	x.after = magicptr
	var j uintptr
	for delta := uintptr(1); delta+delta > delta; delta += delta {
		k := AddUintptr(&x.i, delta)
		j += delta
		if x.i != j || k != j {
			t.Fatalf("delta=%d i=%d j=%d k=%d", delta, x.i, j, k)
		}
	}
	if x.before != magicptr || x.after != magicptr {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magicptr, magicptr)
	}
}

func TestAddUintptrMethod(t *testing.T) {
	var x struct {
		before uintptr
		i      Uintptr
		after  uintptr
	}
	var m uint64 = magic64
	magicptr := uintptr(m)
	x.before = magicptr
	x.after = magicptr
	var j uintptr
	for delta := uintptr(1); delta+delta > delta; delta += delta {
		k := x.i.Add(delta)
		j += delta
		if x.i.Load() != j || k != j {
			t.Fatalf("delta=%d i=%d j=%d k=%d", delta, x.i.Load(), j, k)
		}
	}
	if x.before != magicptr || x.after != magicptr {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magicptr, magicptr)
	}
}

func TestAndInt32(t *testing.T) {
	var x struct {
		before int32
		i      int32
		after  int32
	}
	x.before = magic32
	x.after = magic32
	var j int32
	for mask := int32(1); mask != 0; mask <<= 1 {
		old := x.i
		k := AndInt32(&x.i, mask)
		j &= mask
		if x.i != j || k != old {
			t.Fatalf("mask=%d i=%d j=%d k=%d old=%d", mask, x.i, j, k, old)
		}
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestAndInt32Method(t *testing.T) {
	var x struct {
		before int32
		i      Int32
		after  int32
	}
	x.before = magic32
	x.after = magic32
	var j int32
	for mask := int32(1); mask != 0; mask <<= 1 {
		old := x.i.Load()
		k := x.i.And(mask)
		j &= mask
		if x.i.Load() != j || k != old {
			t.Fatalf("mask=%d i=%d j=%d k=%d old=%d", mask, x.i.Load(), j, k, old)
		}
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestAndUint32(t *testing.T) {
	var x struct {
		before uint32
		i      uint32
		after  uint32
	}
	x.before = magic32
	x.after = magic32
	var j uint32
	for mask := uint32(1); mask != 0; mask <<= 1 {
		old := x.i
		k := AndUint32(&x.i, mask)
		j &= mask
		if x.i != j || k != old {
			t.Fatalf("mask=%d i=%d j=%d k=%d old=%d", mask, x.i, j, k, old)
		}
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestAndUint32Method(t *testing.T) {
	var x struct {
		before uint32
		i      Uint32
		after  uint32
	}
	x.before = magic32
	x.after = magic32
	var j uint32
	for mask := uint32(1); mask != 0; mask <<= 1 {
		old := x.i.Load()
		k := x.i.And(mask)
		j &= mask
		if x.i.Load() != j || k != old {
			t.Fatalf("mask=%d i=%d j=%d k=%d old=%d", mask, x.i.Load(), j, k, old)
		}
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestAndInt64(t *testing.T) {
	var x struct {
		before int64
		i      int64
		after  int64
	}
	magic64 := int64(magic64)
	x.before = magic64
	x.after = magic64
	var j int64
	for mask := int64(1); mask != 0; mask <<= 1 {
		old := x.i
		k := AndInt64(&x.i, mask)
		j &= mask
		if x.i != j || k != old {
			t.Fatalf("mask=%d i=%d j=%d k=%d old=%d", mask, x.i, j, k, old)
		}
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestAndInt64Method(t *testing.T) {
	var x struct {
		before int64
		i      Int64
		after  int64
	}
	magic64 := int64(magic64)
	x.before = magic64
	x.after = magic64
	var j int64
	for mask := int64(1); mask != 0; mask <<= 1 {
		old := x.i.Load()
		k := x.i.And(mask)
		j &= mask
		if x.i.Load() != j || k != old {
			t.Fatalf("mask=%d i=%d j=%d k=%d old=%d", mask, x.i.Load(), j, k, old)
		}
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestAndUint64(t *testing.T) {
	var x struct {
		before uint64
		i      uint64
		after  uint64
	}
	magic64 := uint64(magic64)
	x.before = magic64
	x.after = magic64
	var j uint64
	for mask := uint64(1); mask != 0; mask <<= 1 {
		old := x.i
		k := AndUint64(&x.i, mask)
		j &= mask
		if x.i != j || k != old {
			t.Fatalf("mask=%d i=%d j=%d k=%d old=%d", mask, x.i, j, k, old)
		}
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestAndUint64Method(t *testing.T) {
	var x struct {
		before uint64
		i      Uint64
		after  uint64
	}
	magic64 := uint64(magic64)
	x.before = magic64
	x.after = magic64
	var j uint64
	for mask := uint64(1); mask != 0; mask <<= 1 {
		old := x.i.Load()
		k := x.i.And(mask)
		j &= mask
		if x.i.Load() != j || k != old {
			t.Fatalf("mask=%d i=%d j=%d k=%d old=%d", mask, x.i.Load(), j, k, old)
		}
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestAndUintptr(t *testing.T) {
	var x struct {
		before uintptr
		i      uintptr
		after  uintptr
	}
	var m uint64 = magic64
	magicptr := uintptr(m)
	x.before = magicptr
	x.after = magicptr
	var j uintptr
	for mask := uintptr(1); mask != 0; mask <<= 1 {
		old := x.i
		k := AndUintptr(&x.i, mask)
		j &= mask
		if x.i != j || k != old {
			t.Fatalf("mask=%d i=%d j=%d k=%d old=%d", mask, x.i, j, k, old)
		}
	}
	if x.before != magicptr || x.after != magicptr {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magicptr, magicptr)
	}
}

func TestAndUintptrMethod(t *testing.T) {
	var x struct {
		before uintptr
		i      Uintptr
		after  uintptr
	}
	var m uint64 = magic64
	magicptr := uintptr(m)
	x.before = magicptr
	x.after = magicptr
	var j uintptr
	for mask := uintptr(1); mask != 0; mask <<= 1 {
		old := x.i.Load()
		k := x.i.And(mask)
		j &= mask
		if x.i.Load() != j || k != old {
			t.Fatalf("mask=%d i=%d j=%d k=%d old=%d", mask, x.i.Load(), j, k, old)
		}
	}
	if x.before != magicptr || x.after != magicptr {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magicptr, magicptr)
	}
}

func TestOrInt32(t *testing.T) {
	var x struct {
		before int32
		i      int32
		after  int32
	}
	x.before = magic32
	x.after = magic32
	var j int32
	for mask := int32(1); mask != 0; mask <<= 1 {
		old := x.i
		k := OrInt32(&x.i, mask)
		j |= mask
		if x.i != j || k != old {
			t.Fatalf("mask=%d i=%d j=%d k=%d old=%d", mask, x.i, j, k, old)
		}
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestOrInt32Method(t *testing.T) {
	var x struct {
		before int32
		i      Int32
		after  int32
	}
	x.before = magic32
	x.after = magic32
	var j int32
	for mask := int32(1); mask != 0; mask <<= 1 {
		old := x.i.Load()
		k := x.i.Or(mask)
		j |= mask
		if x.i.Load() != j || k != old {
			t.Fatalf("mask=%d i=%d j=%d k=%d old=%d", mask, x.i.Load(), j, k, old)
		}
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestOrUint32(t *testing.T) {
	var x struct {
		before uint32
		i      uint32
		after  uint32
	}
	x.before = magic32
	x.after = magic32
	var j uint32
	for mask := uint32(1); mask != 0; mask <<= 1 {
		old := x.i
		k := OrUint32(&x.i, mask)
		j |= mask
		if x.i != j || k != old {
			t.Fatalf("mask=%d i=%d j=%d k=%d old=%d", mask, x.i, j, k, old)
		}
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestOrUint32Method(t *testing.T) {
	var x struct {
		before uint32
		i      Uint32
		after  uint32
	}
	x.before = magic32
	x.after = magic32
	var j uint32
	for mask := uint32(1); mask != 0; mask <<= 1 {
		old := x.i.Load()
		k := x.i.Or(mask)
		j |= mask
		if x.i.Load() != j || k != old {
			t.Fatalf("mask=%d i=%d j=%d k=%d old=%d", mask, x.i.Load(), j, k, old)
		}
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestOrInt64(t *testing.T) {
	var x struct {
		before int64
		i      int64
		after  int64
	}
	magic64 := int64(magic64)
	x.before = magic64
	x.after = magic64
	var j int64
	for mask := int64(1); mask != 0; mask <<= 1 {
		old := x.i
		k := OrInt64(&x.i, mask)
		j |= mask
		if x.i != j || k != old {
			t.Fatalf("mask=%d i=%d j=%d k=%d old=%d", mask, x.i, j, k, old)
		}
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestOrInt64Method(t *testing.T) {
	var x struct {
		before int64
		i      Int64
		after  int64
	}
	magic64 := int64(magic64)
	x.before = magic64
	x.after = magic64
	var j int64
	for mask := int64(1); mask != 0; mask <<= 1 {
		old := x.i.Load()
		k := x.i.Or(mask)
		j |= mask
		if x.i.Load() != j || k != old {
			t.Fatalf("mask=%d i=%d j=%d k=%d old=%d", mask, x.i.Load(), j, k, old)
		}
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestOrUint64(t *testing.T) {
	var x struct {
		before uint64
		i      uint64
		after  uint64
	}
	magic64 := uint64(magic64)
	x.before = magic64
	x.after = magic64
	var j uint64
	for mask := uint64(1); mask != 0; mask <<= 1 {
		old := x.i
		k := OrUint64(&x.i, mask)
		j |= mask
		if x.i != j || k != old {
			t.Fatalf("mask=%d i=%d j=%d k=%d old=%d", mask, x.i, j, k, old)
		}
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestOrUint64Method(t *testing.T) {
	var x struct {
		before uint64
		i      Uint64
		after  uint64
	}
	magic64 := uint64(magic64)
	x.before = magic64
	x.after = magic64
	var j uint64
	for mask := uint64(1); mask != 0; mask <<= 1 {
		old := x.i.Load()
		k := x.i.Or(mask)
		j |= mask
		if x.i.Load() != j || k != old {
			t.Fatalf("mask=%d i=%d j=%d k=%d old=%d", mask, x.i.Load(), j, k, old)
		}
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestOrUintptr(t *testing.T) {
	var x struct {
		before uintptr
		i      uintptr
		after  uintptr
	}
	var m uint64 = magic64
	magicptr := uintptr(m)
	x.before = magicptr
	x.after = magicptr
	var j uintptr
	for mask := uintptr(1); mask != 0; mask <<= 1 {
		old := x.i
		k := OrUintptr(&x.i, mask)
		j |= mask
		if x.i != j || k != old {
			t.Fatalf("mask=%d i=%d j=%d k=%d old=%d", mask, x.i, j, k, old)
		}
	}
	if x.before != magicptr || x.after != magicptr {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magicptr, magicptr)
	}
}

func TestOrUintptrMethod(t *testing.T) {
	var x struct {
		before uintptr
		i      Uintptr
		after  uintptr
	}
	var m uint64 = magic64
	magicptr := uintptr(m)
	x.before = magicptr
	x.after = magicptr
	var j uintptr
	for mask := uintptr(1); mask != 0; mask <<= 1 {
		old := x.i.Load()
		k := x.i.Or(mask)
		j |= mask
		if x.i.Load() != j || k != old {
			t.Fatalf("mask=%d i=%d j=%d k=%d old=%d", mask, x.i.Load(), j, k, old)
		}
	}
	if x.before != magicptr || x.after != magicptr {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magicptr, magicptr)
	}
}

func TestCompareAndSwapInt32(t *testing.T) {
	var x struct {
		before int32
		i      int32
		after  int32
	}
	x.before = magic32
	x.after = magic32
	for val := int32(1); val+val > val; val += val {
		x.i = val
		if !CompareAndSwapInt32(&x.i, val, val+1) {
			t.Fatalf("should have swapped %#x %#x", val, val+1)
		}
		if x.i != val+1 {
			t.Fatalf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i, val+1)
		}
		x.i = val + 1
		if CompareAndSwapInt32(&x.i, val, val+2) {
			t.Fatalf("should not have swapped %#x %#x", val, val+2)
		}
		if x.i != val+1 {
			t.Fatalf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i, val+1)
		}
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestCompareAndSwapInt32Method(t *testing.T) {
	var x struct {
		before int32
		i      Int32
		after  int32
	}
	x.before = magic32
	x.after = magic32
	for val := int32(1); val+val > val; val += val {
		x.i.Store(val)
		if !x.i.CompareAndSwap(val, val+1) {
			t.Fatalf("should have swapped %#x %#x", val, val+1)
		}
		if x.i.Load() != val+1 {
			t.Fatalf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i.Load(), val+1)
		}
		x.i.Store(val + 1)
		if x.i.CompareAndSwap(val, val+2) {
			t.Fatalf("should not have swapped %#x %#x", val, val+2)
		}
		if x.i.Load() != val+1 {
			t.Fatalf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i.Load(), val+1)
		}
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestCompareAndSwapUint32(t *testing.T) {
	var x struct {
		before uint32
		i      uint32
		after  uint32
	}
	x.before = magic32
	x.after = magic32
	for val := uint32(1); val+val > val; val += val {
		x.i = val
		if !CompareAndSwapUint32(&x.i, val, val+1) {
			t.Fatalf("should have swapped %#x %#x", val, val+1)
		}
		if x.i != val+1 {
			t.Fatalf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i, val+1)
		}
		x.i = val + 1
		if CompareAndSwapUint32(&x.i, val, val+2) {
			t.Fatalf("should not have swapped %#x %#x", val, val+2)
		}
		if x.i != val+1 {
			t.Fatalf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i, val+1)
		}
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestCompareAndSwapUint32Method(t *testing.T) {
	var x struct {
		before uint32
		i      Uint32
		after  uint32
	}
	x.before = magic32
	x.after = magic32
	for val := uint32(1); val+val > val; val += val {
		x.i.Store(val)
		if !x.i.CompareAndSwap(val, val+1) {
			t.Fatalf("should have swapped %#x %#x", val, val+1)
		}
		if x.i.Load() != val+1 {
			t.Fatalf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i.Load(), val+1)
		}
		x.i.Store(val + 1)
		if x.i.CompareAndSwap(val, val+2) {
			t.Fatalf("should not have swapped %#x %#x", val, val+2)
		}
		if x.i.Load() != val+1 {
			t.Fatalf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i.Load(), val+1)
		}
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestCompareAndSwapInt64(t *testing.T) {
	var x struct {
		before int64
		i      int64
		after  int64
	}
	magic64 := int64(magic64)
	x.before = magic64
	x.after = magic64
	for val := int64(1); val+val > val; val += val {
		x.i = val
		if !CompareAndSwapInt64(&x.i, val, val+1) {
			t.Fatalf("should have swapped %#x %#x", val, val+1)
		}
		if x.i != val+1 {
			t.Fatalf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i, val+1)
		}
		x.i = val + 1
		if CompareAndSwapInt64(&x.i, val, val+2) {
			t.Fatalf("should not have swapped %#x %#x", val, val+2)
		}
		if x.i != val+1 {
			t.Fatalf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i, val+1)
		}
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestCompareAndSwapInt64Method(t *testing.T) {
	var x struct {
		before int64
		i      Int64
		after  int64
	}
	magic64 := int64(magic64)
	x.before = magic64
	x.after = magic64
	for val := int64(1); val+val > val; val += val {
		x.i.Store(val)
		if !x.i.CompareAndSwap(val, val+1) {
			t.Fatalf("should have swapped %#x %#x", val, val+1)
		}
		if x.i.Load() != val+1 {
			t.Fatalf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i.Load(), val+1)
		}
		x.i.Store(val + 1)
		if x.i.CompareAndSwap(val, val+2) {
			t.Fatalf("should not have swapped %#x %#x", val, val+2)
		}
		if x.i.Load() != val+1 {
			t.Fatalf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i.Load(), val+1)
		}
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func testCompareAndSwapUint64(t *testing.T, cas func(*uint64, uint64, uint64) bool) {
	var x struct {
		before uint64
		i      uint64
		after  uint64
	}
	magic64 := uint64(magic64)
	x.before = magic64
	x.after = magic64
	for val := uint64(1); val+val > val; val += val {
		x.i = val
		if !cas(&x.i, val, val+1) {
			t.Fatalf("should have swapped %#x %#x", val, val+1)
		}
		if x.i != val+1 {
			t.Fatalf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i, val+1)
		}
		x.i = val + 1
		if cas(&x.i, val, val+2) {
			t.Fatalf("should not have swapped %#x %#x", val, val+2)
		}
		if x.i != val+1 {
			t.Fatalf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i, val+1)
		}
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestCompareAndSwapUint64(t *testing.T) {
	testCompareAndSwapUint64(t, CompareAndSwapUint64)
}

func TestCompareAndSwapUint64Method(t *testing.T) {
	var x struct {
		before uint64
		i      Uint64
		after  uint64
	}
	magic64 := uint64(magic64)
	x.before = magic64
	x.after = magic64
	for val := uint64(1); val+val > val; val += val {
		x.i.Store(val)
		if !x.i.CompareAndSwap(val, val+1) {
			t.Fatalf("should have swapped %#x %#x", val, val+1)
		}
		if x.i.Load() != val+1 {
			t.Fatalf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i.Load(), val+1)
		}
		x.i.Store(val + 1)
		if x.i.CompareAndSwap(val, val+2) {
			t.Fatalf("should not have swapped %#x %#x", val, val+2)
		}
		if x.i.Load() != val+1 {
			t.Fatalf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i.Load(), val+1)
		}
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestCompareAndSwapUintptr(t *testing.T) {
	var x struct {
		before uintptr
		i      uintptr
		after  uintptr
	}
	var m uint64 = magic64
	magicptr := uintptr(m)
	x.before = magicptr
	x.after = magicptr
	for val := uintptr(1); val+val > val; val += val {
		x.i = val
		if !CompareAndSwapUintptr(&x.i, val, val+1) {
			t.Fatalf("should have swapped %#x %#x", val, val+1)
		}
		if x.i != val+1 {
			t.Fatalf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i, val+1)
		}
		x.i = val + 1
		if CompareAndSwapUintptr(&x.i, val, val+2) {
			t.Fatalf("should not have swapped %#x %#x", val, val+2)
		}
		if x.i != val+1 {
			t.Fatalf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i, val+1)
		}
	}
	if x.before != magicptr || x.after != magicptr {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magicptr, magicptr)
	}
}

func TestCompareAndSwapUintptrMethod(t *testing.T) {
	var x struct {
		before uintptr
		i      Uintptr
		after  uintptr
	}
	var m uint64 = magic64
	magicptr := uintptr(m)
	x.before = magicptr
	x.after = magicptr
	for val := uintptr(1); val+val > val; val += val {
		x.i.Store(val)
		if !x.i.CompareAndSwap(val, val+1) {
			t.Fatalf("should have swapped %#x %#x", val, val+1)
		}
		if x.i.Load() != val+1 {
			t.Fatalf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i.Load(), val+1)
		}
		x.i.Store(val + 1)
		if x.i.CompareAndSwap(val, val+2) {
			t.Fatalf("should not have swapped %#x %#x", val, val+2)
		}
		if x.i.Load() != val+1 {
			t.Fatalf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i.Load(), val+1)
		}
	}
	if x.before != magicptr || x.after != magicptr {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, uintptr(magicptr), uintptr(magicptr))
	}
}

func TestCompareAndSwapPointer(t *testing.T) {
	var x struct {
		before uintptr
		i      unsafe.Pointer
		after  uintptr
	}
	var m uint64 = magic64
	magicptr := uintptr(m)
	x.before = magicptr
	x.after = magicptr
	q := unsafe.Pointer(new(byte))
	for _, p := range testPointers() {
		x.i = p
		if !CompareAndSwapPointer(&x.i, p, q) {
			t.Fatalf("should have swapped %p %p", p, q)
		}
		if x.i != q {
			t.Fatalf("wrong x.i after swap: x.i=%p want %p", x.i, q)
		}
		if CompareAndSwapPointer(&x.i, p, nil) {
			t.Fatalf("should not have swapped %p nil", p)
		}
		if x.i != q {
			t.Fatalf("wrong x.i after swap: x.i=%p want %p", x.i, q)
		}
	}
	if x.before != magicptr || x.after != magicptr {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magicptr, magicptr)
	}
}

func TestCompareAndSwapPointerMethod(t *testing.T) {
	var x struct {
		before uintptr
		i      Pointer[byte]
		after  uintptr
	}
	var m uint64 = magic64
	magicptr := uintptr(m)
	x.before = magicptr
	x.after = magicptr
	q := new(byte)
	for _, p := range testPointers() {
		p := (*byte)(p)
		x.i.Store(p)
		if !x.i.CompareAndSwap(p, q) {
			t.Fatalf("should have swapped %p %p", p, q)
		}
		if x.i.Load() != q {
			t.Fatalf("wrong x.i after swap: x.i=%p want %p", x.i.Load(), q)
		}
		if x.i.CompareAndSwap(p, nil) {
			t.Fatalf("should not have swapped %p nil", p)
		}
		if x.i.Load() != q {
			t.Fatalf("wrong x.i after swap: x.i=%p want %p", x.i.Load(), q)
		}
	}
	if x.before != magicptr || x.after != magicptr {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magicptr, magicptr)
	}
}

func TestLoadInt32(t *testing.T) {
	var x struct {
		before int32
		i      int32
		after  int32
	}
	x.before = magic32
	x.after = magic32
	for delta := int32(1); delta+delta > delta; delta += delta {
		k := LoadInt32(&x.i)
		if k != x.i {
			t.Fatalf("delta=%d i=%d k=%d", delta, x.i, k)
		}
		x.i += delta
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestLoadInt32Method(t *testing.T) {
	var x struct {
		before int32
		i      Int32
		after  int32
	}
	x.before = magic32
	x.after = magic32
	want := int32(0)
	for delta := int32(1); delta+delta > delta; delta += delta {
		k := x.i.Load()
		if k != want {
			t.Fatalf("delta=%d i=%d k=%d want=%d", delta, x.i.Load(), k, want)
		}
		x.i.Store(k + delta)
		want = k + delta
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestLoadUint32(t *testing.T) {
	var x struct {
		before uint32
		i      uint32
		after  uint32
	}
	x.before = magic32
	x.after = magic32
	for delta := uint32(1); delta+delta > delta; delta += delta {
		k := LoadUint32(&x.i)
		if k != x.i {
			t.Fatalf("delta=%d i=%d k=%d", delta, x.i, k)
		}
		x.i += delta
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestLoadUint32Method(t *testing.T) {
	var x struct {
		before uint32
		i      Uint32
		after  uint32
	}
	x.before = magic32
	x.after = magic32
	want := uint32(0)
	for delta := uint32(1); delta+delta > delta; delta += delta {
		k := x.i.Load()
		if k != want {
			t.Fatalf("delta=%d i=%d k=%d want=%d", delta, x.i.Load(), k, want)
		}
		x.i.Store(k + delta)
		want = k + delta
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestLoadInt64(t *testing.T) {
	var x struct {
		before int64
		i      int64
		after  int64
	}
	magic64 := int64(magic64)
	x.before = magic64
	x.after = magic64
	for delta := int64(1); delta+delta > delta; delta += delta {
		k := LoadInt64(&x.i)
		if k != x.i {
			t.Fatalf("delta=%d i=%d k=%d", delta, x.i, k)
		}
		x.i += delta
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestLoadInt64Method(t *testing.T) {
	var x struct {
		before int64
		i      Int64
		after  int64
	}
	magic64 := int64(magic64)
	x.before = magic64
	x.after = magic64
	want := int64(0)
	for delta := int64(1); delta+delta > delta; delta += delta {
		k := x.i.Load()
		if k != want {
			t.Fatalf("delta=%d i=%d k=%d want=%d", delta, x.i.Load(), k, want)
		}
		x.i.Store(k + delta)
		want = k + delta
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestLoadUint64(t *testing.T) {
	var x struct {
		before uint64
		i      uint64
		after  uint64
	}
	magic64 := uint64(magic64)
	x.before = magic64
	x.after = magic64
	for delta := uint64(1); delta+delta > delta; delta += delta {
		k := LoadUint64(&x.i)
		if k != x.i {
			t.Fatalf("delta=%d i=%d k=%d", delta, x.i, k)
		}
		x.i += delta
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestLoadUint64Method(t *testing.T) {
	var x struct {
		before uint64
		i      Uint64
		after  uint64
	}
	magic64 := uint64(magic64)
	x.before = magic64
	x.after = magic64
	want := uint64(0)
	for delta := uint64(1); delta+delta > delta; delta += delta {
		k := x.i.Load()
		if k != want {
			t.Fatalf("delta=%d i=%d k=%d want=%d", delta, x.i.Load(), k, want)
		}
		x.i.Store(k + delta)
		want = k + delta
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestLoadUintptr(t *testing.T) {
	var x struct {
		before uintptr
		i      uintptr
		after  uintptr
	}
	var m uint64 = magic64
	magicptr := uintptr(m)
	x.before = magicptr
	x.after = magicptr
	for delta := uintptr(1); delta+delta > delta; delta += delta {
		k := LoadUintptr(&x.i)
		if k != x.i {
			t.Fatalf("delta=%d i=%d k=%d", delta, x.i, k)
		}
		x.i += delta
	}
	if x.before != magicptr || x.after != magicptr {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magicptr, magicptr)
	}
}

func TestLoadUintptrMethod(t *testing.T) {
	var x struct {
		before uintptr
		i      Uintptr
		after  uintptr
	}
	var m uint64 = magic64
	magicptr := uintptr(m)
	x.before = magicptr
	x.after = magicptr
	want := uintptr(0)
	for delta := uintptr(1); delta+delta > delta; delta += delta {
		k := x.i.Load()
		if k != want {
			t.Fatalf("delta=%d i=%d k=%d want=%d", delta, x.i.Load(), k, want)
		}
		x.i.Store(k + delta)
		want = k + delta
	}
	if x.before != magicptr || x.after != magicptr {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magicptr, magicptr)
	}
}

func TestLoadPointer(t *testing.T) {
	var x struct {
		before uintptr
		i      unsafe.Pointer
		after  uintptr
	}
	var m uint64 = magic64
	magicptr := uintptr(m)
	x.before = magicptr
	x.after = magicptr
	for _, p := range testPointers() {
		x.i = p
		k := LoadPointer(&x.i)
		if k != p {
			t.Fatalf("p=%x k=%x", p, k)
		}
	}
	if x.before != magicptr || x.after != magicptr {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magicptr, magicptr)
	}
}

func TestLoadPointerMethod(t *testing.T) {
	var x struct {
		before uintptr
		i      Pointer[byte]
		after  uintptr
	}
	var m uint64 = magic64
	magicptr := uintptr(m)
	x.before = magicptr
	x.after = magicptr
	for _, p := range testPointers() {
		p := (*byte)(p)
		x.i.Store(p)
		k := x.i.Load()
		if k != p {
			t.Fatalf("p=%x k=%x", p, k)
		}
	}
	if x.before != magicptr || x.after != magicptr {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magicptr, magicptr)
	}
}

func TestStoreInt32(t *testing.T) {
	var x struct {
		before int32
		i      int32
		after  int32
	}
	x.before = magic32
	x.after = magic32
	v := int32(0)
	for delta := int32(1); delta+delta > delta; delta += delta {
		StoreInt32(&x.i, v)
		if x.i != v {
			t.Fatalf("delta=%d i=%d v=%d", delta, x.i, v)
		}
		v += delta
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestStoreInt32Method(t *testing.T) {
	var x struct {
		before int32
		i      Int32
		after  int32
	}
	x.before = magic32
	x.after = magic32
	v := int32(0)
	for delta := int32(1); delta+delta > delta; delta += delta {
		x.i.Store(v)
		if x.i.Load() != v {
			t.Fatalf("delta=%d i=%d v=%d", delta, x.i.Load(), v)
		}
		v += delta
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestStoreUint32(t *testing.T) {
	var x struct {
		before uint32
		i      uint32
		after  uint32
	}
	x.before = magic32
	x.after = magic32
	v := uint32(0)
	for delta := uint32(1); delta+delta > delta; delta += delta {
		StoreUint32(&x.i, v)
		if x.i != v {
			t.Fatalf("delta=%d i=%d v=%d", delta, x.i, v)
		}
		v += delta
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestStoreUint32Method(t *testing.T) {
	var x struct {
		before uint32
		i      Uint32
		after  uint32
	}
	x.before = magic32
	x.after = magic32
	v := uint32(0)
	for delta := uint32(1); delta+delta > delta; delta += delta {
		x.i.Store(v)
		if x.i.Load() != v {
			t.Fatalf("delta=%d i=%d v=%d", delta, x.i.Load(), v)
		}
		v += delta
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestStoreInt64(t *testing.T) {
	var x struct {
		before int64
		i      int64
		after  int64
	}
	magic64 := int64(magic64)
	x.before = magic64
	x.after = magic64
	v := int64(0)
	for delta := int64(1); delta+delta > delta; delta += delta {
		StoreInt64(&x.i, v)
		if x.i != v {
			t.Fatalf("delta=%d i=%d v=%d", delta, x.i, v)
		}
		v += delta
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestStoreInt64Method(t *testing.T) {
	var x struct {
		before int64
		i      Int64
		after  int64
	}
	magic64 := int64(magic64)
	x.before = magic64
	x.after = magic64
	v := int64(0)
	for delta := int64(1); delta+delta > delta; delta += delta {
		x.i.Store(v)
		if x.i.Load() != v {
			t.Fatalf("delta=%d i=%d v=%d", delta, x.i.Load(), v)
		}
		v += delta
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestStoreUint64(t *testing.T) {
	var x struct {
		before uint64
		i      uint64
		after  uint64
	}
	magic64 := uint64(magic64)
	x.before = magic64
	x.after = magic64
	v := uint64(0)
	for delta := uint64(1); delta+delta > delta; delta += delta {
		StoreUint64(&x.i, v)
		if x.i != v {
			t.Fatalf("delta=%d i=%d v=%d", delta, x.i, v)
		}
		v += delta
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestStoreUint64Method(t *testing.T) {
	var x struct {
		before uint64
		i      Uint64
		after  uint64
	}
	magic64 := uint64(magic64)
	x.before = magic64
	x.after = magic64
	v := uint64(0)
	for delta := uint64(1); delta+delta > delta; delta += delta {
		x.i.Store(v)
		if x.i.Load() != v {
			t.Fatalf("delta=%d i=%d v=%d", delta, x.i.Load(), v)
		}
		v += delta
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic64, magic64)
	}
}

func TestStoreUintptr(t *testing.T) {
	var x struct {
		before uintptr
		i      uintptr
		after  uintptr
	}
	var m uint64 = magic64
	magicptr := uintptr(m)
	x.before = magicptr
	x.after = magicptr
	v := uintptr(0)
	for delta := uintptr(1); delta+delta > delta; delta += delta {
		StoreUintptr(&x.i, v)
		if x.i != v {
			t.Fatalf("delta=%d i=%d v=%d", delta, x.i, v)
		}
		v += delta
	}
	if x.before != magicptr || x.after != magicptr {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magicptr, magicptr)
	}
}

func TestStoreUintptrMethod(t *testing.T) {
	var x struct {
		before uintptr
		i      Uintptr
		after  uintptr
	}
	var m uint64 = magic64
	magicptr := uintptr(m)
	x.before = magicptr
	x.after = magicptr
	v := uintptr(0)
	for delta := uintptr(1); delta+delta > delta; delta += delta {
		x.i.Store(v)
		if x.i.Load() != v {
			t.Fatalf("delta=%d i=%d v=%d", delta, x.i.Load(), v)
		}
		v += delta
	}
	if x.before != magicptr || x.after != magicptr {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magicptr, magicptr)
	}
}

func TestStorePointer(t *testing.T) {
	var x struct {
		before uintptr
		i      unsafe.Pointer
		after  uintptr
	}
	var m uint64 = magic64
	magicptr := uintptr(m)
	x.before = magicptr
	x.after = magicptr
	for _, p := range testPointers() {
		StorePointer(&x.i, p)
		if x.i != p {
			t.Fatalf("x.i=%p p=%p", x.i, p)
		}
	}
	if x.before != magicptr || x.after != magicptr {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magicptr, magicptr)
	}
}

func TestStorePointerMethod(t *testing.T) {
	var x struct {
		before uintptr
		i      Pointer[byte]
		after  uintptr
	}
	var m uint64 = magic64
	magicptr := uintptr(m)
	x.before = magicptr
	x.after = magicptr
	for _, p := range testPointers() {
		p := (*byte)(p)
		x.i.Store(p)
		if x.i.Load() != p {
			t.Fatalf("x.i=%p p=%p", x.i.Load(), p)
		}
	}
	if x.before != magicptr || x.after != magicptr {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magicptr, magicptr)
	}
}

// Tests of correct behavior, with contention.
// (Is the function atomic?)
//
// For each function, we write a "hammer" function that repeatedly
// uses the atomic operation to add 1 to a value. After running
// multiple hammers in parallel, check that we end with the correct
// total.
// Swap can't add 1, so it uses a different scheme.
// The functions repeatedly generate a pseudo-random number such that
// low bits are equal to high bits, swap, check that the old value
// has low and high bits equal.

var hammer32 = map[string]func(*uint32, int){
	"SwapInt32":             hammerSwapInt32,
	"SwapUint32":            hammerSwapUint32,
	"SwapUintptr":           hammerSwapUintptr32,
	"AddInt32":              hammerAddInt32,
	"AddUint32":             hammerAddUint32,
	"AddUintptr":            hammerAddUintptr32,
	"CompareAndSwapInt32":   hammerCompareAndSwapInt32,
	"CompareAndSwapUint32":  hammerCompareAndSwapUint32,
	"CompareAndSwapUintptr": hammerCompareAndSwapUintptr32,

	"SwapInt32Method":             hammerSwapInt32Method,
	"SwapUint32Method":            hammerSwapUint32Method,
	"SwapUintptrMethod":           hammerSwapUintptr32Method,
	"AddInt32Method":              hammerAddInt32Method,
	"AddUint32Method":             hammerAddUint32Method,
	"AddUintptrMethod":            hammerAddUintptr32Method,
	"CompareAndSwapInt32Method":   hammerCompareAndSwapInt32Method,
	"CompareAndSwapUint32Method":  hammerCompareAndSwapUint32Method,
	"CompareAndSwapUintptrMethod": hammerCompareAndSwapUintptr32Method,
}

func init() {
	var v uint64 = 1 << 50
	if uintptr(v) != 0 {
		// 64-bit system; clear uintptr tests
		delete(hammer32, "SwapUintptr")
		delete(hammer32, "AddUintptr")
		delete(hammer32, "CompareAndSwapUintptr")
		delete(hammer32, "SwapUintptrMethod")
		delete(hammer32, "AddUintptrMethod")
		delete(hammer32, "CompareAndSwapUintptrMethod")
	}
}

func hammerSwapInt32(uaddr *uint32, count int) {
	addr := (*int32)(unsafe.Pointer(uaddr))
	seed := int(uintptr(unsafe.Pointer(&count)))
	for i := 0; i < count; i++ {
		new := uint32(seed+i)<<16 | uint32(seed+i)<<16>>16
		old := uint32(SwapInt32(addr, int32(new)))
		if old>>16 != old<<16>>16 {
			panic(fmt.Sprintf("SwapInt32 is not atomic: %v", old))
		}
	}
}

func hammerSwapInt32Method(uaddr *uint32, count int) {
	addr := (*Int32)(unsafe.Pointer(uaddr))
	seed := int(uintptr(unsafe.Pointer(&count)))
	for i := 0; i < count; i++ {
		new := uint32(seed+i)<<16 | uint32(seed+i)<<16>>16
		old := uint32(addr.Swap(int32(new)))
		if old>>16 != old<<16>>16 {
			panic(fmt.Sprintf("SwapInt32 is not atomic: %v", old))
		}
	}
}

func hammerSwapUint32(addr *uint32, count int) {
	seed := int(uintptr(unsafe.Pointer(&count)))
	for i := 0; i < count; i++ {
		new := uint32(seed+i)<<16 | uint32(seed+i)<<16>>16
		old := SwapUint32(addr, new)
		if old>>16 != old<<16>>16 {
			panic(fmt.Sprintf("SwapUint32 is not atomic: %v", old))
		}
	}
}

func hammerSwapUint32Method(uaddr *uint32, count int) {
	addr := (*Uint32)(unsafe.Pointer(uaddr))
	seed := int(uintptr(unsafe.Pointer(&count)))
	for i := 0; i < count; i++ {
		new := uint32(seed+i)<<16 | uint32(seed+i)<<16>>16
		old := addr.Swap(new)
		if old>>16 != old<<16>>16 {
			panic(fmt.Sprintf("SwapUint32 is not atomic: %v", old))
		}
	}
}

func hammerSwapUintptr32(uaddr *uint32, count int) {
	// only safe when uintptr is 32-bit.
	// not called on 64-bit systems.
	addr := (*uintptr)(unsafe.Pointer(uaddr))
	seed := int(uintptr(unsafe.Pointer(&count)))
	for i := 0; i < count; i++ {
		new := uintptr(seed+i)<<16 | uintptr(seed+i)<<16>>16
		old := SwapUintptr(addr, new)
		if old>>16 != old<<16>>16 {
			panic(fmt.Sprintf("SwapUintptr is not atomic: %#08x", old))
		}
	}
}

func hammerSwapUintptr32Method(uaddr *uint32, count int) {
	// only safe when uintptr is 32-bit.
	// not called on 64-bit systems.
	addr := (*Uintptr)(unsafe.Pointer(uaddr))
	seed := int(uintptr(unsafe.Pointer(&count)))
	for i := 0; i < count; i++ {
		new := uintptr(seed+i)<<16 | uintptr(seed+i)<<16>>16
		old := addr.Swap(new)
		if old>>16 != old<<16>>16 {
			panic(fmt.Sprintf("Uintptr.Swap is not atomic: %#08x", old))
		}
	}
}

func hammerAddInt32(uaddr *uint32, count int) {
	addr := (*int32)(unsafe.Pointer(uaddr))
	for i := 0; i < count; i++ {
		AddInt32(addr, 1)
	}
}

func hammerAddInt32Method(uaddr *uint32, count int) {
	addr := (*Int32)(unsafe.Pointer(uaddr))
	for i := 0; i < count; i++ {
		addr.Add(1)
	}
}

func hammerAddUint32(addr *uint32, count int) {
	for i := 0; i < count; i++ {
		AddUint32(addr, 1)
	}
}

func hammerAddUint32Method(uaddr *uint32, count int) {
	addr := (*Uint32)(unsafe.Pointer(uaddr))
	for i := 0; i < count; i++ {
		addr.Add(1)
	}
}

func hammerAddUintptr32(uaddr *uint32, count int) {
	// only safe when uintptr is 32-bit.
	// not called on 64-bit systems.
	addr := (*uintptr)(unsafe.Pointer(uaddr))
	for i := 0; i < count; i++ {
		AddUintptr(addr, 1)
	}
}

func hammerAddUintptr32Method(uaddr *uint32, count int) {
	// only safe when uintptr is 32-bit.
	// not called on 64-bit systems.
	addr := (*Uintptr)(unsafe.Pointer(uaddr))
	for i := 0; i < count; i++ {
		addr.Add(1)
	}
}

func hammerCompareAndSwapInt32(uaddr *uint32, count int) {
	addr := (*int32)(unsafe.Pointer(uaddr))
	for i := 0; i < count; i++ {
		for {
			v := LoadInt32(addr)
			if CompareAndSwapInt32(addr, v, v+1) {
				break
			}
		}
	}
}

func hammerCompareAndSwapInt32Method(uaddr *uint32, count int) {
	addr := (*Int32)(unsafe.Pointer(uaddr))
	for i := 0; i < count; i++ {
		for {
			v := addr.Load()
			if addr.CompareAndSwap(v, v+1) {
				break
			}
		}
	}
}

func hammerCompareAndSwapUint32(addr *uint32, count int) {
	for i := 0; i < count; i++ {
		for {
			v := LoadUint32(addr)
			if CompareAndSwapUint32(addr, v, v+1) {
				break
			}
		}
	}
}

func hammerCompareAndSwapUint32Method(uaddr *uint32, count int) {
	addr := (*Uint32)(unsafe.Pointer(uaddr))
	for i := 0; i < count; i++ {
		for {
			v := addr.Load()
			if addr.CompareAndSwap(v, v+1) {
				break
			}
		}
	}
}

func hammerCompareAndSwapUintptr32(uaddr *uint32, count int) {
	// only safe when uintptr is 32-bit.
	// not called on 64-bit systems.
	addr := (*uintptr)(unsafe.Pointer(uaddr))
	for i := 0; i < count; i++ {
		for {
			v := LoadUintptr(addr)
			if CompareAndSwapUintptr(addr, v, v+1) {
				break
			}
		}
	}
}

func hammerCompareAndSwapUintptr32Method(uaddr *uint32, count int) {
	// only safe when uintptr is 32-bit.
	// not called on 64-bit systems.
	addr := (*Uintptr)(unsafe.Pointer(uaddr))
	for i := 0; i < count; i++ {
		for {
			v := addr.Load()
			if addr.CompareAndSwap(v, v+1) {
				break
			}
		}
	}
}

func TestHammer32(t *testing.T) {
	const p = 4
	n := 100000
	if testing.Short() {
		n = 1000
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(p))

	for name, testf := range hammer32 {
		c := make(chan int)
		var val uint32
		for i := 0; i < p; i++ {
			go func() {
				defer func() {
					if err := recover(); err != nil {
						t.Error(err.(string))
					}
					c <- 1
				}()
				testf(&val, n)
			}()
		}
		for i := 0; i < p; i++ {
			<-c
		}
		if !strings.HasPrefix(name, "Swap") && val != uint32(n)*p {
			t.Fatalf("%s: val=%d want %d", name, val, n*p)
		}
	}
}

var hammer64 = map[string]func(*uint64, int){
	"SwapInt64":             hammerSwapInt64,
	"SwapUint64":            hammerSwapUint64,
	"SwapUintptr":           hammerSwapUintptr64,
	"AddInt64":              hammerAddInt64,
	"AddUint64":             hammerAddUint64,
	"AddUintptr":            hammerAddUintptr64,
	"CompareAndSwapInt64":   hammerCompareAndSwapInt64,
	"CompareAndSwapUint64":  hammerCompareAndSwapUint64,
	"CompareAndSwapUintptr": hammerCompareAndSwapUintptr64,

	"SwapInt64Method":             hammerSwapInt64Method,
	"SwapUint64Method":            hammerSwapUint64Method,
	"SwapUintptrMethod":           hammerSwapUintptr64Method,
	"AddInt64Method":              hammerAddInt64Method,
	"AddUint64Method":             hammerAddUint64Method,
	"AddUintptrMethod":            hammerAddUintptr64Method,
	"CompareAndSwapInt64Method":   hammerCompareAndSwapInt64Method,
	"CompareAndSwapUint64Method":  hammerCompareAndSwapUint64Method,
	"CompareAndSwapUintptrMethod": hammerCompareAndSwapUintptr64Method,
}

func init() {
	var v uint64 = 1 << 50
	if uintptr(v) == 0 {
		// 32-bit system; clear uintptr tests
		delete(hammer64, "SwapUintptr")
		delete(hammer64, "SwapUintptrMethod")
		delete(hammer64, "AddUintptr")
		delete(hammer64, "AddUintptrMethod")
		delete(hammer64, "CompareAndSwapUintptr")
		delete(hammer64, "CompareAndSwapUintptrMethod")
	}
}

func hammerSwapInt64(uaddr *uint64, count int) {
	addr := (*int64)(unsafe.Pointer(uaddr))
	seed := int(uintptr(unsafe.Pointer(&count)))
	for i := 0; i < count; i++ {
		new := uint64(seed+i)<<32 | uint64(seed+i)<<32>>32
		old := uint64(SwapInt64(addr, int64(new)))
		if old>>32 != old<<32>>32 {
			panic(fmt.Sprintf("SwapInt64 is not atomic: %v", old))
		}
	}
}

func hammerSwapInt64Method(uaddr *uint64, count int) {
	addr := (*Int64)(unsafe.Pointer(uaddr))
	seed := int(uintptr(unsafe.Pointer(&count)))
	for i := 0; i < count; i++ {
		new := uint64(seed+i)<<32 | uint64(seed+i)<<32>>32
		old := uint64(addr.Swap(int64(new)))
		if old>>32 != old<<32>>32 {
			panic(fmt.Sprintf("SwapInt64 is not atomic: %v", old))
		}
	}
}

func hammerSwapUint64(addr *uint64, count int) {
	seed := int(uintptr(unsafe.Pointer(&count)))
	for i := 0; i < count; i++ {
		new := uint64(seed+i)<<32 | uint64(seed+i)<<32>>32
		old := SwapUint64(addr, new)
		if old>>32 != old<<32>>32 {
			panic(fmt.Sprintf("SwapUint64 is not atomic: %v", old))
		}
	}
}

func hammerSwapUint64Method(uaddr *uint64, count int) {
	addr := (*Uint64)(unsafe.Pointer(uaddr))
	seed := int(uintptr(unsafe.Pointer(&count)))
	for i := 0; i < count; i++ {
		new := uint64(seed+i)<<32 | uint64(seed+i)<<32>>32
		old := addr.Swap(new)
		if old>>32 != old<<32>>32 {
			panic(fmt.Sprintf("SwapUint64 is not atomic: %v", old))
		}
	}
}

const arch32 = unsafe.Sizeof(uintptr(0)) == 4

func hammerSwapUintptr64(uaddr *uint64, count int) {
	// only safe when uintptr is 64-bit.
	// not called on 32-bit systems.
	if !arch32 {
		addr := (*uintptr)(unsafe.Pointer(uaddr))
		seed := int(uintptr(unsafe.Pointer(&count)))
		for i := 0; i < count; i++ {
			new := uintptr(seed+i)<<32 | uintptr(seed+i)<<32>>32
			old := SwapUintptr(addr, new)
			if old>>32 != old<<32>>32 {
				panic(fmt.Sprintf("SwapUintptr is not atomic: %v", old))
			}
		}
	}
}

func hammerSwapUintptr64Method(uaddr *uint64, count int) {
	// only safe when uintptr is 64-bit.
	// not called on 32-bit systems.
	if !arch32 {
		addr := (*Uintptr)(unsafe.Pointer(uaddr))
		seed := int(uintptr(unsafe.Pointer(&count)))
		for i := 0; i < count; i++ {
			new := uintptr(seed+i)<<32 | uintptr(seed+i)<<32>>32
			old := addr.Swap(new)
			if old>>32 != old<<32>>32 {
				panic(fmt.Sprintf("SwapUintptr is not atomic: %v", old))
			}
		}
	}
}

func hammerAddInt64(uaddr *uint64, count int) {
	addr := (*int64)(unsafe.Pointer(uaddr))
	for i := 0; i < count; i++ {
		AddInt64(addr, 1)
	}
}

func hammerAddInt64Method(uaddr *uint64, count int) {
	addr := (*Int64)(unsafe.Pointer(uaddr))
	for i := 0; i < count; i++ {
		addr.Add(1)
	}
}

func hammerAddUint64(addr *uint64, count int) {
	for i := 0; i < count; i++ {
		AddUint64(addr, 1)
	}
}

func hammerAddUint64Method(uaddr *uint64, count int) {
	addr := (*Uint64)(unsafe.Pointer(uaddr))
	for i := 0; i < count; i++ {
		addr.Add(1)
	}
}

func hammerAddUintptr64(uaddr *uint64, count int) {
	// only safe when uintptr is 64-bit.
	// not called on 32-bit systems.
	addr := (*uintptr)(unsafe.Pointer(uaddr))
	for i := 0; i < count; i++ {
		AddUintptr(addr, 1)
	}
}

func hammerAddUintptr64Method(uaddr *uint64, count int) {
	// only safe when uintptr is 64-bit.
	// not called on 32-bit systems.
	addr := (*Uintptr)(unsafe.Pointer(uaddr))
	for i := 0; i < count; i++ {
		addr.Add(1)
	}
}

func hammerCompareAndSwapInt64(uaddr *uint64, count int) {
	addr := (*int64)(unsafe.Pointer(uaddr))
	for i := 0; i < count; i++ {
		for {
			v := LoadInt64(addr)
			if CompareAndSwapInt64(addr, v, v+1) {
				break
			}
		}
	}
}

func hammerCompareAndSwapInt64Method(uaddr *uint64, count int) {
	addr := (*Int64)(unsafe.Pointer(uaddr))
	for i := 0; i < count; i++ {
		for {
			v := addr.Load()
			if addr.CompareAndSwap(v, v+1) {
				break
			}
		}
	}
}

func hammerCompareAndSwapUint64(addr *uint64, count int) {
	for i := 0; i < count; i++ {
		for {
			v := LoadUint64(addr)
			if CompareAndSwapUint64(addr, v, v+1) {
				break
			}
		}
	}
}

func hammerCompareAndSwapUint64Method(uaddr *uint64, count int) {
	addr := (*Uint64)(unsafe.Pointer(uaddr))
	for i := 0; i < count; i++ {
		for {
			v := addr.Load()
			if addr.CompareAndSwap(v, v+1) {
				break
			}
		}
	}
}

func hammerCompareAndSwapUintptr64(uaddr *uint64, count int) {
	// only safe when uintptr is 64-bit.
	// not called on 32-bit systems.
	addr := (*uintptr)(unsafe.Pointer(uaddr))
	for i := 0; i < count; i++ {
		for {
			v := LoadUintptr(addr)
			if CompareAndSwapUintptr(addr, v, v+1) {
				break
			}
		}
	}
}

func hammerCompareAndSwapUintptr64Method(uaddr *uint64, count int) {
	// only safe when uintptr is 64-bit.
	// not called on 32-bit systems.
	addr := (*Uintptr)(unsafe.Pointer(uaddr))
	for i := 0; i < count; i++ {
		for {
			v := addr.Load()
			if addr.CompareAndSwap(v, v+1) {
				break
			}
		}
	}
}

func TestHammer64(t *testing.T) {
	const p = 4
	n := 100000
	if testing.Short() {
		n = 1000
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(p))

	for name, testf := range hammer64 {
		c := make(chan int)
		var val uint64
		for i := 0; i < p; i++ {
			go func() {
				defer func() {
					if err := recover(); err != nil {
						t.Error(err.(string))
					}
					c <- 1
				}()
				testf(&val, n)
			}()
		}
		for i := 0; i < p; i++ {
			<-c
		}
		if !strings.HasPrefix(name, "Swap") && val != uint64(n)*p {
			t.Fatalf("%s: val=%d want %d", name, val, n*p)
		}
	}
}

func hammerStoreLoadInt32(t *testing.T, paddr unsafe.Pointer) {
	addr := (*int32)(paddr)
	v := LoadInt32(addr)
	vlo := v & ((1 << 16) - 1)
	vhi := v >> 16
	if vlo != vhi {
		t.Fatalf("Int32: %#x != %#x", vlo, vhi)
	}
	new := v + 1 + 1<<16
	if vlo == 1e4 {
		new = 0
	}
	StoreInt32(addr, new)
}

func hammerStoreLoadInt32Method(t *testing.T, paddr unsafe.Pointer) {
	addr := (*int32)(paddr)
	v := LoadInt32(addr)
	vlo := v & ((1 << 16) - 1)
	vhi := v >> 16
	if vlo != vhi {
		t.Fatalf("Int32: %#x != %#x", vlo, vhi)
	}
	new := v + 1 + 1<<16
	if vlo == 1e4 {
		new = 0
	}
	StoreInt32(addr, new)
}

func hammerStoreLoadUint32(t *testing.T, paddr unsafe.Pointer) {
	addr := (*uint32)(paddr)
	v := LoadUint32(addr)
	vlo := v & ((1 << 16) - 1)
	vhi := v >> 16
	if vlo != vhi {
		t.Fatalf("Uint32: %#x != %#x", vlo, vhi)
	}
	new := v + 1 + 1<<16
	if vlo == 1e4 {
		new = 0
	}
	StoreUint32(addr, new)
}

func hammerStoreLoadUint32Method(t *testing.T, paddr unsafe.Pointer) {
	addr := (*Uint32)(paddr)
	v := addr.Load()
	vlo := v & ((1 << 16) - 1)
	vhi := v >> 16
	if vlo != vhi {
		t.Fatalf("Uint32: %#x != %#x", vlo, vhi)
	}
	new := v + 1 + 1<<16
	if vlo == 1e4 {
		new = 0
	}
	addr.Store(new)
}

func hammerStoreLoadInt64(t *testing.T, paddr unsafe.Pointer) {
	addr := (*int64)(paddr)
	v := LoadInt64(addr)
	vlo := v & ((1 << 32) - 1)
	vhi := v >> 32
	if vlo != vhi {
		t.Fatalf("Int64: %#x != %#x", vlo, vhi)
	}
	new := v + 1 + 1<<32
	StoreInt64(addr, new)
}

func hammerStoreLoadInt64Method(t *testing.T, paddr unsafe.Pointer) {
	addr := (*Int64)(paddr)
	v := addr.Load()
	vlo := v & ((1 << 32) - 1)
	vhi := v >> 32
	if vlo != vhi {
		t.Fatalf("Int64: %#x != %#x", vlo, vhi)
	}
	new := v + 1 + 1<<32
	addr.Store(new)
}

func hammerStoreLoadUint64(t *testing.T, paddr unsafe.Pointer) {
	addr := (*uint64)(paddr)
	v := LoadUint64(addr)
	vlo := v & ((1 << 32) - 1)
	vhi := v >> 32
	if vlo != vhi {
		t.Fatalf("Uint64: %#x != %#x", vlo, vhi)
	}
	new := v + 1 + 1<<32
	StoreUint64(addr, new)
}

func hammerStoreLoadUint64Method(t *testing.T, paddr unsafe.Pointer) {
	addr := (*Uint64)(paddr)
	v := addr.Load()
	vlo := v & ((1 << 32) - 1)
	vhi := v >> 32
	if vlo != vhi {
		t.Fatalf("Uint64: %#x != %#x", vlo, vhi)
	}
	new := v + 1 + 1<<32
	addr.Store(new)
}

func hammerStoreLoadUintptr(t *testing.T, paddr unsafe.Pointer) {
	addr := (*uintptr)(paddr)
	v := LoadUintptr(addr)
	new := v
	if arch32 {
		vlo := v & ((1 << 16) - 1)
		vhi := v >> 16
		if vlo != vhi {
			t.Fatalf("Uintptr: %#x != %#x", vlo, vhi)
		}
		new = v + 1 + 1<<16
		if vlo == 1e4 {
			new = 0
		}
	} else {
		vlo := v & ((1 << 32) - 1)
		vhi := v >> 32
		if vlo != vhi {
			t.Fatalf("Uintptr: %#x != %#x", vlo, vhi)
		}
		inc := uint64(1 + 1<<32)
		new = v + uintptr(inc)
	}
	StoreUintptr(addr, new)
}

//go:nocheckptr
func hammerStoreLoadUintptrMethod(t *testing.T, paddr unsafe.Pointer) {
	addr := (*Uintptr)(paddr)
	v := addr.Load()
	new := v
	if arch32 {
		vlo := v & ((1 << 16) - 1)
		vhi := v >> 16
		if vlo != vhi {
			t.Fatalf("Uintptr: %#x != %#x", vlo, vhi)
		}
		new = v + 1 + 1<<16
		if vlo == 1e4 {
			new = 0
		}
	} else {
		vlo := v & ((1 << 32) - 1)
		vhi := v >> 32
		if vlo != vhi {
			t.Fatalf("Uintptr: %#x != %#x", vlo, vhi)
		}
		inc := uint64(1 + 1<<32)
		new = v + uintptr(inc)
	}
	addr.Store(new)
}

// This code is just testing that LoadPointer/StorePointer operate
// atomically; it's not actually calculating pointers.
//
//go:nocheckptr
func hammerStoreLoadPointer(t *testing.T, paddr unsafe.Pointer) {
	addr := (*unsafe.Pointer)(paddr)
	v := uintptr(LoadPointer(addr))
	new := v
	if arch32 {
		vlo := v & ((1 << 16) - 1)
		vhi := v >> 16
		if vlo != vhi {
			t.Fatalf("Pointer: %#x != %#x", vlo, vhi)
		}
		new = v + 1 + 1<<16
		if vlo == 1e4 {
			new = 0
		}
	} else {
		vlo := v & ((1 << 32) - 1)
		vhi := v >> 32
		if vlo != vhi {
			t.Fatalf("Pointer: %#x != %#x", vlo, vhi)
		}
		inc := uint64(1 + 1<<32)
		new = v + uintptr(inc)
	}
	StorePointer(addr, unsafe.Pointer(new))
}

// This code is just testing that LoadPointer/StorePointer operate
// atomically; it's not actually calculating pointers.
//
//go:nocheckptr
func hammerStoreLoadPointerMethod(t *testing.T, paddr unsafe.Pointer) {
	addr := (*Pointer[byte])(paddr)
	v := uintptr(unsafe.Pointer(addr.Load()))
	new := v
	if arch32 {
		vlo := v & ((1 << 16) - 1)
		vhi := v >> 16
		if vlo != vhi {
			t.Fatalf("Pointer: %#x != %#x", vlo, vhi)
		}
		new = v + 1 + 1<<16
		if vlo == 1e4 {
			new = 0
		}
	} else {
		vlo := v & ((1 << 32) - 1)
		vhi := v >> 32
		if vlo != vhi {
			t.Fatalf("Pointer: %#x != %#x", vlo, vhi)
		}
		inc := uint64(1 + 1<<32)
		new = v + uintptr(inc)
	}
	addr.Store((*byte)(unsafe.Pointer(new)))
}

func TestHammerStoreLoad(t *testing.T) {
	tests := []func(*testing.T, unsafe.Pointer){
		hammerStoreLoadInt32, hammerStoreLoadUint32,
		hammerStoreLoadUintptr, hammerStoreLoadPointer,
		hammerStoreLoadInt32Method, hammerStoreLoadUint32Method,
		hammerStoreLoadUintptrMethod, hammerStoreLoadPointerMethod,
		hammerStoreLoadInt64, hammerStoreLoadUint64,
		hammerStoreLoadInt64Method, hammerStoreLoadUint64Method,
	}
	n := int(1e6)
	if testing.Short() {
		n = int(1e4)
	}
	const procs = 8
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(procs))
	// Disable the GC because hammerStoreLoadPointer invokes
	// write barriers on values that aren't real pointers.
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	// Ensure any in-progress GC is finished.
	runtime.GC()
	for _, tt := range tests {
		c := make(chan int)
		var val uint64
		for p := 0; p < procs; p++ {
			go func() {
				for i := 0; i < n; i++ {
					tt(t, unsafe.Pointer(&val))
				}
				c <- 1
			}()
		}
		for p := 0; p < procs; p++ {
			<-c
		}
	}
}

func TestStoreLoadSeqCst32(t *testing.T) {
	if runtime.NumCPU() == 1 {
		t.Skipf("Skipping test on %v processor machine", runtime.NumCPU())
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(4))
	N := int32(1e3)
	if testing.Short() {
		N = int32(1e2)
	}
	c := make(chan bool, 2)
	X := [2]int32{}
	ack := [2][3]int32{{-1, -1, -1}, {-1, -1, -1}}
	for p := 0; p < 2; p++ {
		go func(me int) {
			he := 1 - me
			for i := int32(1); i < N; i++ {
				StoreInt32(&X[me], i)
				my := LoadInt32(&X[he])
				StoreInt32(&ack[me][i%3], my)
				for w := 1; LoadInt32(&ack[he][i%3]) == -1; w++ {
					if w%1000 == 0 {
						runtime.Gosched()
					}
				}
				his := LoadInt32(&ack[he][i%3])
				if (my != i && my != i-1) || (his != i && his != i-1) {
					t.Errorf("invalid values: %d/%d (%d)", my, his, i)
					break
				}
				if my != i && his != i {
					t.Errorf("store/load are not sequentially consistent: %d/%d (%d)", my, his, i)
					break
				}
				StoreInt32(&ack[me][(i-1)%3], -1)
			}
			c <- true
		}(p)
	}
	<-c
	<-c
}

func TestStoreLoadSeqCst64(t *testing.T) {
	if runtime.NumCPU() == 1 {
		t.Skipf("Skipping test on %v processor machine", runtime.NumCPU())
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(4))
	N := int64(1e3)
	if testing.Short() {
		N = int64(1e2)
	}
	c := make(chan bool, 2)
	X := [2]int64{}
	ack := [2][3]int64{{-1, -1, -1}, {-1, -1, -1}}
	for p := 0; p < 2; p++ {
		go func(me int) {
			he := 1 - me
			for i := int64(1); i < N; i++ {
				StoreInt64(&X[me], i)
				my := LoadInt64(&X[he])
				StoreInt64(&ack[me][i%3], my)
				for w := 1; LoadInt64(&ack[he][i%3]) == -1; w++ {
					if w%1000 == 0 {
						runtime.Gosched()
					}
				}
				his := LoadInt64(&ack[he][i%3])
				if (my != i && my != i-1) || (his != i && his != i-1) {
					t.Errorf("invalid values: %d/%d (%d)", my, his, i)
					break
				}
				if my != i && his != i {
					t.Errorf("store/load are not sequentially consistent: %d/%d (%d)", my, his, i)
					break
				}
				StoreInt64(&ack[me][(i-1)%3], -1)
			}
			c <- true
		}(p)
	}
	<-c
	<-c
}

func TestStoreLoadRelAcq32(t *testing.T) {
	if runtime.NumCPU() == 1 {
		t.Skipf("Skipping test on %v processor machine", runtime.NumCPU())
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(4))
	N := int32(1e3)
	if testing.Short() {
		N = int32(1e2)
	}
	c := make(chan bool, 2)
	type Data struct {
		signal int32
		pad1   [128]int8
		data1  int32
		pad2   [128]int8
		data2  float32
	}
	var X Data
	for p := int32(0); p < 2; p++ {
		go func(p int32) {
			for i := int32(1); i < N; i++ {
				if (i+p)%2 == 0 {
					X.data1 = i
					X.data2 = float32(i)
					StoreInt32(&X.signal, i)
				} else {
					for w := 1; LoadInt32(&X.signal) != i; w++ {
						if w%1000 == 0 {
							runtime.Gosched()
						}
					}
					d1 := X.data1
					d2 := X.data2
					if d1 != i || d2 != float32(i) {
						t.Errorf("incorrect data: %d/%g (%d)", d1, d2, i)
						break
					}
				}
			}
			c <- true
		}(p)
	}
	<-c
	<-c
}

func TestStoreLoadRelAcq64(t *testing.T) {
	if runtime.NumCPU() == 1 {
		t.Skipf("Skipping test on %v processor machine", runtime.NumCPU())
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(4))
	N := int64(1e3)
	if testing.Short() {
		N = int64(1e2)
	}
	c := make(chan bool, 2)
	type Data struct {
		signal int64
		pad1   [128]int8
		data1  int64
		pad2   [128]int8
		data2  float64
	}
	var X Data
	for p := int64(0); p < 2; p++ {
		go func(p int64) {
			for i := int64(1); i < N; i++ {
				if (i+p)%2 == 0 {
					X.data1 = i
					X.data2 = float64(i)
					StoreInt64(&X.signal, i)
				} else {
					for w := 1; LoadInt64(&X.signal) != i; w++ {
						if w%1000 == 0 {
							runtime.Gosched()
						}
					}
					d1 := X.data1
					d2 := X.data2
					if d1 != i || d2 != float64(i) {
						t.Errorf("incorrect data: %d/%g (%d)", d1, d2, i)
						break
					}
				}
			}
			c <- true
		}(p)
	}
	<-c
	<-c
}

func shouldPanic(t *testing.T, name string, f func()) {
	defer func() {
		// Check that all GC maps are sane.
		runtime.GC()

		err := recover()
		want := "unaligned 64-bit atomic operation"
		if err == nil {
			t.Errorf("%s did not panic", name)
		} else if s, _ := err.(string); s != want {
			t.Errorf("%s: wanted panic %q, got %q", name, want, err)
		}
	}()
	f()
}

func TestUnaligned64(t *testing.T) {
	// Unaligned 64-bit atomics on 32-bit systems are
	// a continual source of pain. Test that on 32-bit systems they crash
	// instead of failing silently.
	if !arch32 {
		t.Skip("test only runs on 32-bit systems")
	}

	x := make([]uint32, 4)
	p := (*uint64)(unsafe.Pointer(&x[1])) // misaligned

	shouldPanic(t, "LoadUint64", func() { LoadUint64(p) })
	shouldPanic(t, "LoadUint64Method", func() { (*Uint64)(unsafe.Pointer(p)).Load() })
	shouldPanic(t, "StoreUint64", func() { StoreUint64(p, 1) })
	shouldPanic(t, "StoreUint64Method", func() { (*Uint64)(unsafe.Pointer(p)).Store(1) })
	shouldPanic(t, "CompareAndSwapUint64", func() { CompareAndSwapUint64(p, 1, 2) })
	shouldPanic(t, "CompareAndSwapUint64Method", func() { (*Uint64)(unsafe.Pointer(p)).CompareAndSwap(1, 2) })
	shouldPanic(t, "AddUint64", func() { AddUint64(p, 3) })
	shouldPanic(t, "AddUint64Method", func() { (*Uint64)(unsafe.Pointer(p)).Add(3) })
}

func TestAutoAligned64(t *testing.T) {
	var signed struct {
		_ uint32
		i Int64
	}
	if o := reflect.TypeOf(&signed).Elem().Field(1).Offset; o != 8 {
		t.Fatalf("Int64 offset = %d, want 8", o)
	}
	if p := reflect.ValueOf(&signed).Elem().Field(1).Addr().Pointer(); p&7 != 0 {
		t.Fatalf("Int64 pointer = %#x, want 8-aligned", p)
	}

	var unsigned struct {
		_ uint32
		i Uint64
	}
	if o := reflect.TypeOf(&unsigned).Elem().Field(1).Offset; o != 8 {
		t.Fatalf("Uint64 offset = %d, want 8", o)
	}
	if p := reflect.ValueOf(&unsigned).Elem().Field(1).Addr().Pointer(); p&7 != 0 {
		t.Fatalf("Int64 pointer = %#x, want 8-aligned", p)
	}
}

func TestNilDeref(t *testing.T) {
	funcs := [...]func(){
		func() { CompareAndSwapInt32(nil, 0, 0) },
		func() { (*Int32)(nil).CompareAndSwap(0, 0) },
		func() { CompareAndSwapInt64(nil, 0, 0) },
		func() { (*Int64)(nil).CompareAndSwap(0, 0) },
		func() { CompareAndSwapUint32(nil, 0, 0) },
		func() { (*Uint32)(nil).CompareAndSwap(0, 0) },
		func() { CompareAndSwapUint64(nil, 0, 0) },
		func() { (*Uint64)(nil).CompareAndSwap(0, 0) },
		func() { CompareAndSwapUintptr(nil, 0, 0) },
		func() { (*Uintptr)(nil).CompareAndSwap(0, 0) },
		func() { CompareAndSwapPointer(nil, nil, nil) },
		func() { (*Pointer[byte])(nil).CompareAndSwap(nil, nil) },
		func() { SwapInt32(nil, 0) },
		func() { (*Int32)(nil).Swap(0) },
		func() { SwapUint32(nil, 0) },
		func() { (*Uint32)(nil).Swap(0) },
		func() { SwapInt64(nil, 0) },
		func() { (*Int64)(nil).Swap(0) },
		func() { SwapUint64(nil, 0) },
		func() { (*Uint64)(nil).Swap(0) },
		func() { SwapUintptr(nil, 0) },
		func() { (*Uintptr)(nil).Swap(0) },
		func() { SwapPointer(nil, nil) },
		func() { (*Pointer[byte])(nil).Swap(nil) },
		func() { AddInt32(nil, 0) },
		func() { (*Int32)(nil).Add(0) },
		func() { AddUint32(nil, 0) },
		func() { (*Uint32)(nil).Add(0) },
		func() { AddInt64(nil, 0) },
		func() { (*Int64)(nil).Add(0) },
		func() { AddUint64(nil, 0) },
		func() { (*Uint64)(nil).Add(0) },
		func() { AddUintptr(nil, 0) },
		func() { (*Uintptr)(nil).Add(0) },
		func() { LoadInt32(nil) },
		func() { (*Int32)(nil).Load() },
		func() { LoadInt64(nil) },
		func() { (*Int64)(nil).Load() },
		func() { LoadUint32(nil) },
		func() { (*Uint32)(nil).Load() },
		func() { LoadUint64(nil) },
		func() { (*Uint64)(nil).Load() },
		func() { LoadUintptr(nil) },
		func() { (*Uintptr)(nil).Load() },
		func() { LoadPointer(nil) },
		func() { (*Pointer[byte])(nil).Load() },
		func() { StoreInt32(nil, 0) },
		func() { (*Int32)(nil).Store(0) },
		func() { StoreInt64(nil, 0) },
		func() { (*Int64)(nil).Store(0) },
		func() { StoreUint32(nil, 0) },
		func() { (*Uint32)(nil).Store(0) },
		func() { StoreUint64(nil, 0) },
		func() { (*Uint64)(nil).Store(0) },
		func() { StoreUintptr(nil, 0) },
		func() { (*Uintptr)(nil).Store(0) },
		func() { StorePointer(nil, nil) },
		func() { (*Pointer[byte])(nil).Store(nil) },
	}
	for _, f := range funcs {
		func() {
			defer func() {
				runtime.GC()
				recover()
			}()
			f()
		}()
	}
}

// Test that this compiles.
// When atomic.Pointer used _ [0]T, it did not.
type List struct {
	Next Pointer[List]
}
