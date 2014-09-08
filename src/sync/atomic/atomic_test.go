// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package atomic_test

import (
	"fmt"
	"runtime"
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

// Do the 64-bit functions panic?  If so, don't bother testing.
var test64err = func() (err interface{}) {
	defer func() {
		err = recover()
	}()
	var x int64
	AddInt64(&x, 1)
	return nil
}()

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

func TestSwapInt64(t *testing.T) {
	if test64err != nil {
		t.Skipf("Skipping 64-bit tests: %v", test64err)
	}
	var x struct {
		before int64
		i      int64
		after  int64
	}
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
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, uint64(magic64), uint64(magic64))
	}
}

func TestSwapUint64(t *testing.T) {
	if test64err != nil {
		t.Skipf("Skipping 64-bit tests: %v", test64err)
	}
	var x struct {
		before uint64
		i      uint64
		after  uint64
	}
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
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, uint64(magic64), uint64(magic64))
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
	var j uintptr
	for delta := uintptr(1); delta+delta > delta; delta += delta {
		k := SwapPointer(&x.i, unsafe.Pointer(delta))
		if uintptr(x.i) != delta || uintptr(k) != j {
			t.Fatalf("delta=%d i=%d j=%d k=%d", delta, x.i, j, k)
		}
		j = delta
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

func TestAddInt64(t *testing.T) {
	if test64err != nil {
		t.Skipf("Skipping 64-bit tests: %v", test64err)
	}
	var x struct {
		before int64
		i      int64
		after  int64
	}
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
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, int64(magic64), int64(magic64))
	}
}

func TestAddUint64(t *testing.T) {
	if test64err != nil {
		t.Skipf("Skipping 64-bit tests: %v", test64err)
	}
	var x struct {
		before uint64
		i      uint64
		after  uint64
	}
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
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, uint64(magic64), uint64(magic64))
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

func TestCompareAndSwapInt64(t *testing.T) {
	if test64err != nil {
		t.Skipf("Skipping 64-bit tests: %v", test64err)
	}
	var x struct {
		before int64
		i      int64
		after  int64
	}
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
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, uint64(magic64), uint64(magic64))
	}
}

func testCompareAndSwapUint64(t *testing.T, cas func(*uint64, uint64, uint64) bool) {
	if test64err != nil {
		t.Skipf("Skipping 64-bit tests: %v", test64err)
	}
	var x struct {
		before uint64
		i      uint64
		after  uint64
	}
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
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, uint64(magic64), uint64(magic64))
	}
}

func TestCompareAndSwapUint64(t *testing.T) {
	testCompareAndSwapUint64(t, CompareAndSwapUint64)
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
	for val := uintptr(1); val+val > val; val += val {
		x.i = unsafe.Pointer(val)
		if !CompareAndSwapPointer(&x.i, unsafe.Pointer(val), unsafe.Pointer(val+1)) {
			t.Fatalf("should have swapped %#x %#x", val, val+1)
		}
		if x.i != unsafe.Pointer(val+1) {
			t.Fatalf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i, val+1)
		}
		x.i = unsafe.Pointer(val + 1)
		if CompareAndSwapPointer(&x.i, unsafe.Pointer(val), unsafe.Pointer(val+2)) {
			t.Fatalf("should not have swapped %#x %#x", val, val+2)
		}
		if x.i != unsafe.Pointer(val+1) {
			t.Fatalf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i, val+1)
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

func TestLoadInt64(t *testing.T) {
	if test64err != nil {
		t.Skipf("Skipping 64-bit tests: %v", test64err)
	}
	var x struct {
		before int64
		i      int64
		after  int64
	}
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
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, uint64(magic64), uint64(magic64))
	}
}

func TestLoadUint64(t *testing.T) {
	if test64err != nil {
		t.Skipf("Skipping 64-bit tests: %v", test64err)
	}
	var x struct {
		before uint64
		i      uint64
		after  uint64
	}
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
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, uint64(magic64), uint64(magic64))
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
	for delta := uintptr(1); delta+delta > delta; delta += delta {
		k := LoadPointer(&x.i)
		if k != x.i {
			t.Fatalf("delta=%d i=%d k=%d", delta, x.i, k)
		}
		x.i = unsafe.Pointer(uintptr(x.i) + delta)
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

func TestStoreInt64(t *testing.T) {
	if test64err != nil {
		t.Skipf("Skipping 64-bit tests: %v", test64err)
	}
	var x struct {
		before int64
		i      int64
		after  int64
	}
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
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, uint64(magic64), uint64(magic64))
	}
}

func TestStoreUint64(t *testing.T) {
	if test64err != nil {
		t.Skipf("Skipping 64-bit tests: %v", test64err)
	}
	var x struct {
		before uint64
		i      uint64
		after  uint64
	}
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
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, uint64(magic64), uint64(magic64))
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
	v := unsafe.Pointer(uintptr(0))
	for delta := uintptr(1); delta+delta > delta; delta += delta {
		StorePointer(&x.i, unsafe.Pointer(v))
		if x.i != v {
			t.Fatalf("delta=%d i=%d v=%d", delta, x.i, v)
		}
		v = unsafe.Pointer(uintptr(v) + delta)
	}
	if x.before != magicptr || x.after != magicptr {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magicptr, magicptr)
	}
}

// Tests of correct behavior, with contention.
// (Is the function atomic?)
//
// For each function, we write a "hammer" function that repeatedly
// uses the atomic operation to add 1 to a value.  After running
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
	"SwapPointer":           hammerSwapPointer32,
	"AddInt32":              hammerAddInt32,
	"AddUint32":             hammerAddUint32,
	"AddUintptr":            hammerAddUintptr32,
	"CompareAndSwapInt32":   hammerCompareAndSwapInt32,
	"CompareAndSwapUint32":  hammerCompareAndSwapUint32,
	"CompareAndSwapUintptr": hammerCompareAndSwapUintptr32,
	"CompareAndSwapPointer": hammerCompareAndSwapPointer32,
}

func init() {
	var v uint64 = 1 << 50
	if uintptr(v) != 0 {
		// 64-bit system; clear uintptr tests
		delete(hammer32, "SwapUintptr")
		delete(hammer32, "SwapPointer")
		delete(hammer32, "AddUintptr")
		delete(hammer32, "CompareAndSwapUintptr")
		delete(hammer32, "CompareAndSwapPointer")
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

func hammerSwapPointer32(uaddr *uint32, count int) {
	// only safe when uintptr is 32-bit.
	// not called on 64-bit systems.
	addr := (*unsafe.Pointer)(unsafe.Pointer(uaddr))
	seed := int(uintptr(unsafe.Pointer(&count)))
	for i := 0; i < count; i++ {
		new := uintptr(seed+i)<<16 | uintptr(seed+i)<<16>>16
		old := uintptr(SwapPointer(addr, unsafe.Pointer(new)))
		if old>>16 != old<<16>>16 {
			panic(fmt.Sprintf("SwapPointer is not atomic: %#08x", old))
		}
	}
}

func hammerAddInt32(uaddr *uint32, count int) {
	addr := (*int32)(unsafe.Pointer(uaddr))
	for i := 0; i < count; i++ {
		AddInt32(addr, 1)
	}
}

func hammerAddUint32(addr *uint32, count int) {
	for i := 0; i < count; i++ {
		AddUint32(addr, 1)
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

func hammerCompareAndSwapPointer32(uaddr *uint32, count int) {
	// only safe when uintptr is 32-bit.
	// not called on 64-bit systems.
	addr := (*unsafe.Pointer)(unsafe.Pointer(uaddr))
	for i := 0; i < count; i++ {
		for {
			v := LoadPointer(addr)
			if CompareAndSwapPointer(addr, v, unsafe.Pointer(uintptr(v)+1)) {
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
	"SwapPointer":           hammerSwapPointer64,
	"AddInt64":              hammerAddInt64,
	"AddUint64":             hammerAddUint64,
	"AddUintptr":            hammerAddUintptr64,
	"CompareAndSwapInt64":   hammerCompareAndSwapInt64,
	"CompareAndSwapUint64":  hammerCompareAndSwapUint64,
	"CompareAndSwapUintptr": hammerCompareAndSwapUintptr64,
	"CompareAndSwapPointer": hammerCompareAndSwapPointer64,
}

func init() {
	var v uint64 = 1 << 50
	if uintptr(v) == 0 {
		// 32-bit system; clear uintptr tests
		delete(hammer64, "SwapUintptr")
		delete(hammer64, "SwapPointer")
		delete(hammer64, "AddUintptr")
		delete(hammer64, "CompareAndSwapUintptr")
		delete(hammer64, "CompareAndSwapPointer")
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

func hammerSwapUintptr64(uaddr *uint64, count int) {
	// only safe when uintptr is 64-bit.
	// not called on 32-bit systems.
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

func hammerSwapPointer64(uaddr *uint64, count int) {
	// only safe when uintptr is 64-bit.
	// not called on 32-bit systems.
	addr := (*unsafe.Pointer)(unsafe.Pointer(uaddr))
	seed := int(uintptr(unsafe.Pointer(&count)))
	for i := 0; i < count; i++ {
		new := uintptr(seed+i)<<32 | uintptr(seed+i)<<32>>32
		old := uintptr(SwapPointer(addr, unsafe.Pointer(new)))
		if old>>32 != old<<32>>32 {
			panic(fmt.Sprintf("SwapPointer is not atomic: %v", old))
		}
	}
}

func hammerAddInt64(uaddr *uint64, count int) {
	addr := (*int64)(unsafe.Pointer(uaddr))
	for i := 0; i < count; i++ {
		AddInt64(addr, 1)
	}
}

func hammerAddUint64(addr *uint64, count int) {
	for i := 0; i < count; i++ {
		AddUint64(addr, 1)
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

func hammerCompareAndSwapPointer64(uaddr *uint64, count int) {
	// only safe when uintptr is 64-bit.
	// not called on 32-bit systems.
	addr := (*unsafe.Pointer)(unsafe.Pointer(uaddr))
	for i := 0; i < count; i++ {
		for {
			v := LoadPointer(addr)
			if CompareAndSwapPointer(addr, v, unsafe.Pointer(uintptr(v)+1)) {
				break
			}
		}
	}
}

func TestHammer64(t *testing.T) {
	if test64err != nil {
		t.Skipf("Skipping 64-bit tests: %v", test64err)
	}
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

func hammerStoreLoadUintptr(t *testing.T, paddr unsafe.Pointer) {
	addr := (*uintptr)(paddr)
	var test64 uint64 = 1 << 50
	arch32 := uintptr(test64) == 0
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

func hammerStoreLoadPointer(t *testing.T, paddr unsafe.Pointer) {
	addr := (*unsafe.Pointer)(paddr)
	var test64 uint64 = 1 << 50
	arch32 := uintptr(test64) == 0
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

func TestHammerStoreLoad(t *testing.T) {
	var tests []func(*testing.T, unsafe.Pointer)
	tests = append(tests, hammerStoreLoadInt32, hammerStoreLoadUint32,
		hammerStoreLoadUintptr, hammerStoreLoadPointer)
	if test64err == nil {
		tests = append(tests, hammerStoreLoadInt64, hammerStoreLoadUint64)
	}
	n := int(1e6)
	if testing.Short() {
		n = int(1e4)
	}
	const procs = 8
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(procs))
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
					t.Fatalf("invalid values: %d/%d (%d)", my, his, i)
				}
				if my != i && his != i {
					t.Fatalf("store/load are not sequentially consistent: %d/%d (%d)", my, his, i)
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
	if test64err != nil {
		t.Skipf("Skipping 64-bit tests: %v", test64err)
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
					t.Fatalf("invalid values: %d/%d (%d)", my, his, i)
				}
				if my != i && his != i {
					t.Fatalf("store/load are not sequentially consistent: %d/%d (%d)", my, his, i)
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
						t.Fatalf("incorrect data: %d/%g (%d)", d1, d2, i)
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
	if test64err != nil {
		t.Skipf("Skipping 64-bit tests: %v", test64err)
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
						t.Fatalf("incorrect data: %d/%g (%d)", d1, d2, i)
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
		if recover() == nil {
			t.Errorf("%s did not panic", name)
		}
	}()
	f()
}

func TestUnaligned64(t *testing.T) {
	// Unaligned 64-bit atomics on 32-bit systems are
	// a continual source of pain. Test that on 32-bit systems they crash
	// instead of failing silently.
	if unsafe.Sizeof(int(0)) != 4 {
		t.Skip("test only runs on 32-bit systems")
	}

	x := make([]uint32, 4)
	p := (*uint64)(unsafe.Pointer(&x[1])) // misaligned

	shouldPanic(t, "LoadUint64", func() { LoadUint64(p) })
	shouldPanic(t, "StoreUint64", func() { StoreUint64(p, 1) })
	shouldPanic(t, "CompareAndSwapUint64", func() { CompareAndSwapUint64(p, 1, 2) })
	shouldPanic(t, "AddUint64", func() { AddUint64(p, 3) })
}

func TestNilDeref(t *testing.T) {
	if p := runtime.GOOS + "/" + runtime.GOARCH; p == "freebsd/arm" || p == "netbsd/arm" {
		t.Skipf("issue 7338: skipping test on %q", p)
	}
	funcs := [...]func(){
		func() { CompareAndSwapInt32(nil, 0, 0) },
		func() { CompareAndSwapInt64(nil, 0, 0) },
		func() { CompareAndSwapUint32(nil, 0, 0) },
		func() { CompareAndSwapUint64(nil, 0, 0) },
		func() { CompareAndSwapUintptr(nil, 0, 0) },
		func() { CompareAndSwapPointer(nil, nil, nil) },
		func() { SwapInt32(nil, 0) },
		func() { SwapUint32(nil, 0) },
		func() { SwapInt64(nil, 0) },
		func() { SwapUint64(nil, 0) },
		func() { SwapUintptr(nil, 0) },
		func() { SwapPointer(nil, nil) },
		func() { AddInt32(nil, 0) },
		func() { AddUint32(nil, 0) },
		func() { AddInt64(nil, 0) },
		func() { AddUint64(nil, 0) },
		func() { AddUintptr(nil, 0) },
		func() { LoadInt32(nil) },
		func() { LoadInt64(nil) },
		func() { LoadUint32(nil) },
		func() { LoadUint64(nil) },
		func() { LoadUintptr(nil) },
		func() { LoadPointer(nil) },
		func() { StoreInt32(nil, 0) },
		func() { StoreInt64(nil, 0) },
		func() { StoreUint32(nil, 0) },
		func() { StoreUint64(nil, 0) },
		func() { StoreUintptr(nil, 0) },
		func() { StorePointer(nil, nil) },
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
