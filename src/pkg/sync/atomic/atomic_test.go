// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package atomic_test

import (
	"runtime"
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
		t.Logf("Skipping 64-bit tests: %v", test64err)
		return
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
		t.Logf("Skipping 64-bit tests: %v", test64err)
		return
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
			t.Errorf("should have swapped %#x %#x", val, val+1)
		}
		if x.i != val+1 {
			t.Errorf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i, val+1)
		}
		x.i = val + 1
		if CompareAndSwapInt32(&x.i, val, val+2) {
			t.Errorf("should not have swapped %#x %#x", val, val+2)
		}
		if x.i != val+1 {
			t.Errorf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i, val+1)
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
			t.Errorf("should have swapped %#x %#x", val, val+1)
		}
		if x.i != val+1 {
			t.Errorf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i, val+1)
		}
		x.i = val + 1
		if CompareAndSwapUint32(&x.i, val, val+2) {
			t.Errorf("should not have swapped %#x %#x", val, val+2)
		}
		if x.i != val+1 {
			t.Errorf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i, val+1)
		}
	}
	if x.before != magic32 || x.after != magic32 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, magic32, magic32)
	}
}

func TestCompareAndSwapInt64(t *testing.T) {
	if test64err != nil {
		t.Logf("Skipping 64-bit tests: %v", test64err)
		return
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
			t.Errorf("should have swapped %#x %#x", val, val+1)
		}
		if x.i != val+1 {
			t.Errorf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i, val+1)
		}
		x.i = val + 1
		if CompareAndSwapInt64(&x.i, val, val+2) {
			t.Errorf("should not have swapped %#x %#x", val, val+2)
		}
		if x.i != val+1 {
			t.Errorf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i, val+1)
		}
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, uint64(magic64), uint64(magic64))
	}
}

func TestCompareAndSwapUint64(t *testing.T) {
	if test64err != nil {
		t.Logf("Skipping 64-bit tests: %v", test64err)
		return
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
		if !CompareAndSwapUint64(&x.i, val, val+1) {
			t.Errorf("should have swapped %#x %#x", val, val+1)
		}
		if x.i != val+1 {
			t.Errorf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i, val+1)
		}
		x.i = val + 1
		if CompareAndSwapUint64(&x.i, val, val+2) {
			t.Errorf("should not have swapped %#x %#x", val, val+2)
		}
		if x.i != val+1 {
			t.Errorf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i, val+1)
		}
	}
	if x.before != magic64 || x.after != magic64 {
		t.Fatalf("wrong magic: %#x _ %#x != %#x _ %#x", x.before, x.after, uint64(magic64), uint64(magic64))
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
			t.Errorf("should have swapped %#x %#x", val, val+1)
		}
		if x.i != val+1 {
			t.Errorf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i, val+1)
		}
		x.i = val + 1
		if CompareAndSwapUintptr(&x.i, val, val+2) {
			t.Errorf("should not have swapped %#x %#x", val, val+2)
		}
		if x.i != val+1 {
			t.Errorf("wrong x.i after swap: x.i=%#x val+1=%#x", x.i, val+1)
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

// Tests of correct behavior, with contention.
// (Is the function atomic?)
//
// For each function, we write a "hammer" function that repeatedly
// uses the atomic operation to add 1 to a value.  After running
// multiple hammers in parallel, check that we end with the correct
// total.

var hammer32 = []struct {
	name string
	f    func(*uint32, int)
}{
	{"AddInt32", hammerAddInt32},
	{"AddUint32", hammerAddUint32},
	{"AddUintptr", hammerAddUintptr32},
	{"CompareAndSwapInt32", hammerCompareAndSwapInt32},
	{"CompareAndSwapUint32", hammerCompareAndSwapUint32},
	{"CompareAndSwapUintptr", hammerCompareAndSwapUintptr32},
}

func init() {
	var v uint64 = 1 << 50
	if uintptr(v) != 0 {
		// 64-bit system; clear uintptr tests
		hammer32[2].f = nil
		hammer32[5].f = nil
	}
}

func hammerAddInt32(uval *uint32, count int) {
	val := (*int32)(unsafe.Pointer(uval))
	for i := 0; i < count; i++ {
		AddInt32(val, 1)
	}
}

func hammerAddUint32(val *uint32, count int) {
	for i := 0; i < count; i++ {
		AddUint32(val, 1)
	}
}

func hammerAddUintptr32(uval *uint32, count int) {
	// only safe when uintptr is 32-bit.
	// not called on 64-bit systems.
	val := (*uintptr)(unsafe.Pointer(uval))
	for i := 0; i < count; i++ {
		AddUintptr(val, 1)
	}
}

func hammerCompareAndSwapInt32(uval *uint32, count int) {
	val := (*int32)(unsafe.Pointer(uval))
	for i := 0; i < count; i++ {
		for {
			v := *val
			if CompareAndSwapInt32(val, v, v+1) {
				break
			}
		}
	}
}

func hammerCompareAndSwapUint32(val *uint32, count int) {
	for i := 0; i < count; i++ {
		for {
			v := *val
			if CompareAndSwapUint32(val, v, v+1) {
				break
			}
		}
	}
}

func hammerCompareAndSwapUintptr32(uval *uint32, count int) {
	// only safe when uintptr is 32-bit.
	// not called on 64-bit systems.
	val := (*uintptr)(unsafe.Pointer(uval))
	for i := 0; i < count; i++ {
		for {
			v := *val
			if CompareAndSwapUintptr(val, v, v+1) {
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

	for _, tt := range hammer32 {
		if tt.f == nil {
			continue
		}
		c := make(chan int)
		var val uint32
		for i := 0; i < p; i++ {
			go func() {
				tt.f(&val, n)
				c <- 1
			}()
		}
		for i := 0; i < p; i++ {
			<-c
		}
		if val != uint32(n)*p {
			t.Errorf("%s: val=%d want %d", tt.name, val, n*p)
		}
	}
}

var hammer64 = []struct {
	name string
	f    func(*uint64, int)
}{
	{"AddInt64", hammerAddInt64},
	{"AddUint64", hammerAddUint64},
	{"AddUintptr", hammerAddUintptr64},
	{"CompareAndSwapInt64", hammerCompareAndSwapInt64},
	{"CompareAndSwapUint64", hammerCompareAndSwapUint64},
	{"CompareAndSwapUintptr", hammerCompareAndSwapUintptr64},
}

func init() {
	var v uint64 = 1 << 50
	if uintptr(v) == 0 {
		// 32-bit system; clear uintptr tests
		hammer64[2].f = nil
		hammer64[5].f = nil
	}
}

func hammerAddInt64(uval *uint64, count int) {
	val := (*int64)(unsafe.Pointer(uval))
	for i := 0; i < count; i++ {
		AddInt64(val, 1)
	}
}

func hammerAddUint64(val *uint64, count int) {
	for i := 0; i < count; i++ {
		AddUint64(val, 1)
	}
}

func hammerAddUintptr64(uval *uint64, count int) {
	// only safe when uintptr is 64-bit.
	// not called on 32-bit systems.
	val := (*uintptr)(unsafe.Pointer(uval))
	for i := 0; i < count; i++ {
		AddUintptr(val, 1)
	}
}

func hammerCompareAndSwapInt64(uval *uint64, count int) {
	val := (*int64)(unsafe.Pointer(uval))
	for i := 0; i < count; i++ {
		for {
			v := *val
			if CompareAndSwapInt64(val, v, v+1) {
				break
			}
		}
	}
}

func hammerCompareAndSwapUint64(val *uint64, count int) {
	for i := 0; i < count; i++ {
		for {
			v := *val
			if CompareAndSwapUint64(val, v, v+1) {
				break
			}
		}
	}
}

func hammerCompareAndSwapUintptr64(uval *uint64, count int) {
	// only safe when uintptr is 64-bit.
	// not called on 32-bit systems.
	val := (*uintptr)(unsafe.Pointer(uval))
	for i := 0; i < count; i++ {
		for {
			v := *val
			if CompareAndSwapUintptr(val, v, v+1) {
				break
			}
		}
	}
}

func TestHammer64(t *testing.T) {
	if test64err != nil {
		t.Logf("Skipping 64-bit tests: %v", test64err)
		return
	}
	const p = 4
	n := 100000
	if testing.Short() {
		n = 1000
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(p))

	for _, tt := range hammer64 {
		if tt.f == nil {
			continue
		}
		c := make(chan int)
		var val uint64
		for i := 0; i < p; i++ {
			go func() {
				tt.f(&val, n)
				c <- 1
			}()
		}
		for i := 0; i < p; i++ {
			<-c
		}
		if val != uint64(n)*p {
			t.Errorf("%s: val=%d want %d", tt.name, val, n*p)
		}
	}
}

func hammerLoadInt32(t *testing.T, uval *uint32) {
	val := (*int32)(unsafe.Pointer(uval))
	for {
		v := LoadInt32(val)
		vlo := v & ((1 << 16) - 1)
		vhi := v >> 16
		if vlo != vhi {
			t.Fatalf("LoadInt32: %#x != %#x", vlo, vhi)
		}
		new := v + 1 + 1<<16
		if vlo == 1e4 {
			new = 0
		}
		if CompareAndSwapInt32(val, v, new) {
			break
		}
	}
}

func hammerLoadUint32(t *testing.T, val *uint32) {
	for {
		v := LoadUint32(val)
		vlo := v & ((1 << 16) - 1)
		vhi := v >> 16
		if vlo != vhi {
			t.Fatalf("LoadUint32: %#x != %#x", vlo, vhi)
		}
		new := v + 1 + 1<<16
		if vlo == 1e4 {
			new = 0
		}
		if CompareAndSwapUint32(val, v, new) {
			break
		}
	}
}

func TestHammerLoad(t *testing.T) {
	tests := [...]func(*testing.T, *uint32){hammerLoadInt32, hammerLoadUint32}
	n := 100000
	if testing.Short() {
		n = 10000
	}
	const procs = 8
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(procs))
	for _, tt := range tests {
		c := make(chan int)
		var val uint32
		for p := 0; p < procs; p++ {
			go func() {
				for i := 0; i < n; i++ {
					tt(t, &val)
				}
				c <- 1
			}()
		}
		for p := 0; p < procs; p++ {
			<-c
		}
	}
}
