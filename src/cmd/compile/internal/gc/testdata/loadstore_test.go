// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests load/store ordering

package main

import "testing"

// testLoadStoreOrder tests for reordering of stores/loads.
func testLoadStoreOrder(t *testing.T) {
	z := uint32(1000)
	if testLoadStoreOrder_ssa(&z, 100) == 0 {
		t.Errorf("testLoadStoreOrder failed")
	}
}

//go:noinline
func testLoadStoreOrder_ssa(z *uint32, prec uint) int {
	old := *z         // load
	*z = uint32(prec) // store
	if *z < old {     // load
		return 1
	}
	return 0
}

func testStoreSize(t *testing.T) {
	a := [4]uint16{11, 22, 33, 44}
	testStoreSize_ssa(&a[0], &a[2], 77)
	want := [4]uint16{77, 22, 33, 44}
	if a != want {
		t.Errorf("testStoreSize failed.  want = %d, got = %d", want, a)
	}
}

//go:noinline
func testStoreSize_ssa(p *uint16, q *uint16, v uint32) {
	// Test to make sure that (Store ptr (Trunc32to16 val) mem)
	// does not end up as a 32-bit store. It must stay a 16 bit store
	// even when Trunc32to16 is rewritten to be a nop.
	// To ensure that we get rewrite the Trunc32to16 before
	// we rewrite the Store, we force the truncate into an
	// earlier basic block by using it on both branches.
	w := uint16(v)
	if p != nil {
		*p = w
	} else {
		*q = w
	}
}

//go:noinline
func testExtStore_ssa(p *byte, b bool) int {
	x := *p
	*p = 7
	if b {
		return int(x)
	}
	return 0
}

func testExtStore(t *testing.T) {
	const start = 8
	var b byte = start
	if got := testExtStore_ssa(&b, true); got != start {
		t.Errorf("testExtStore failed.  want = %d, got = %d", start, got)
	}
}

var b int

// testDeadStorePanic_ssa ensures that we don't optimize away stores
// that could be read by after recover().  Modeled after fixedbugs/issue1304.
//go:noinline
func testDeadStorePanic_ssa(a int) (r int) {
	defer func() {
		recover()
		r = a
	}()
	a = 2      // store
	b := a - a // optimized to zero
	c := 4
	a = c / b // store, but panics
	a = 3     // store
	r = a
	return
}

func testDeadStorePanic(t *testing.T) {
	if want, got := 2, testDeadStorePanic_ssa(1); want != got {
		t.Errorf("testDeadStorePanic failed.  want = %d, got = %d", want, got)
	}
}

//go:noinline
func loadHitStore8(x int8, p *int8) int32 {
	x *= x           // try to trash high bits (arch-dependent)
	*p = x           // store
	return int32(*p) // load and cast
}

//go:noinline
func loadHitStoreU8(x uint8, p *uint8) uint32 {
	x *= x            // try to trash high bits (arch-dependent)
	*p = x            // store
	return uint32(*p) // load and cast
}

//go:noinline
func loadHitStore16(x int16, p *int16) int32 {
	x *= x           // try to trash high bits (arch-dependent)
	*p = x           // store
	return int32(*p) // load and cast
}

//go:noinline
func loadHitStoreU16(x uint16, p *uint16) uint32 {
	x *= x            // try to trash high bits (arch-dependent)
	*p = x            // store
	return uint32(*p) // load and cast
}

//go:noinline
func loadHitStore32(x int32, p *int32) int64 {
	x *= x           // try to trash high bits (arch-dependent)
	*p = x           // store
	return int64(*p) // load and cast
}

//go:noinline
func loadHitStoreU32(x uint32, p *uint32) uint64 {
	x *= x            // try to trash high bits (arch-dependent)
	*p = x            // store
	return uint64(*p) // load and cast
}

func testLoadHitStore(t *testing.T) {
	// Test that sign/zero extensions are kept when a load-hit-store
	// is replaced by a register-register move.
	{
		var in int8 = (1 << 6) + 1
		var p int8
		got := loadHitStore8(in, &p)
		want := int32(in * in)
		if got != want {
			t.Errorf("testLoadHitStore (int8) failed. want = %d, got = %d", want, got)
		}
	}
	{
		var in uint8 = (1 << 6) + 1
		var p uint8
		got := loadHitStoreU8(in, &p)
		want := uint32(in * in)
		if got != want {
			t.Errorf("testLoadHitStore (uint8) failed. want = %d, got = %d", want, got)
		}
	}
	{
		var in int16 = (1 << 10) + 1
		var p int16
		got := loadHitStore16(in, &p)
		want := int32(in * in)
		if got != want {
			t.Errorf("testLoadHitStore (int16) failed. want = %d, got = %d", want, got)
		}
	}
	{
		var in uint16 = (1 << 10) + 1
		var p uint16
		got := loadHitStoreU16(in, &p)
		want := uint32(in * in)
		if got != want {
			t.Errorf("testLoadHitStore (uint16) failed. want = %d, got = %d", want, got)
		}
	}
	{
		var in int32 = (1 << 30) + 1
		var p int32
		got := loadHitStore32(in, &p)
		want := int64(in * in)
		if got != want {
			t.Errorf("testLoadHitStore (int32) failed. want = %d, got = %d", want, got)
		}
	}
	{
		var in uint32 = (1 << 30) + 1
		var p uint32
		got := loadHitStoreU32(in, &p)
		want := uint64(in * in)
		if got != want {
			t.Errorf("testLoadHitStore (uint32) failed. want = %d, got = %d", want, got)
		}
	}
}

func TestLoadStore(t *testing.T) {
	testLoadStoreOrder(t)
	testStoreSize(t)
	testExtStore(t)
	testDeadStorePanic(t)
	testLoadHitStore(t)
}
