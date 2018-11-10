// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests load/store ordering

package main

import "fmt"

// testLoadStoreOrder tests for reordering of stores/loads.
func testLoadStoreOrder() {
	z := uint32(1000)
	if testLoadStoreOrder_ssa(&z, 100) == 0 {
		println("testLoadStoreOrder failed")
		failed = true
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

func testStoreSize() {
	a := [4]uint16{11, 22, 33, 44}
	testStoreSize_ssa(&a[0], &a[2], 77)
	want := [4]uint16{77, 22, 33, 44}
	if a != want {
		fmt.Println("testStoreSize failed.  want =", want, ", got =", a)
		failed = true
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

var failed = false

//go:noinline
func testExtStore_ssa(p *byte, b bool) int {
	x := *p
	*p = 7
	if b {
		return int(x)
	}
	return 0
}

func testExtStore() {
	const start = 8
	var b byte = start
	if got := testExtStore_ssa(&b, true); got != start {
		fmt.Println("testExtStore failed.  want =", start, ", got =", got)
		failed = true
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

func testDeadStorePanic() {
	if want, got := 2, testDeadStorePanic_ssa(1); want != got {
		fmt.Println("testDeadStorePanic failed.  want =", want, ", got =", got)
		failed = true
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

func testLoadHitStore() {
	// Test that sign/zero extensions are kept when a load-hit-store
	// is replaced by a register-register move.
	{
		var in int8 = (1 << 6) + 1
		var p int8
		got := loadHitStore8(in, &p)
		want := int32(in * in)
		if got != want {
			fmt.Println("testLoadHitStore (int8) failed. want =", want, ", got =", got)
			failed = true
		}
	}
	{
		var in uint8 = (1 << 6) + 1
		var p uint8
		got := loadHitStoreU8(in, &p)
		want := uint32(in * in)
		if got != want {
			fmt.Println("testLoadHitStore (uint8) failed. want =", want, ", got =", got)
			failed = true
		}
	}
	{
		var in int16 = (1 << 10) + 1
		var p int16
		got := loadHitStore16(in, &p)
		want := int32(in * in)
		if got != want {
			fmt.Println("testLoadHitStore (int16) failed. want =", want, ", got =", got)
			failed = true
		}
	}
	{
		var in uint16 = (1 << 10) + 1
		var p uint16
		got := loadHitStoreU16(in, &p)
		want := uint32(in * in)
		if got != want {
			fmt.Println("testLoadHitStore (uint16) failed. want =", want, ", got =", got)
			failed = true
		}
	}
	{
		var in int32 = (1 << 30) + 1
		var p int32
		got := loadHitStore32(in, &p)
		want := int64(in * in)
		if got != want {
			fmt.Println("testLoadHitStore (int32) failed. want =", want, ", got =", got)
			failed = true
		}
	}
	{
		var in uint32 = (1 << 30) + 1
		var p uint32
		got := loadHitStoreU32(in, &p)
		want := uint64(in * in)
		if got != want {
			fmt.Println("testLoadHitStore (uint32) failed. want =", want, ", got =", got)
			failed = true
		}
	}
}

func main() {

	testLoadStoreOrder()
	testStoreSize()
	testExtStore()
	testDeadStorePanic()
	testLoadHitStore()

	if failed {
		panic("failed")
	}
}
