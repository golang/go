// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that the implementation catches nil ptr indirection
// in a large address space.

// +build aix

package main

import "unsafe"

// Having a big address space means that indexing
// at a 1G + 256 MB offset from a nil pointer might not
// cause a memory access fault. This test checks
// that Go is doing the correct explicit checks to catch
// these nil pointer accesses, not just relying on the hardware.
// The reason of the 1G offset is because AIX addresses start after 1G.
var dummy [256 << 20]byte // give us a big address space

func main() {
	// the test only tests what we intend to test
	// if dummy starts in the first 256 MB of memory.
	// otherwise there might not be anything mapped
	// at the address that might be accidentally
	// dereferenced below.
	if uintptr(unsafe.Pointer(&dummy)) < 1<<32 {
		panic("dummy not far enough")
	}

	shouldPanic(p1)
	shouldPanic(p2)
	shouldPanic(p3)
	shouldPanic(p4)
	shouldPanic(p5)
	shouldPanic(p6)
	shouldPanic(p7)
	shouldPanic(p8)
	shouldPanic(p9)
	shouldPanic(p10)
	shouldPanic(p11)
	shouldPanic(p12)
	shouldPanic(p13)
	shouldPanic(p14)
	shouldPanic(p15)
	shouldPanic(p16)
}

func shouldPanic(f func()) {
	defer func() {
		if recover() == nil {
			panic("memory reference did not panic")
		}
	}()
	f()
}

func p1() {
	// Array index.
	var p *[1 << 33]byte = nil
	println(p[1<<32+256<<20]) // very likely to be inside dummy, but should panic
}

var xb byte

func p2() {
	var p *[1 << 33]byte = nil
	xb = 123

	// Array index.
	println(p[uintptr(unsafe.Pointer(&xb))]) // should panic
}

func p3() {
	// Array to slice.
	var p *[1 << 33]byte = nil
	var x []byte = p[0:] // should panic
	_ = x
}

var q *[1 << 33]byte

func p4() {
	// Array to slice.
	var x []byte
	var y = &x
	*y = q[0:] // should crash (uses arraytoslice runtime routine)
}

func fb([]byte) {
	panic("unreachable")
}

func p5() {
	// Array to slice.
	var p *[1 << 33]byte = nil
	fb(p[0:]) // should crash
}

func p6() {
	// Array to slice.
	var p *[1 << 33]byte = nil
	var _ []byte = p[10 : len(p)-10] // should crash
}

type T struct {
	x [1<<32 + 256<<20]byte
	i int
}

func f() *T {
	return nil
}

var y *T
var x = &y

func p7() {
	// Struct field access with large offset.
	println(f().i) // should crash
}

func p8() {
	// Struct field access with large offset.
	println((*x).i) // should crash
}

func p9() {
	// Struct field access with large offset.
	var t *T
	println(&t.i) // should crash
}

func p10() {
	// Struct field access with large offset.
	var t *T
	println(t.i) // should crash
}

type T1 struct {
	T
}

type T2 struct {
	*T1
}

func p11() {
	t := &T2{}
	p := &t.i
	println(*p)
}

// ADDR(DOT(IND(p))) needs a check also
func p12() {
	var p *T = nil
	println(*(&((*p).i)))
}

// Tests suggested in golang.org/issue/6080.

func p13() {
	var x *[10]int
	y := x[:]
	_ = y
}

func p14() {
	println((*[1]int)(nil)[:])
}

func p15() {
	for i := range (*[1]int)(nil)[:] {
		_ = i
	}
}

func p16() {
	for i, v := range (*[1]int)(nil)[:] {
		_ = i + v
	}
}
