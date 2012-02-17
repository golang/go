// run

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type I interface {
	m(a, b, c, d, e, f, g, h byte)
}

type Int8 int8

func (x Int8) m(a, b, c, d, e, f, g, h byte) {
	check("Int8", int64(x), 0x01, a, b, c, d, e, f, g, h)
}

type Uint8 uint8

func (x Uint8) m(a, b, c, d, e, f, g, h byte) {
	check("Uint8", int64(x), 0x01, a, b, c, d, e, f, g, h)
}

type Int16 int16

func (x Int16) m(a, b, c, d, e, f, g, h byte) {
	check("Int16", int64(x), 0x0102, a, b, c, d, e, f, g, h)
}

type Uint16 uint16

func (x Uint16) m(a, b, c, d, e, f, g, h byte) {
	check("Uint16", int64(x), 0x0102, a, b, c, d, e, f, g, h)
}

type Int32 int32

func (x Int32) m(a, b, c, d, e, f, g, h byte) {
	check("Int32", int64(x), 0x01020304, a, b, c, d, e, f, g, h)
}

type Uint32 uint32

func (x Uint32) m(a, b, c, d, e, f, g, h byte) {
	check("Uint32", int64(x), 0x01020304, a, b, c, d, e, f, g, h)
}

type Int64 int64

func (x Int64) m(a, b, c, d, e, f, g, h byte) {
	check("Int64", int64(x), 0x0102030405060708, a, b, c, d, e, f, g, h)
}

type Uint64 uint64

func (x Uint64) m(a, b, c, d, e, f, g, h byte) {
	check("Uint64", int64(x), 0x0102030405060708, a, b, c, d, e, f, g, h)
}

var test = []I{
	Int8(0x01),
	Uint8(0x01),
	Int16(0x0102),
	Uint16(0x0102),
	Int32(0x01020304),
	Uint32(0x01020304),
	Int64(0x0102030405060708),
	Uint64(0x0102030405060708),
}

func main() {
	for _, t := range test {
		t.m(0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17)
	}
}

var bug = false

func check(desc string, have, want int64, a, b, c, d, e, f, g, h byte) {
	if have != want || a != 0x10 || b != 0x11 || c != 0x12 || d != 0x13 || e != 0x14 || f != 0x15 || g != 0x16 || h != 0x17 {
		if !bug {
			bug = true
			println("BUG")
		}
		println(desc, "check", have, want, a, b, c, d, e, f, g, h)
	}
}
