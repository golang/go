// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 21048: s390x merged address generation into stores
// to unaligned global variables. This resulted in an illegal
// instruction.

package main

type T struct {
	_ [1]byte
	a [2]byte // offset: 1
	_ [3]byte
	b [2]uint16 // offset: 6
	_ [2]byte
	c [2]uint32 // offset: 12
	_ [2]byte
	d [2]int16 // offset: 22
	_ [2]byte
	e [2]int32 // offset: 28
}

var Source, Sink T

func newT() T {
	return T{
		a: [2]byte{1, 2},
		b: [2]uint16{1, 2},
		c: [2]uint32{1, 2},
		d: [2]int16{1, 2},
		e: [2]int32{1, 2},
	}
}

//go:noinline
func moves() {
	Sink.a = Source.a
	Sink.b = Source.b
	Sink.c = Source.c
	Sink.d = Source.d
	Sink.e = Source.e
}

//go:noinline
func loads() *T {
	t := newT()
	t.a = Source.a
	t.b = Source.b
	t.c = Source.c
	t.d = Source.d
	t.e = Source.e
	return &t
}

//go:noinline
func stores() {
	t := newT()
	Sink.a = t.a
	Sink.b = t.b
	Sink.c = t.c
	Sink.d = t.d
	Sink.e = t.e
}

func main() {
	moves()
	loads()
	stores()
}
