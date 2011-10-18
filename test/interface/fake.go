// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Interface comparisons using types hidden
// inside reflected-on structs.

package main

import "reflect"

type T struct {
	F float32
	G float32

	S string
	T string

	U uint32
	V uint32

	W uint32
	X uint32

	Y uint32
	Z uint32
}

func add(s, t string) string {
	return s + t
}

func assert(b bool) {
	if !b {
		panic("assert")
	}
}

func main() {
	var x T
	x.F = 1.0
	x.G = x.F
	x.S = add("abc", "def")
	x.T = add("abc", "def")
	x.U = 1
	x.V = 2
	x.W = 1 << 28
	x.X = 2 << 28
	x.Y = 0x12345678
	x.Z = x.Y

	// check mem and string
	v := reflect.ValueOf(x)
	i := v.Field(0)
	j := v.Field(1)
	assert(i.Interface() == j.Interface())

	s := v.Field(2)
	t := v.Field(3)
	assert(s.Interface() == t.Interface())

	// make sure different values are different.
	// make sure whole word is being compared,
	// not just a single byte.
	i = v.Field(4)
	j = v.Field(5)
	assert(i.Interface() != j.Interface())

	i = v.Field(6)
	j = v.Field(7)
	assert(i.Interface() != j.Interface())

	i = v.Field(8)
	j = v.Field(9)
	assert(i.Interface() == j.Interface())
}

/*
comparing uncomparable type float32
throw: interface compare

panic PC=0x28ceb8 [1]
throw+0x41 /Users/rsc/goX/src/runtime/runtime.c:54
	throw(0x3014a, 0x0)
ifaceeq+0x15c /Users/rsc/goX/src/runtime/iface.c:501
	ifaceeq(0x2aa7c0, 0x0, 0x0, 0x0, 0x2aa7c0, ...)
sys·ifaceeq+0x48 /Users/rsc/goX/src/runtime/iface.c:527
	sys·ifaceeq(0x2aa7c0, 0x0, 0x0, 0x0, 0x2aa7c0, ...)
main·main+0x190 /Users/rsc/goX/src/cmd/gc/x.go:10
	main·main()
mainstart+0xf /Users/rsc/goX/src/runtime/amd64/asm.s:53
	mainstart()
sys·Goexit /Users/rsc/goX/src/runtime/proc.c:124
	sys·Goexit()
*/
