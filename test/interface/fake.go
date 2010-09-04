// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Interface comparisons using types hidden
// inside reflected-on structs.

package main

import "reflect"

type T struct {
	f float32
	g float32

	s string
	t string

	u uint32
	v uint32

	w uint32
	x uint32

	y uint32
	z uint32
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
	x.f = 1.0
	x.g = x.f
	x.s = add("abc", "def")
	x.t = add("abc", "def")
	x.u = 1
	x.v = 2
	x.w = 1<<28
	x.x = 2<<28
	x.y = 0x12345678
	x.z = x.y

	// check mem and string
	v := reflect.NewValue(x)
	i := v.(*reflect.StructValue).Field(0)
	j := v.(*reflect.StructValue).Field(1)
	assert(i.Interface() == j.Interface())

	s := v.(*reflect.StructValue).Field(2)
	t := v.(*reflect.StructValue).Field(3)
	assert(s.Interface() == t.Interface())

	// make sure different values are different.
	// make sure whole word is being compared,
	// not just a single byte.
	i = v.(*reflect.StructValue).Field(4)
	j = v.(*reflect.StructValue).Field(5)
	assert(i.Interface() != j.Interface())

	i = v.(*reflect.StructValue).Field(6)
	j = v.(*reflect.StructValue).Field(7)
	assert(i.Interface() != j.Interface())

	i = v.(*reflect.StructValue).Field(8)
	j = v.(*reflect.StructValue).Field(9)
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
