// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"unsafe"
)

// these 4 functions are complicated enough that we will share
// the print logic with the C printf.
var (
	printstring_m,
	printuint_m,
	printhex_m,
	printfloat_m mFunction
)

func printstring(s string) {
	mp := acquirem()
	mp.scalararg[0] = uintptr(len(s))
	mp.ptrarg[0] = (*stringStruct)(unsafe.Pointer(&s)).str
	onM(&printstring_m)
	releasem(mp)
}

func printuint(x uint64) {
	mp := acquirem()
	*(*uint64)(unsafe.Pointer(&mp.scalararg[0])) = x
	onM(&printuint_m)
	releasem(mp)
}

func printhex(x uintptr) {
	mp := acquirem()
	mp.scalararg[0] = uintptr(x)
	onM(&printhex_m)
	releasem(mp)
}

func printfloat(x float64) {
	mp := acquirem()
	*(*float64)(unsafe.Pointer(&mp.scalararg[0])) = x
	onM(&printfloat_m)
	releasem(mp)
}

// all other print functions are expressible as combinations
// of the above 4 functions.
func printnl() {
	printstring("\n")
}

func printsp() {
	printstring(" ")
}

func printbool(b bool) {
	if b {
		printstring("true")
	} else {
		printstring("false")
	}
}

func printpointer(p unsafe.Pointer) {
	printhex(uintptr(p))
}

func printint(x int64) {
	if x < 0 {
		printstring("-")
		x = -x
	}
	printuint(uint64(x))
}

func printcomplex(x complex128) {
	printstring("(")
	printfloat(real(x))
	printfloat(imag(x))
	printstring("i)")
}

func printiface(i interface {
	f()
}) {
	printstring("(")
	printhex((*[2]uintptr)(unsafe.Pointer(&i))[0])
	printstring(",")
	printhex((*[2]uintptr)(unsafe.Pointer(&i))[1])
	printstring(")")
}

func printeface(e interface{}) {
	printstring("(")
	printhex((*[2]uintptr)(unsafe.Pointer(&e))[0])
	printstring(",")
	printhex((*[2]uintptr)(unsafe.Pointer(&e))[1])
	printstring(")")
}

func printslice(b []byte) {
	printstring("[")
	printint(int64(len(b)))
	printstring("/")
	printint(int64(cap(b)))
	printstring("]")
	printhex((*[3]uintptr)(unsafe.Pointer(&b))[0])
}
