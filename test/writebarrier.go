// errorcheck -0 -l -d=wb

// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test where write barriers are and are not emitted.

package p

import "unsafe"

func f(x **byte, y *byte) {
	*x = y // ERROR "write barrier"

	z := y // no barrier
	*x = z // ERROR "write barrier"
}

func f1(x *[]byte, y []byte) {
	*x = y // ERROR "write barrier"

	z := y // no barrier
	*x = z // ERROR "write barrier"
}

func f1a(x *[]byte, y *[]byte) {
	*x = *y // ERROR "write barrier"

	z := *y // no barrier
	*x = z // ERROR "write barrier"
}

func f2(x *interface{}, y interface{}) {
	*x = y // ERROR "write barrier"

	z := y // no barrier
	*x = z // ERROR "write barrier"
}

func f2a(x *interface{}, y *interface{}) {
	*x = *y // ERROR "write barrier"

	z := y // no barrier
	*x = z // ERROR "write barrier"
}

func f3(x *string, y string) {
	*x = y // ERROR "write barrier"

	z := y // no barrier
	*x = z // ERROR "write barrier"
}

func f3a(x *string, y *string) {
	*x = *y // ERROR "write barrier"

	z := *y // no barrier
	*x = z // ERROR "write barrier"
}

func f4(x *[2]string, y [2]string) {
	*x = y // ERROR "write barrier"

	z := y // no barrier
	*x = z // ERROR "write barrier"
}

func f4a(x *[2]string, y *[2]string) {
	*x = *y // ERROR "write barrier"

	z := *y // no barrier
	*x = z // ERROR "write barrier"
}

type T struct {
	X *int
	Y int
	M map[int]int
}

func f5(t, u *T) {
	t.X = &u.Y // ERROR "write barrier"
}

func f6(t *T) {
	t.M = map[int]int{1: 2} // ERROR "write barrier"
}

func f7(x, y *int) []*int {
	var z [3]*int
	i := 0
	z[i] = x // ERROR "write barrier"
	i++
	z[i] = y // ERROR "write barrier"
	i++
	return z[:i]
}

func f9(x *interface{}, v *byte) {
	*x = v // ERROR "write barrier"
}

func f10(x *byte, f func(interface{})) {
	f(x)
}

func f11(x *unsafe.Pointer, y unsafe.Pointer) {
	*x = unsafe.Pointer(uintptr(y) + 1) // ERROR "write barrier"
}
