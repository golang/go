// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T0 T0 // ERROR "invalid recursive type"
type _ map[T0]int

type T1 struct{ T1 } // ERROR "invalid recursive type"
type _ map[T1]int

func f() {
	type T2 T2 // ERROR "invalid recursive type"
	type _ map[T2]int
}

func g() {
	type T3 struct{ T3 } // ERROR "invalid recursive type"
	type _ map[T3]int
}

func h() {
	type T4 struct{ m map[T4]int } // ERROR "invalid map key"
	type _ map[T4]int              // GC_ERROR "invalid map key"
}
