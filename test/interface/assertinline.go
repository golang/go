// errorcheck -0 -d=typeassert

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func assertptr(x interface{}) *int {
	return x.(*int) // ERROR "type assertion inlined"
}

func assertptr2(x interface{}) (*int, bool) {
	z, ok := x.(*int) // ERROR "type assertion inlined"
	return z, ok
}

func assertfunc(x interface{}) func() {
	return x.(func()) // ERROR "type assertion inlined"
}

func assertfunc2(x interface{}) (func(), bool) {
	z, ok := x.(func()) // ERROR "type assertion inlined"
	return z, ok
}

func assertstruct(x interface{}) struct{ *int } {
	return x.(struct{ *int }) // ERROR "type assertion inlined"
}

func assertstruct2(x interface{}) (struct{ *int }, bool) {
	z, ok := x.(struct{ *int }) // ERROR "type assertion inlined"
	return z, ok
}

func assertbig(x interface{}) complex128 {
	return x.(complex128) // ERROR "type assertion inlined"
}

func assertbig2(x interface{}) (complex128, bool) {
	z, ok := x.(complex128) // ERROR "type assertion inlined"
	return z, ok
}

func assertbig2ok(x interface{}) (complex128, bool) {
	_, ok := x.(complex128) // ERROR "type assertion inlined"
	return 0, ok
}

func assertslice(x interface{}) []int {
	return x.([]int) // ERROR "type assertion inlined"
}

func assertslice2(x interface{}) ([]int, bool) {
	z, ok := x.([]int) // ERROR "type assertion inlined"
	return z, ok
}

func assertslice2ok(x interface{}) ([]int, bool) {
	_, ok := x.([]int) // ERROR "type assertion inlined"
	return nil, ok
}

type I interface {
	foo()
}

func assertInter(x interface{}) I {
	return x.(I) // ERROR "type assertion not inlined"
}
func assertInter2(x interface{}) (I, bool) {
	z, ok := x.(I) // ERROR "type assertion not inlined"
	return z, ok
}
