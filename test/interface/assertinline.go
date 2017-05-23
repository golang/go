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

// TODO(rsc): struct{*int} is stored directly in the interface
// and should be possible to fetch back out of the interface,
// but more of the general data movement code needs to
// realize that before we can inline the assertion.

func assertstruct(x interface{}) struct{ *int } {
	return x.(struct{ *int }) // ERROR "type assertion not inlined"
}

func assertstruct2(x interface{}) (struct{ *int }, bool) {
	z, ok := x.(struct{ *int }) // ERROR "type assertion not inlined"
	return z, ok
}

func assertbig(x interface{}) complex128 {
	return x.(complex128) // ERROR "type assertion not inlined"
}

func assertbig2(x interface{}) (complex128, bool) {
	z, ok := x.(complex128) // ERROR "type assertion not inlined"
	return z, ok
}

func assertbig2ok(x interface{}) (complex128, bool) {
	_, ok := x.(complex128) // ERROR "type assertion [(]ok only[)] inlined"
	return 0, ok
}
