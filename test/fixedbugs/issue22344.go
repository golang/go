// compile

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test iota inside a function in a ConstSpec is accepted
package main

import (
	"unsafe"
)

// iotas are usable inside closures in constant declarations (#22345)
const (
	_ = iota
	_ = len([iota]byte{})
	_ = unsafe.Sizeof(iota)
	_ = unsafe.Sizeof(func() { _ = iota })
	_ = unsafe.Sizeof(func() { var _ = iota })
	_ = unsafe.Sizeof(func() { const _ = iota })
	_ = unsafe.Sizeof(func() { type _ [iota]byte })
	_ = unsafe.Sizeof(func() { func() int { return iota }() })
)

// verify inner and outer const declarations have distinct iotas
const (
	zero = iota
	one  = iota
	_    = unsafe.Sizeof(func() {
		var x [iota]int // [2]int
		var y [iota]int // [2]int
		const (
			Zero = iota
			One
			Two
			_ = unsafe.Sizeof([iota - 1]int{} == x) // assert types are equal
			_ = unsafe.Sizeof([iota - 2]int{} == y) // assert types are equal
			_ = unsafe.Sizeof([Two]int{} == x)      // assert types are equal
		)
		var z [iota]int                  // [2]int
		_ = unsafe.Sizeof([2]int{} == z) // assert types are equal
	})
	three = iota // the sequence continues
)

var _ [three]int = [3]int{} // assert 'three' has correct value

func main() {

	const (
		_ = iota
		_ = len([iota]byte{})
		_ = unsafe.Sizeof(iota)
		_ = unsafe.Sizeof(func() { _ = iota })
		_ = unsafe.Sizeof(func() { var _ = iota })
		_ = unsafe.Sizeof(func() { const _ = iota })
		_ = unsafe.Sizeof(func() { type _ [iota]byte })
		_ = unsafe.Sizeof(func() { func() int { return iota }() })
	)

	const (
		zero = iota
		one  = iota
		_    = unsafe.Sizeof(func() {
			var x [iota]int // [2]int
			var y [iota]int // [2]int
			const (
				Zero = iota
				One
				Two
				_ = unsafe.Sizeof([iota - 1]int{} == x) // assert types are equal
				_ = unsafe.Sizeof([iota - 2]int{} == y) // assert types are equal
				_ = unsafe.Sizeof([Two]int{} == x)      // assert types are equal
			)
			var z [iota]int                  // [2]int
			_ = unsafe.Sizeof([2]int{} == z) // assert types are equal
		})
		three = iota // the sequence continues
	)

	var _ [three]int = [3]int{} // assert 'three' has correct value
}
