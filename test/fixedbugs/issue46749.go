// errorcheck

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var s string
var b bool
var i int
var iface interface{}

var (
	_ = "" + b   // ERROR "invalid operation.*mismatched types.*untyped string and bool"
	_ = "" + i   // ERROR "invalid operation.*mismatched types.*untyped string and int"
	_ = "" + nil // ERROR "invalid operation.*mismatched types.*untyped string and nil|(untyped nil)"
)

var (
	_ = s + false // ERROR "invalid operation.*mismatched types.*string and untyped bool"
	_ = s + 1     // ERROR "invalid operation.*mismatched types.*string and untyped int"
	_ = s + nil   // ERROR "invalid operation.*mismatched types.*string and nil|(untyped nil)"
)

var (
	_ = "" + false // ERROR "invalid operation.*mismatched types.*untyped string and untyped bool"
	_ = "" + 1     // ERROR "invalid operation.*mismatched types.*untyped string and untyped int"
)

var (
	_ = b + 1         // ERROR "invalid operation.*mismatched types.*bool and untyped int"
	_ = i + false     // ERROR "invalid operation.*mismatched types.*int and untyped bool"
	_ = iface + 1     // ERROR "invalid operation.*mismatched types.*interface *{} and int"
	_ = iface + 1.0   // ERROR "invalid operation.*mismatched types.*interface *{} and float64"
	_ = iface + false // ERROR "invalid operation.*mismatched types.*interface *{} and bool"
)
