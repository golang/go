// errorcheck -e

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T struct{}

var s string
var b bool
var i int
var t T
var a [1]int

var (
	_ = s == nil // ERROR "invalid operation:.*mismatched types string and (untyped )?nil"
	_ = b == nil // ERROR "invalid operation:.*mismatched types bool and (untyped )?nil"
	_ = i == nil // ERROR "invalid operation:.*mismatched types int and (untyped )?nil"
	_ = t == nil // ERROR "invalid operation:.*mismatched types T and (untyped )?nil"
	_ = a == nil // ERROR "invalid operation:.*mismatched types \[1\]int and (untyped )?nil"
)
