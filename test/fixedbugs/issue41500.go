// errorcheck

// Copyright 2020 The Go Authors. All rights reserved.  Use of this
// source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package p

type s struct {
	slice []int
}

func f() {
	var x *s

	_ = x == nil || len(x.slice) // ERROR "invalid operation: .+ \(operator \|\| not defined on int\)|incompatible types|mismatched types untyped bool and int"
	_ = len(x.slice) || x == nil // ERROR "invalid operation: .+ \(operator \|\| not defined on int\)|incompatible types|mismatched types int and untyped bool"
	_ = x == nil && len(x.slice) // ERROR "invalid operation: .+ \(operator && not defined on int\)|incompatible types|mismatched types untyped bool and int"
	_ = len(x.slice) && x == nil // ERROR "invalid operation: .+ \(operator && not defined on int\)|incompatible types|mismatched types int and untyped bool"
}
