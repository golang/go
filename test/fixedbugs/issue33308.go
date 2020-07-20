// errorcheck

// Copyright 2020 The Go Authors. All rights reserved.  Use of this
// source code is governed by a BSD-style license that can be found in
// the LICENSE file.

// Test that the compiler does not crash on a []byte conversion of an
// untyped expression.
package p

var v uint
var x = []byte((1 << v) + 1) // ERROR "cannot convert"
