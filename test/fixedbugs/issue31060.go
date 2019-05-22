// errorcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

const (
	f = 1.0
	c = 1.0i

	_ = ^f // ERROR "invalid operation|expected integer"
	_ = ^c // ERROR "invalid operation|expected integer"

	_ = f % f // ERROR "invalid operation|expected integer"
	_ = c % c // ERROR "invalid operation|expected integer"

	_ = f & f // ERROR "invalid operation|expected integer"
	_ = c & c // ERROR "invalid operation|expected integer"

	_ = f | f // ERROR "invalid operation|expected integer"
	_ = c | c // ERROR "invalid operation|expected integer"

	_ = f ^ f // ERROR "invalid operation|expected integer"
	_ = c ^ c // ERROR "invalid operation|expected integer"

	_ = f &^ f // ERROR "invalid operation|expected integer"
	_ = c &^ c // ERROR "invalid operation|expected integer"
)
