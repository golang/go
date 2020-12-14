// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that follow-on errors due to conflicting
// struct field and method names are suppressed.

package p

type T struct {
	a, b, c int
	E
}

type E struct{}

func (T) b()  {} // ERROR "field and method named b|redeclares struct field name"
func (*T) E() {} // ERROR "field and method named E|redeclares struct field name"

func _() {
	var x T
	_ = x.a
	_ = x.b // no follow-on error here
	x.b()   // no follow-on error here
	_ = x.c
	_ = x.E // no follow-on error here
	x.E()   // no follow-on error here
}
