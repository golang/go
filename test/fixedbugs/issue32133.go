// errorcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// errors for the //line-adjusted code below
// ERROR "newline in string"
// ERROR "newline in character literal|newline in rune literal"
// ERROR "newline in string"
// ERROR "string not terminated"

//line :10:1
import "foo

//line :19:1
func _() {
	0x // ERROR "hexadecimal literal has no digits"
}

func _() {
	0x1.0 // ERROR "hexadecimal mantissa requires a 'p' exponent"
}

func _() {
	0_i // ERROR "'_' must separate successive digits"
}

func _() {
//line :11:1
	'
}

func _() {
//line :12:1
	"
}

func _() {
//line :13:1
	`