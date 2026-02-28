// errorcheck -lang=go1.12

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// numeric literals
const (
	_ = 1_000 // ERROR "underscore in numeric literal requires go1.13 or later \(-lang was set to go1.12; check go.mod\)|requires go1.13"
	_ = 0b111 // ERROR "binary literal requires go1.13 or later"
	_ = 0o567 // ERROR "0o/0O-style octal literal requires go1.13 or later"
	_ = 0xabc // ok
	_ = 0x0p1 // ERROR "hexadecimal floating-point literal requires go1.13 or later"

	_ = 0b111 // ERROR "binary"
	_ = 0o567 // ERROR "octal"
	_ = 0xabc // ok
	_ = 0x0p1 // ERROR "hexadecimal floating-point"

	_ = 1_000i // ERROR "underscore"
	_ = 0b111i // ERROR "binary"
	_ = 0o567i // ERROR "octal"
	_ = 0xabci // ERROR "hexadecimal floating-point"
	_ = 0x0p1i // ERROR "hexadecimal floating-point"
)

// signed shift counts
var (
	s int
	_ = 1 << s // ERROR "invalid operation: 1 << s \(signed shift count type int\) requires go1.13 or later|signed shift count"
	_ = 1 >> s // ERROR "signed shift count"
)
