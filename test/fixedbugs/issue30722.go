// errorcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that we only get one error per invalid integer literal.

package p

const (
	_ = 1_       // ERROR "'_' must separate successive digits"
	_ = 0b       // ERROR "binary literal has no digits|invalid numeric literal"
	_ = 0o       // ERROR "octal literal has no digits|invalid numeric literal"
	_ = 0x       // ERROR "hexadecimal literal has no digits|invalid numeric literal"
	_ = 0xde__ad // ERROR "'_' must separate successive digits"
)
