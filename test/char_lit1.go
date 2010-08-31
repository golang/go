// errchk $G -e $F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const (
	// check that surrogate pair elements are invalid
	// (d800-dbff, dc00-dfff).
	_ = '\ud7ff' // ok
	_ = '\ud800'  // ERROR "Unicode|unicode"
	_ = "\U0000D999"  // ERROR "Unicode|unicode"
	_ = '\udc01' // ERROR "Unicode|unicode"
	_ = '\U0000dddd'  // ERROR "Unicode|unicode"
	_ = '\udfff' // ERROR "Unicode|unicode"
	_ = '\ue000' // ok
	_ = '\U0010ffff'  // ok
	_ = '\U00110000'  // ERROR "Unicode|unicode"
	_ = "abc\U0010ffffdef"  // ok
	_ = "abc\U00110000def"  // ERROR "Unicode|unicode"
	_ = '\Uffffffff'  // ERROR "Unicode|unicode"
)

