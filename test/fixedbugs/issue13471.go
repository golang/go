// errorcheck

// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests for golang.org/issue/13471

package main

func main() {
	const _ int64 = 1e646456992 // ERROR "1e\+646456992 overflows integer"
	const _ int32 = 1e64645699  // ERROR "1e\+64645699 overflows integer"
	const _ int16 = 1e6464569   // ERROR "1e\+6464569 overflows integer"
	const _ int8 = 1e646456     // ERROR "1e\+646456 overflows integer"
	const _ int = 1e64645       // ERROR "1e\+64645 overflows integer"

	const _ uint64 = 1e646456992 // ERROR "1e\+646456992 overflows integer"
	const _ uint32 = 1e64645699  // ERROR "1e\+64645699 overflows integer"
	const _ uint16 = 1e6464569   // ERROR "1e\+6464569 overflows integer"
	const _ uint8 = 1e646456     // ERROR "1e\+646456 overflows integer"
	const _ uint = 1e64645       // ERROR "1e\+64645 overflows integer"

	const _ rune = 1e64645 // ERROR "1e\+64645 overflows integer"
}
