// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

// Issue 21809.  Compile C `typedef` to go type aliases.

// typedef long MySigned_t;
// /* tests alias-to-alias */
// typedef MySigned_t MySigned2_t;
//
// long takes_long(long x) { return x * x; }
// MySigned_t takes_typedef(MySigned_t x) { return x * x; }
import "C"

import "testing"

func test21809(t *testing.T) {
	longVar := C.long(3)
	typedefVar := C.MySigned_t(4)
	typedefTypedefVar := C.MySigned2_t(5)

	// all three should be considered identical to `long`
	if ret := C.takes_long(longVar); ret != 9 {
		t.Errorf("got %v but expected %v", ret, 9)
	}
	if ret := C.takes_long(typedefVar); ret != 16 {
		t.Errorf("got %v but expected %v", ret, 16)
	}
	if ret := C.takes_long(typedefTypedefVar); ret != 25 {
		t.Errorf("got %v but expected %v", ret, 25)
	}

	// They should also be identical to the typedef'd type
	if ret := C.takes_typedef(longVar); ret != 9 {
		t.Errorf("got %v but expected %v", ret, 9)
	}
	if ret := C.takes_typedef(typedefVar); ret != 16 {
		t.Errorf("got %v but expected %v", ret, 16)
	}
	if ret := C.takes_typedef(typedefTypedefVar); ret != 25 {
		t.Errorf("got %v but expected %v", ret, 25)
	}
}
