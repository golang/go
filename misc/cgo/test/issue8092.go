// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 8092. Test that linker defined symbols (e.g., text, data) don't
// conflict with C symbols.

package cgotest

/*
char text[] = "text";
char data[] = "data";
char *ctext(void) { return text; }
char *cdata(void) { return data; }
*/
import "C"

import "testing"

func test8092(t *testing.T) {
	tests := []struct {
		s    string
		a, b *C.char
	}{
		{"text", &C.text[0], C.ctext()},
		{"data", &C.data[0], C.cdata()},
	}
	for _, test := range tests {
		if test.a != test.b {
			t.Errorf("%s: pointer mismatch: %v != %v", test.s, test.a, test.b)
		}
		if got := C.GoString(test.a); got != test.s {
			t.Errorf("%s: points at %#v, want %#v", test.s, got, test.s)
		}
	}
}
