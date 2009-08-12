// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unicode

import "testing"

var upper = []int{
	0x41,
	0xc0,
	0xd8,
	0x100,
	0x139,
	0x14a,
	0x178,
	0x181,
	0x376,
	0x3cf,
	0x1f2a,
	0x2102,
	0x2c00,
	0x2c10,
	0x2c20,
	0xa650,
	0xa722,
	0xff3a,
	0x10400,
	0x1d400,
	0x1d7ca,
}

var notupper = []int{
	0x40,
	0x5b,
	0x61,
	0x185,
	0x1b0,
	0x377,
	0x387,
	0x2150,
	0xffff,
	0x10000,
}

var letter = []int{
	0x41,
	0x61,
	0xaa,
	0xba,
	0xc8,
	0xdb,
	0xf9,
	0x2ec,
	0x535,
	0x6e6,
	0x93d,
	0xa15,
	0xb99,
	0xdc0,
	0xedd,
	0x1000,
	0x1200,
	0x1312,
	0x1401,
	0x1885,
	0x2c00,
	0xa800,
	0xf900,
	0xfa30,
	0xffda,
	0xffdc,
	0x10000,
	0x10300,
	0x10400,
	0x20000,
	0x2f800,
	0x2fa1d,
}

var notletter = []int{
	0x20,
	0x35,
	0x375,
	0x620,
	0x700,
	0xfffe,
	0x1ffff,
	0x10ffff,
}

func TestIsLetter(t *testing.T) {
	for i, r := range upper {
		if !IsLetter(r) {
			t.Errorf("IsLetter(%#x) = false, want true\n", r);
		}
	}
	for i, r := range letter {
		if !IsLetter(r) {
			t.Errorf("IsLetter(%#x) = false, want true\n", r);
		}
	}
	for i, r := range notletter {
		if IsLetter(r) {
			t.Errorf("IsLetter(%#x) = true, want false\n", r);
		}
	}
}

func TestIsUpper(t *testing.T) {
	for i, r := range upper {
		if !IsUpper(r) {
			t.Errorf("IsUpper(%#x) = false, want true\n", r);
		}
	}
	for i, r := range notupper {
		if IsUpper(r) {
			t.Errorf("IsUpper(%#x) = true, want false\n", r);
		}
	}
	for i, r := range notletter {
		if IsUpper(r) {
			t.Errorf("IsUpper(%#x) = true, want false\n", r);
		}
	}
}
