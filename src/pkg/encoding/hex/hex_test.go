// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hex

import (
	"bytes"
	"testing"
)

type encodeTest struct {
	in, out []byte
}

var encodeTests = []encodeTest{
	encodeTest{[]byte{}, []byte{}},
	encodeTest{[]byte{0x01}, []byte{'0', '1'}},
	encodeTest{[]byte{0xff}, []byte{'f', 'f'}},
	encodeTest{[]byte{0xff, 00}, []byte{'f', 'f', '0', '0'}},
	encodeTest{[]byte{0}, []byte{'0', '0'}},
	encodeTest{[]byte{1}, []byte{'0', '1'}},
	encodeTest{[]byte{2}, []byte{'0', '2'}},
	encodeTest{[]byte{3}, []byte{'0', '3'}},
	encodeTest{[]byte{4}, []byte{'0', '4'}},
	encodeTest{[]byte{5}, []byte{'0', '5'}},
	encodeTest{[]byte{6}, []byte{'0', '6'}},
	encodeTest{[]byte{7}, []byte{'0', '7'}},
	encodeTest{[]byte{8}, []byte{'0', '8'}},
	encodeTest{[]byte{9}, []byte{'0', '9'}},
	encodeTest{[]byte{10}, []byte{'0', 'a'}},
	encodeTest{[]byte{11}, []byte{'0', 'b'}},
	encodeTest{[]byte{12}, []byte{'0', 'c'}},
	encodeTest{[]byte{13}, []byte{'0', 'd'}},
	encodeTest{[]byte{14}, []byte{'0', 'e'}},
	encodeTest{[]byte{15}, []byte{'0', 'f'}},
}

func TestEncode(t *testing.T) {
	for i, test := range encodeTests {
		dst := make([]byte, EncodedLen(len(test.in)))
		n := Encode(dst, test.in)
		if n != len(dst) {
			t.Errorf("#%d: bad return value: got: %d want: %d", i, n, len(dst))
		}
		if bytes.Compare(dst, test.out) != 0 {
			t.Errorf("#%d: got: %#v want: %#v", i, dst, test.out)
		}
	}
}

type decodeTest struct {
	in, out []byte
	ok      bool
}

var decodeTests = []decodeTest{
	decodeTest{[]byte{}, []byte{}, true},
	decodeTest{[]byte{'0'}, []byte{}, false},
	decodeTest{[]byte{'0', 'g'}, []byte{}, false},
	decodeTest{[]byte{'0', '0'}, []byte{0}, true},
	decodeTest{[]byte{'0', '1'}, []byte{1}, true},
	decodeTest{[]byte{'0', '2'}, []byte{2}, true},
	decodeTest{[]byte{'0', '3'}, []byte{3}, true},
	decodeTest{[]byte{'0', '4'}, []byte{4}, true},
	decodeTest{[]byte{'0', '5'}, []byte{5}, true},
	decodeTest{[]byte{'0', '6'}, []byte{6}, true},
	decodeTest{[]byte{'0', '7'}, []byte{7}, true},
	decodeTest{[]byte{'0', '8'}, []byte{8}, true},
	decodeTest{[]byte{'0', '9'}, []byte{9}, true},
	decodeTest{[]byte{'0', 'a'}, []byte{10}, true},
	decodeTest{[]byte{'0', 'b'}, []byte{11}, true},
	decodeTest{[]byte{'0', 'c'}, []byte{12}, true},
	decodeTest{[]byte{'0', 'd'}, []byte{13}, true},
	decodeTest{[]byte{'0', 'e'}, []byte{14}, true},
	decodeTest{[]byte{'0', 'f'}, []byte{15}, true},
	decodeTest{[]byte{'0', 'A'}, []byte{10}, true},
	decodeTest{[]byte{'0', 'B'}, []byte{11}, true},
	decodeTest{[]byte{'0', 'C'}, []byte{12}, true},
	decodeTest{[]byte{'0', 'D'}, []byte{13}, true},
	decodeTest{[]byte{'0', 'E'}, []byte{14}, true},
	decodeTest{[]byte{'0', 'F'}, []byte{15}, true},
}

func TestDecode(t *testing.T) {
	for i, test := range decodeTests {
		dst := make([]byte, DecodedLen(len(test.in)))
		n, err := Decode(dst, test.in)
		if err == nil && n != len(dst) {
			t.Errorf("#%d: bad return value: got:%d want:%d", i, n, len(dst))
		}
		if test.ok != (err == nil) {
			t.Errorf("#%d: unexpected err value: %s", i, err)
		}
		if err == nil && bytes.Compare(dst, test.out) != 0 {
			t.Errorf("#%d: got: %#v want: %#v", i, dst, test.out)
		}
	}
}

type encodeStringTest struct {
	in  []byte
	out string
}

var encodeStringTests = []encodeStringTest{
	encodeStringTest{[]byte{}, ""},
	encodeStringTest{[]byte{0}, "00"},
	encodeStringTest{[]byte{0, 1}, "0001"},
	encodeStringTest{[]byte{0, 1, 255}, "0001ff"},
}

func TestEncodeToString(t *testing.T) {
	for i, test := range encodeStringTests {
		s := EncodeToString(test.in)
		if s != test.out {
			t.Errorf("#%d got:%s want:%s", i, s, test.out)
		}
	}
}

type decodeStringTest struct {
	in  string
	out []byte
	ok  bool
}

var decodeStringTests = []decodeStringTest{
	decodeStringTest{"", []byte{}, true},
	decodeStringTest{"0", []byte{}, false},
	decodeStringTest{"00", []byte{0}, true},
	decodeStringTest{"0g", []byte{}, false},
	decodeStringTest{"00ff00", []byte{0, 255, 0}, true},
	decodeStringTest{"0000ff", []byte{0, 0, 255}, true},
}

func TestDecodeString(t *testing.T) {
	for i, test := range decodeStringTests {
		dst, err := DecodeString(test.in)
		if test.ok != (err == nil) {
			t.Errorf("#%d: unexpected err value: %s", i, err)
		}
		if err == nil && bytes.Compare(dst, test.out) != 0 {
			t.Errorf("#%d: got: %#v want: #%v", i, dst, test.out)
		}
	}
}
