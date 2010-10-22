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
	{[]byte{}, []byte{}},
	{[]byte{0x01}, []byte{'0', '1'}},
	{[]byte{0xff}, []byte{'f', 'f'}},
	{[]byte{0xff, 00}, []byte{'f', 'f', '0', '0'}},
	{[]byte{0}, []byte{'0', '0'}},
	{[]byte{1}, []byte{'0', '1'}},
	{[]byte{2}, []byte{'0', '2'}},
	{[]byte{3}, []byte{'0', '3'}},
	{[]byte{4}, []byte{'0', '4'}},
	{[]byte{5}, []byte{'0', '5'}},
	{[]byte{6}, []byte{'0', '6'}},
	{[]byte{7}, []byte{'0', '7'}},
	{[]byte{8}, []byte{'0', '8'}},
	{[]byte{9}, []byte{'0', '9'}},
	{[]byte{10}, []byte{'0', 'a'}},
	{[]byte{11}, []byte{'0', 'b'}},
	{[]byte{12}, []byte{'0', 'c'}},
	{[]byte{13}, []byte{'0', 'd'}},
	{[]byte{14}, []byte{'0', 'e'}},
	{[]byte{15}, []byte{'0', 'f'}},
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
	{[]byte{}, []byte{}, true},
	{[]byte{'0'}, []byte{}, false},
	{[]byte{'0', 'g'}, []byte{}, false},
	{[]byte{'0', '\x01'}, []byte{}, false},
	{[]byte{'0', '0'}, []byte{0}, true},
	{[]byte{'0', '1'}, []byte{1}, true},
	{[]byte{'0', '2'}, []byte{2}, true},
	{[]byte{'0', '3'}, []byte{3}, true},
	{[]byte{'0', '4'}, []byte{4}, true},
	{[]byte{'0', '5'}, []byte{5}, true},
	{[]byte{'0', '6'}, []byte{6}, true},
	{[]byte{'0', '7'}, []byte{7}, true},
	{[]byte{'0', '8'}, []byte{8}, true},
	{[]byte{'0', '9'}, []byte{9}, true},
	{[]byte{'0', 'a'}, []byte{10}, true},
	{[]byte{'0', 'b'}, []byte{11}, true},
	{[]byte{'0', 'c'}, []byte{12}, true},
	{[]byte{'0', 'd'}, []byte{13}, true},
	{[]byte{'0', 'e'}, []byte{14}, true},
	{[]byte{'0', 'f'}, []byte{15}, true},
	{[]byte{'0', 'A'}, []byte{10}, true},
	{[]byte{'0', 'B'}, []byte{11}, true},
	{[]byte{'0', 'C'}, []byte{12}, true},
	{[]byte{'0', 'D'}, []byte{13}, true},
	{[]byte{'0', 'E'}, []byte{14}, true},
	{[]byte{'0', 'F'}, []byte{15}, true},
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
	{[]byte{}, ""},
	{[]byte{0}, "00"},
	{[]byte{0, 1}, "0001"},
	{[]byte{0, 1, 255}, "0001ff"},
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
	{"", []byte{}, true},
	{"0", []byte{}, false},
	{"00", []byte{0}, true},
	{"0\x01", []byte{}, false},
	{"0g", []byte{}, false},
	{"00ff00", []byte{0, 255, 0}, true},
	{"0000ff", []byte{0, 0, 255}, true},
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
