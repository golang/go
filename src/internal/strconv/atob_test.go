// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	"bytes"
	. "internal/strconv"
	"testing"
)

type atobTest struct {
	in  string
	out bool
	err error
}

var atobtests = []atobTest{
	{"", false, ErrSyntax},
	{"asdf", false, ErrSyntax},
	{"0", false, nil},
	{"f", false, nil},
	{"F", false, nil},
	{"FALSE", false, nil},
	{"false", false, nil},
	{"False", false, nil},
	{"1", true, nil},
	{"t", true, nil},
	{"T", true, nil},
	{"TRUE", true, nil},
	{"true", true, nil},
	{"True", true, nil},
}

func TestParseBool(t *testing.T) {
	for _, test := range atobtests {
		b, e := ParseBool(test.in)
		if b != test.out || e != test.err {
			t.Errorf("ParseBool(%s) = %v, %v, want %v, %v", test.in, b, e, test.out, test.err)
		}
	}
}

var boolString = map[bool]string{
	true:  "true",
	false: "false",
}

func TestFormatBool(t *testing.T) {
	for b, s := range boolString {
		if f := FormatBool(b); f != s {
			t.Errorf("FormatBool(%v) = %q; want %q", b, f, s)
		}
	}
}

type appendBoolTest struct {
	b   bool
	in  []byte
	out []byte
}

var appendBoolTests = []appendBoolTest{
	{true, []byte("foo "), []byte("foo true")},
	{false, []byte("foo "), []byte("foo false")},
}

func TestAppendBool(t *testing.T) {
	for _, test := range appendBoolTests {
		b := AppendBool(test.in, test.b)
		if !bytes.Equal(b, test.out) {
			t.Errorf("AppendBool(%q, %v) = %q; want %q", test.in, test.b, b, test.out)
		}
	}
}
