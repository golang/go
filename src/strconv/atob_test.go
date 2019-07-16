// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	"bytes"
	. "strconv"
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
		if test.err != nil {
			// expect an error
			if e == nil {
				t.Errorf("%s: expected %s but got nil", test.in, test.err)
			} else {
				// NumError assertion must succeed; it's the only thing we return.
				if test.err != e.(*NumError).Err {
					t.Errorf("%s: expected %s but got %s", test.in, test.err, e)
				}
			}
		} else {
			if e != nil {
				t.Errorf("%s: expected no error but got %s", test.in, e)
			}
			if b != test.out {
				t.Errorf("%s: expected %t but got %t", test.in, test.out, b)
			}
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
			t.Errorf(`FormatBool(%v): expected %q but got %q`, b, s, f)
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
			t.Errorf("AppendBool(%q, %v): expected %q but got %q", test.in, test.b, test.out, b)
		}
	}
}
