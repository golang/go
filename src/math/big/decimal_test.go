// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import "testing"

func TestDecimalString(t *testing.T) {
	for _, test := range []struct {
		x    decimal
		want string
	}{
		{want: "0"},
		{decimal{nil, 1000}, "0"}, // exponent of 0 is ignored
		{decimal{[]byte("12345"), 0}, "0.12345"},
		{decimal{[]byte("12345"), -3}, "0.00012345"},
		{decimal{[]byte("12345"), +3}, "123.45"},
		{decimal{[]byte("12345"), +10}, "1234500000"},
	} {
		if got := test.x.String(); got != test.want {
			t.Errorf("%v == %s; want %s", test.x, got, test.want)
		}
	}
}

func TestDecimalInit(t *testing.T) {
	for _, test := range []struct {
		x     Word
		shift int
		want  string
	}{
		{0, 0, "0"},
		{0, -100, "0"},
		{0, 100, "0"},
		{1, 0, "1"},
		{1, 10, "1024"},
		{1, 100, "1267650600228229401496703205376"},
		{1, -100, "0.0000000000000000000000000000007888609052210118054117285652827862296732064351090230047702789306640625"},
		{12345678, 8, "3160493568"},
		{12345678, -8, "48225.3046875"},
		{195312, 9, "99999744"},
		{1953125, 9, "1000000000"},
	} {
		var d decimal
		d.init(nat{test.x}.norm(), test.shift)
		if got := d.String(); got != test.want {
			t.Errorf("%d << %d == %s; want %s", test.x, test.shift, got, test.want)
		}
	}
}
