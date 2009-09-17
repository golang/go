// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	. "strconv";
	"testing";
)

type shiftTest struct {
	i uint64;
	shift int;
	out string;
}

var shifttests = []shiftTest {
	shiftTest{ 0, -100, "0" },
	shiftTest{ 0, 100, "0" },
	shiftTest{ 1, 100, "1267650600228229401496703205376" },
	shiftTest{ 1, -100,
		"0.00000000000000000000000000000078886090522101180541"
		"17285652827862296732064351090230047702789306640625" },
	shiftTest{ 12345678, 8, "3160493568" },
	shiftTest{ 12345678, -8, "48225.3046875" },
	shiftTest{ 195312, 9, "99999744" },
	shiftTest{ 1953125, 9, "1000000000" },
}

func TestDecimalShift(t *testing.T) {
	for i := 0; i < len(shifttests); i++ {
		test := &shifttests[i];
		s := NewDecimal(test.i).Shift(test.shift).String();
		if s != test.out {
			t.Errorf("Decimal %v << %v = %v, want %v\n",
				test.i, test.shift, s, test.out);
		}
	}
}

type roundTest struct {
	i uint64;
	nd int;
	down, round, up string;
	int uint64;
}

var roundtests = []roundTest {
	roundTest{ 0, 4, "0", "0", "0", 0 },
	roundTest{ 12344999, 4, "12340000", "12340000", "12350000", 12340000 },
	roundTest{ 12345000, 4, "12340000", "12340000", "12350000", 12340000 },
	roundTest{ 12345001, 4, "12340000", "12350000", "12350000", 12350000 },
	roundTest{ 23454999, 4, "23450000", "23450000", "23460000", 23450000 },
	roundTest{ 23455000, 4, "23450000", "23460000", "23460000", 23460000 },
	roundTest{ 23455001, 4, "23450000", "23460000", "23460000", 23460000 },

	roundTest{ 99994999, 4, "99990000", "99990000", "100000000", 99990000 },
	roundTest{ 99995000, 4, "99990000", "100000000", "100000000", 100000000 },
	roundTest{ 99999999, 4, "99990000", "100000000", "100000000", 100000000 },

	roundTest{ 12994999, 4, "12990000", "12990000", "13000000", 12990000 },
	roundTest{ 12995000, 4, "12990000", "13000000", "13000000", 13000000 },
	roundTest{ 12999999, 4, "12990000", "13000000", "13000000", 13000000 },
}

func TestDecimalRound(t *testing.T) {
	for i := 0; i < len(roundtests); i++ {
		test := &roundtests[i];
		s := NewDecimal(test.i).RoundDown(test.nd).String();
		if s != test.down {
			t.Errorf("Decimal %v RoundDown %d = %v, want %v\n",
				test.i, test.nd, s, test.down);
		}
		s = NewDecimal(test.i).Round(test.nd).String();
		if s != test.round {
			t.Errorf("Decimal %v Round %d = %v, want %v\n",
				test.i, test.nd, s, test.down);
		}
		s = NewDecimal(test.i).RoundUp(test.nd).String();
		if s != test.up {
			t.Errorf("Decimal %v RoundUp %d = %v, want %v\n",
				test.i, test.nd, s, test.up);
		}
	}
}

type roundIntTest struct {
	i uint64;
	shift int;
	int uint64;
}

var roundinttests = []roundIntTest {
	roundIntTest{ 0, 100, 0 },
	roundIntTest{ 512, -8, 2 },
	roundIntTest{ 513, -8, 2 },
	roundIntTest{ 640, -8, 2 },
	roundIntTest{ 641, -8, 3 },
	roundIntTest{ 384, -8, 2 },
	roundIntTest{ 385, -8, 2 },
	roundIntTest{ 383, -8, 1 },
	roundIntTest{ 1, 100, 1<<64-1 },
	roundIntTest{ 1000, 0, 1000 },
}

func TestDecimalRoundedInteger(t *testing.T) {
	for i := 0; i < len(roundinttests); i++ {
		test := roundinttests[i];
		int := NewDecimal(test.i).Shift(test.shift).RoundedInteger();
		if int != test.int {
			t.Errorf("Decimal %v >> %v RoundedInteger = %v, want %v\n",
				test.i, test.shift, int, test.int);
		}
	}
}
