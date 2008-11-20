// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

import (
	"fmt";
	"strconv";
	"testing";
)

type ShiftTest struct {
	i uint64;
	shift int;
	out string;
}

var shifttests = []ShiftTest {
	ShiftTest{ 0, -100, "0" },
	ShiftTest{ 0, 100, "0" },
	ShiftTest{ 1, 100, "1267650600228229401496703205376" },
	ShiftTest{ 1, -100,
		"0.00000000000000000000000000000078886090522101180541"
		"17285652827862296732064351090230047702789306640625" },
	ShiftTest{ 12345678, 8, "3160493568" },
	ShiftTest{ 12345678, -8, "48225.3046875" },
	ShiftTest{ 195312, 9, "99999744" },
	ShiftTest{ 1953125, 9, "1000000000" },
}

export func TestDecimalShift(t *testing.T) {
	ok := true;
	for i := 0; i < len(shifttests); i++ {
		test := &shifttests[i];
		s := strconv.NewDecimal(test.i).Shift(test.shift).String();
		if s != test.out {
			t.Errorf("Decimal %v << %v = %v, want %v\n",
				test.i, test.shift, s, test.out);
		}
	}
}

type RoundTest struct {
	i uint64;
	nd int;
	down, round, up string;
	int uint64;
}

var roundtests = []RoundTest {
	RoundTest{ 0, 4, "0", "0", "0", 0 },
	RoundTest{ 12344999, 4, "12340000", "12340000", "12350000", 12340000 },
	RoundTest{ 12345000, 4, "12340000", "12340000", "12350000", 12340000 },
	RoundTest{ 12345001, 4, "12340000", "12350000", "12350000", 12350000 },
	RoundTest{ 23454999, 4, "23450000", "23450000", "23460000", 23450000 },
	RoundTest{ 23455000, 4, "23450000", "23460000", "23460000", 23460000 },
	RoundTest{ 23455001, 4, "23450000", "23460000", "23460000", 23460000 },

	RoundTest{ 99994999, 4, "99990000", "99990000", "100000000", 99990000 },
	RoundTest{ 99995000, 4, "99990000", "100000000", "100000000", 100000000 },
	RoundTest{ 99999999, 4, "99990000", "100000000", "100000000", 100000000 },

	RoundTest{ 12994999, 4, "12990000", "12990000", "13000000", 12990000 },
	RoundTest{ 12995000, 4, "12990000", "13000000", "13000000", 13000000 },
	RoundTest{ 12999999, 4, "12990000", "13000000", "13000000", 13000000 },
}

export func TestDecimalRound(t *testing.T) {
	for i := 0; i < len(roundtests); i++ {
		test := &roundtests[i];
		s := strconv.NewDecimal(test.i).RoundDown(test.nd).String();
		if s != test.down {
			t.Errorf("Decimal %v RoundDown %d = %v, want %v\n",
				test.i, test.nd, s, test.down);
		}
		s = strconv.NewDecimal(test.i).Round(test.nd).String();
		if s != test.round {
			t.Errorf("Decimal %v Round %d = %v, want %v\n",
				test.i, test.nd, s, test.down);
		}
		s = strconv.NewDecimal(test.i).RoundUp(test.nd).String();
		if s != test.up {
			t.Errorf("Decimal %v RoundUp %d = %v, want %v\n",
				test.i, test.nd, s, test.up);
		}
	}
}

type RoundIntTest struct {
	i uint64;
	shift int;
	int uint64;
}

var roundinttests = []RoundIntTest {
	RoundIntTest{ 0, 100, 0 },
	RoundIntTest{ 512, -8, 2 },
	RoundIntTest{ 513, -8, 2 },
	RoundIntTest{ 640, -8, 2 },
	RoundIntTest{ 641, -8, 3 },
	RoundIntTest{ 384, -8, 2 },
	RoundIntTest{ 385, -8, 2 },
	RoundIntTest{ 383, -8, 1 },
	RoundIntTest{ 1, 100, 1<<64-1 },
	RoundIntTest{ 1000, 0, 1000 },
}

export func TestDecimalRoundedInteger(t *testing.T) {
	for i := 0; i < len(roundinttests); i++ {
		test := roundinttests[i];
		// TODO: should be able to use int := here.
		int1 := strconv.NewDecimal(test.i).Shift(test.shift).RoundedInteger();
		if int1 != test.int {
			t.Errorf("Decimal %v >> %v RoundedInteger = %v, want %v\n",
				test.i, test.shift, int1, test.int);
		}
	}
}
