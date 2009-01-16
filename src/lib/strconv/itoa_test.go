// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

import (
	"fmt";
	"os";
	"strconv";
	"testing";
)

type Itoa64Test struct {
	in int64;
	out string;
}

var itoa64tests = []Itoa64Test {
	Itoa64Test{ 0, "0" },
	Itoa64Test{ 1, "1" },
	Itoa64Test{ -1, "-1" },
	Itoa64Test{ 12345678, "12345678" },
	Itoa64Test{ -987654321, "-987654321" },
	Itoa64Test{ 1<<31-1, "2147483647" },
	Itoa64Test{ -1<<31+1, "-2147483647" },
	Itoa64Test{ 1<<31, "2147483648" },
	Itoa64Test{ -1<<31, "-2147483648" },
	Itoa64Test{ 1<<31+1, "2147483649" },
	Itoa64Test{ -1<<31-1, "-2147483649" },
	Itoa64Test{ 1<<32-1, "4294967295" },
	Itoa64Test{ -1<<32+1, "-4294967295" },
	Itoa64Test{ 1<<32, "4294967296" },
	Itoa64Test{ -1<<32, "-4294967296" },
	Itoa64Test{ 1<<32+1, "4294967297" },
	Itoa64Test{ -1<<32-1, "-4294967297" },
	Itoa64Test{ 1<<50, "1125899906842624" },
	Itoa64Test{ 1<<63-1, "9223372036854775807" },
	Itoa64Test{ -1<<63+1, "-9223372036854775807" },
	Itoa64Test{ -1<<63, "-9223372036854775808" },
}

export func TestItoa(t *testing.T) {
	for i := 0; i < len(itoa64tests); i++ {
		test := itoa64tests[i];
		s := strconv.itoa64(test.in);
		if s != test.out {
			t.Error("strconv.itoa64(%v) = %v want %v\n",
				test.in, s, test.out);
		}
		if int64(int(test.in)) == test.in {
			s := strconv.itoa(int(test.in));
			if s != test.out {
				t.Error("strconv.itoa(%v) = %v want %v\n",
					test.in, s, test.out);
			}
		}
	}
}

// TODO: Use once there is a strconv.uitoa
type Uitoa64Test struct {
	in uint64;
	out string;
}

// TODO: should be able to call this atoui64tests.
var uitoa64tests = []Uitoa64Test {
	Uitoa64Test{ 1<<63-1, "9223372036854775807" },
	Uitoa64Test{ 1<<63, "9223372036854775808" },
	Uitoa64Test{ 1<<63+1, "9223372036854775809" },
	Uitoa64Test{ 1<<64-2, "18446744073709551614" },
	Uitoa64Test{ 1<<64-1, "18446744073709551615" },
}

