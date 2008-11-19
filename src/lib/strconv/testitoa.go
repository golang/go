// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

import (
	"fmt";
	"os";
	"strconv";
)

type Int64Test struct {
	in int64;
	out string;
}

// TODO: should be called int64tests

var xint64tests = []Int64Test {
	Int64Test{ 0, "0" },
	Int64Test{ 1, "1" },
	Int64Test{ -1, "-1" },
	Int64Test{ 12345678, "12345678" },
	Int64Test{ -987654321, "-987654321" },
	Int64Test{ 1<<31-1, "2147483647" },
	Int64Test{ -1<<31+1, "-2147483647" },
	Int64Test{ 1<<31, "2147483648" },
	Int64Test{ -1<<31, "-2147483648" },
	Int64Test{ 1<<31+1, "2147483649" },
	Int64Test{ -1<<31-1, "-2147483649" },
	Int64Test{ 1<<32-1, "4294967295" },
	Int64Test{ -1<<32+1, "-4294967295" },
	Int64Test{ 1<<32, "4294967296" },
	Int64Test{ -1<<32, "-4294967296" },
	Int64Test{ 1<<32+1, "4294967297" },
	Int64Test{ -1<<32-1, "-4294967297" },
	Int64Test{ 1<<50, "1125899906842624" },
	Int64Test{ 1<<63-1, "9223372036854775807" },
	Int64Test{ -1<<63+1, "-9223372036854775807" },
	Int64Test{ -1<<63, "-9223372036854775808" },
}

export func TestItoa() bool {
	ok := true;
	for i := 0; i < len(xint64tests); i++ {
		t := xint64tests[i];
		s := strconv.itoa64(t.in);
		if s != t.out {
			fmt.printf("strconv.itoa64(%v) = %v want %v\n",
				t.in, s, t.out);
			ok = false;
		}
		if int64(int(t.in)) == t.in {
			s := strconv.itoa(int(t.in));
			if s != t.out {
				fmt.printf("strconv.itoa(%v) = %v want %v\n",
					t.in, s, t.out);
				ok = false;
			}
		}
	}
	return ok;
}

// TODO: Use once there is a strconv.uitoa
type Uint64Test struct {
	in uint64;
	out string;
}

// TODO: should be able to call this uint64tests.
var xuint64tests = []Uint64Test {
	Uint64Test{ 1<<63-1, "9223372036854775807" },
	Uint64Test{ 1<<63, "9223372036854775808" },
	Uint64Test{ 1<<63+1, "9223372036854775809" },
	Uint64Test{ 1<<64-2, "18446744073709551614" },
	Uint64Test{ 1<<64-1, "18446744073709551615" },
}

