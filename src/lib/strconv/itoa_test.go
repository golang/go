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

type itoa64Test struct {
	in int64;
	out string;
}

var itoa64tests = []itoa64Test (
	itoa64Test( 0, "0" ),
	itoa64Test( 1, "1" ),
	itoa64Test( -1, "-1" ),
	itoa64Test( 12345678, "12345678" ),
	itoa64Test( -987654321, "-987654321" ),
	itoa64Test( 1<<31-1, "2147483647" ),
	itoa64Test( -1<<31+1, "-2147483647" ),
	itoa64Test( 1<<31, "2147483648" ),
	itoa64Test( -1<<31, "-2147483648" ),
	itoa64Test( 1<<31+1, "2147483649" ),
	itoa64Test( -1<<31-1, "-2147483649" ),
	itoa64Test( 1<<32-1, "4294967295" ),
	itoa64Test( -1<<32+1, "-4294967295" ),
	itoa64Test( 1<<32, "4294967296" ),
	itoa64Test( -1<<32, "-4294967296" ),
	itoa64Test( 1<<32+1, "4294967297" ),
	itoa64Test( -1<<32-1, "-4294967297" ),
	itoa64Test( 1<<50, "1125899906842624" ),
	itoa64Test( 1<<63-1, "9223372036854775807" ),
	itoa64Test( -1<<63+1, "-9223372036854775807" ),
	itoa64Test( -1<<63, "-9223372036854775808" ),
)

func TestItoa(t *testing.T) {
	for i := 0; i < len(itoa64tests); i++ {
		test := itoa64tests[i];
		s := strconv.Itoa64(test.in);
		if s != test.out {
			t.Error("strconv.Itoa64(%v) = %v want %v\n",
				test.in, s, test.out);
		}
		if int64(int(test.in)) == test.in {
			s := strconv.Itoa(int(test.in));
			if s != test.out {
				t.Error("strconv.Itoa(%v) = %v want %v\n",
					test.in, s, test.out);
			}
		}
	}
}

// TODO: Use once there is a strconv.uitoa
type uitoa64Test struct {
	in uint64;
	out string;
}

// TODO: should be able to call this atoui64tests.
var uitoa64tests = []uitoa64Test (
	uitoa64Test( 1<<63-1, "9223372036854775807" ),
	uitoa64Test( 1<<63, "9223372036854775808" ),
	uitoa64Test( 1<<63+1, "9223372036854775809" ),
	uitoa64Test( 1<<64-2, "18446744073709551614" ),
	uitoa64Test( 1<<64-1, "18446744073709551615" ),
)

