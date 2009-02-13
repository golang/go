// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv
import (
	"os";
	"fmt";
	"strconv";
	"testing"
)

type atoui64Test struct {
	in string;
	out uint64;
	err *os.Error;
}

var atoui64tests = []atoui64Test (
	atoui64Test( "", 0, os.EINVAL ),
	atoui64Test( "0", 0, nil ),
	atoui64Test( "1", 1, nil ),
	atoui64Test( "12345", 12345, nil ),
	atoui64Test( "012345", 0, os.EINVAL ),
	atoui64Test( "12345x", 0, os.EINVAL ),
	atoui64Test( "98765432100", 98765432100, nil ),
	atoui64Test( "18446744073709551615", 1<<64-1, nil ),
	atoui64Test( "18446744073709551616", 1<<64-1, os.ERANGE ),
	atoui64Test( "18446744073709551620", 1<<64-1, os.ERANGE ),
)

type atoi64Test struct {
	in string;
	out int64;
	err *os.Error;
}

var atoi64test = []atoi64Test (
	atoi64Test( "", 0, os.EINVAL ),
	atoi64Test( "0", 0, nil ),
	atoi64Test( "-0", 0, nil ),
	atoi64Test( "1", 1, nil ),
	atoi64Test( "-1", -1, nil ),
	atoi64Test( "12345", 12345, nil ),
	atoi64Test( "-12345", -12345, nil ),
	atoi64Test( "012345", 0, os.EINVAL ),
	atoi64Test( "-012345", 0, os.EINVAL ),
	atoi64Test( "12345x", 0, os.EINVAL ),
	atoi64Test( "-12345x", 0, os.EINVAL ),
	atoi64Test( "98765432100", 98765432100, nil ),
	atoi64Test( "-98765432100", -98765432100, nil ),
	atoi64Test( "9223372036854775807", 1<<63-1, nil ),
	atoi64Test( "-9223372036854775807", -(1<<63-1), nil ),
	atoi64Test( "9223372036854775808", 1<<63-1, os.ERANGE ),
	atoi64Test( "-9223372036854775808", -1<<63, nil ),
	atoi64Test( "9223372036854775809", 1<<63-1, os.ERANGE ),
	atoi64Test( "-9223372036854775809", -1<<63, os.ERANGE ),
)

type atoui32Test struct {
	in string;
	out uint32;
	err *os.Error;
}

var atoui32tests = []atoui32Test (
	atoui32Test( "", 0, os.EINVAL ),
	atoui32Test( "0", 0, nil ),
	atoui32Test( "1", 1, nil ),
	atoui32Test( "12345", 12345, nil ),
	atoui32Test( "012345", 0, os.EINVAL ),
	atoui32Test( "12345x", 0, os.EINVAL ),
	atoui32Test( "987654321", 987654321, nil ),
	atoui32Test( "4294967295", 1<<32-1, nil ),
	atoui32Test( "4294967296", 1<<32-1, os.ERANGE ),
)

type atoi32Test struct {
	in string;
	out int32;
	err *os.Error;
}

var atoi32tests = []atoi32Test (
	atoi32Test( "", 0, os.EINVAL ),
	atoi32Test( "0", 0, nil ),
	atoi32Test( "-0", 0, nil ),
	atoi32Test( "1", 1, nil ),
	atoi32Test( "-1", -1, nil ),
	atoi32Test( "12345", 12345, nil ),
	atoi32Test( "-12345", -12345, nil ),
	atoi32Test( "012345", 0, os.EINVAL ),
	atoi32Test( "-012345", 0, os.EINVAL ),
	atoi32Test( "12345x", 0, os.EINVAL ),
	atoi32Test( "-12345x", 0, os.EINVAL ),
	atoi32Test( "987654321", 987654321, nil ),
	atoi32Test( "-987654321", -987654321, nil ),
	atoi32Test( "2147483647", 1<<31-1, nil ),
	atoi32Test( "-2147483647", -(1<<31-1), nil ),
	atoi32Test( "2147483648", 1<<31-1, os.ERANGE ),
	atoi32Test( "-2147483648", -1<<31, nil ),
	atoi32Test( "2147483649", 1<<31-1, os.ERANGE ),
	atoi32Test( "-2147483649", -1<<31, os.ERANGE ),
)

func TestAtoui64(t *testing.T) {
	for i := 0; i < len(atoui64tests); i++ {
		test := &atoui64tests[i];
		out, err := strconv.Atoui64(test.in);
		if test.out != out || test.err != err {
			t.Errorf("strconv.Atoui64(%v) = %v, %v want %v, %v\n",
				test.in, out, err, test.out, test.err);
		}
	}
}

func TestAtoi64(t *testing.T) {
	for i := 0; i < len(atoi64test); i++ {
		test := &atoi64test[i];
		out, err := strconv.Atoi64(test.in);
		if test.out != out || test.err != err {
			t.Errorf("strconv.Atoi64(%v) = %v, %v want %v, %v\n",
				test.in, out, err, test.out, test.err);
		}
	}
}

func TestAtoui(t *testing.T) {
	switch intsize {
	case 32:
		for i := 0; i < len(atoui32tests); i++ {
			test := &atoui32tests[i];
			out, err := strconv.Atoui(test.in);
			if test.out != uint32(out) || test.err != err {
				t.Errorf("strconv.Atoui(%v) = %v, %v want %v, %v\n",
					test.in, out, err, test.out, test.err);
			}
		}
	case 64:
		for i := 0; i < len(atoui64tests); i++ {
			test := &atoui64tests[i];
			out, err := strconv.Atoui(test.in);
			if test.out != uint64(out) || test.err != err {
				t.Errorf("strconv.Atoui(%v) = %v, %v want %v, %v\n",
					test.in, out, err, test.out, test.err);
			}
		}
	}
}

func TestAtoi(t *testing.T) {
	switch intsize {
	case 32:
		for i := 0; i < len(atoi32tests); i++ {
			test := &atoi32tests[i];
			out, err := strconv.Atoi(test.in);
			if test.out != int32(out) || test.err != err {
				t.Errorf("strconv.Atoi(%v) = %v, %v want %v, %v\n",
					test.in, out, err, test.out, test.err);
			}
		}
	case 64:
		for i := 0; i < len(atoi64test); i++ {
			test := &atoi64test[i];
			out, err := strconv.Atoi(test.in);
			if test.out != int64(out) || test.err != err {
				t.Errorf("strconv.Atoi(%v) = %v, %v want %v, %v\n",
					test.in, out, err, test.out, test.err);
			}
		}
	}
}

