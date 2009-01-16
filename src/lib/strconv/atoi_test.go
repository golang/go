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

type Atoui64Test struct {
	in string;
	out uint64;
	err *os.Error;
}

var atoui64tests = []Atoui64Test {
	Atoui64Test{ "", 0, os.EINVAL },
	Atoui64Test{ "0", 0, nil },
	Atoui64Test{ "1", 1, nil },
	Atoui64Test{ "12345", 12345, nil },
	Atoui64Test{ "012345", 0, os.EINVAL },
	Atoui64Test{ "12345x", 0, os.EINVAL },
	Atoui64Test{ "98765432100", 98765432100, nil },
	Atoui64Test{ "18446744073709551615", 1<<64-1, nil },
	Atoui64Test{ "18446744073709551616", 1<<64-1, os.ERANGE },
	Atoui64Test{ "18446744073709551620", 1<<64-1, os.ERANGE },
}

type Atoi64Test struct {
	in string;
	out int64;
	err *os.Error;
}

var atoi64test = []Atoi64Test {
	Atoi64Test{ "", 0, os.EINVAL },
	Atoi64Test{ "0", 0, nil },
	Atoi64Test{ "-0", 0, nil },
	Atoi64Test{ "1", 1, nil },
	Atoi64Test{ "-1", -1, nil },
	Atoi64Test{ "12345", 12345, nil },
	Atoi64Test{ "-12345", -12345, nil },
	Atoi64Test{ "012345", 0, os.EINVAL },
	Atoi64Test{ "-012345", 0, os.EINVAL },
	Atoi64Test{ "12345x", 0, os.EINVAL },
	Atoi64Test{ "-12345x", 0, os.EINVAL },
	Atoi64Test{ "98765432100", 98765432100, nil },
	Atoi64Test{ "-98765432100", -98765432100, nil },
	Atoi64Test{ "9223372036854775807", 1<<63-1, nil },
	Atoi64Test{ "-9223372036854775807", -(1<<63-1), nil },
	Atoi64Test{ "9223372036854775808", 1<<63-1, os.ERANGE },
	Atoi64Test{ "-9223372036854775808", -1<<63, nil },
	Atoi64Test{ "9223372036854775809", 1<<63-1, os.ERANGE },
	Atoi64Test{ "-9223372036854775809", -1<<63, os.ERANGE },
}

type Atoui32Test struct {
	in string;
	out uint32;
	err *os.Error;
}

var atoui32tests = []Atoui32Test {
	Atoui32Test{ "", 0, os.EINVAL },
	Atoui32Test{ "0", 0, nil },
	Atoui32Test{ "1", 1, nil },
	Atoui32Test{ "12345", 12345, nil },
	Atoui32Test{ "012345", 0, os.EINVAL },
	Atoui32Test{ "12345x", 0, os.EINVAL },
	Atoui32Test{ "987654321", 987654321, nil },
	Atoui32Test{ "4294967295", 1<<32-1, nil },
	Atoui32Test{ "4294967296", 1<<32-1, os.ERANGE },
}

type Atoi32Test struct {
	in string;
	out int32;
	err *os.Error;
}

var atoi32tests = []Atoi32Test {
	Atoi32Test{ "", 0, os.EINVAL },
	Atoi32Test{ "0", 0, nil },
	Atoi32Test{ "-0", 0, nil },
	Atoi32Test{ "1", 1, nil },
	Atoi32Test{ "-1", -1, nil },
	Atoi32Test{ "12345", 12345, nil },
	Atoi32Test{ "-12345", -12345, nil },
	Atoi32Test{ "012345", 0, os.EINVAL },
	Atoi32Test{ "-012345", 0, os.EINVAL },
	Atoi32Test{ "12345x", 0, os.EINVAL },
	Atoi32Test{ "-12345x", 0, os.EINVAL },
	Atoi32Test{ "987654321", 987654321, nil },
	Atoi32Test{ "-987654321", -987654321, nil },
	Atoi32Test{ "2147483647", 1<<31-1, nil },
	Atoi32Test{ "-2147483647", -(1<<31-1), nil },
	Atoi32Test{ "2147483648", 1<<31-1, os.ERANGE },
	Atoi32Test{ "-2147483648", -1<<31, nil },
	Atoi32Test{ "2147483649", 1<<31-1, os.ERANGE },
	Atoi32Test{ "-2147483649", -1<<31, os.ERANGE },
}

export func TestAtoui64(t *testing.T) {
	for i := 0; i < len(atoui64tests); i++ {
		test := &atoui64tests[i];
		out, err := strconv.atoui64(test.in);
		if test.out != out || test.err != err {
			t.Errorf("strconv.atoui64(%v) = %v, %v want %v, %v\n",
				test.in, out, err, test.out, test.err);
		}
	}
}

export func TestAtoi64(t *testing.T) {
	for i := 0; i < len(atoi64test); i++ {
		test := &atoi64test[i];
		out, err := strconv.atoi64(test.in);
		if test.out != out || test.err != err {
			t.Errorf("strconv.atoi64(%v) = %v, %v want %v, %v\n",
				test.in, out, err, test.out, test.err);
		}
	}
}

func IntSize1() uint {
	tmp := 1;
	if tmp<<16<<16 == 0 {
		return 32;
	}
	return 64;
}

export func TestAtoui(t *testing.T) {
	switch IntSize1() {
	case 32:
		for i := 0; i < len(atoui32tests); i++ {
			test := &atoui32tests[i];
			out, err := strconv.atoui(test.in);
			if test.out != uint32(out) || test.err != err {
				t.Errorf("strconv.atoui(%v) = %v, %v want %v, %v\n",
					test.in, out, err, test.out, test.err);
			}
		}
	case 64:
		for i := 0; i < len(atoui64tests); i++ {
			test := &atoui64tests[i];
			out, err := strconv.atoui(test.in);
			if test.out != uint64(out) || test.err != err {
				t.Errorf("strconv.atoui(%v) = %v, %v want %v, %v\n",
					test.in, out, err, test.out, test.err);
			}
		}
	}
}

export func TestAtoi(t *testing.T) {
	switch IntSize1() {
	case 32:
		for i := 0; i < len(atoi32tests); i++ {
			test := &atoi32tests[i];
			out, err := strconv.atoi(test.in);
			if test.out != int32(out) || test.err != err {
				t.Errorf("strconv.atoi(%v) = %v, %v want %v, %v\n",
					test.in, out, err, test.out, test.err);
			}
		}
	case 64:
		for i := 0; i < len(atoi64test); i++ {
			test := &atoi64test[i];
			out, err := strconv.atoi(test.in);
			if test.out != int64(out) || test.err != err {
				t.Errorf("strconv.atoi(%v) = %v, %v want %v, %v\n",
					test.in, out, err, test.out, test.err);
			}
		}
	}
}

