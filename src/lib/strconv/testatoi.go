// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv
import (
	"os";
	"fmt";
	"strconv"
)

type Uint64Test struct {
	in string;
	out uint64;
	err *os.Error;
}

var uint64tests = []Uint64Test {
	Uint64Test{ "0", 0, nil },
	Uint64Test{ "1", 1, nil },
	Uint64Test{ "12345", 12345, nil },
	Uint64Test{ "012345", 0, os.EINVAL },
	Uint64Test{ "12345x", 0, os.EINVAL },
	Uint64Test{ "98765432100", 98765432100, nil },
	Uint64Test{ "18446744073709551615", 1<<64-1, nil },
	Uint64Test{ "18446744073709551616", 1<<64-1, os.ERANGE },
}

type Int64Test struct {
	in string;
	out int64;
	err *os.Error;
}

var int64tests = []Int64Test {
	Int64Test{ "0", 0, nil },
	Int64Test{ "-0", 0, nil },
	Int64Test{ "1", 1, nil },
	Int64Test{ "-1", -1, nil },
	Int64Test{ "12345", 12345, nil },
	Int64Test{ "-12345", -12345, nil },
	Int64Test{ "012345", 0, os.EINVAL },
	Int64Test{ "-012345", 0, os.EINVAL },
	Int64Test{ "12345x", 0, os.EINVAL },
	Int64Test{ "-12345x", 0, os.EINVAL },
	Int64Test{ "98765432100", 98765432100, nil },
	Int64Test{ "-98765432100", -98765432100, nil },
	Int64Test{ "9223372036854775807", 1<<63-1, nil },
	Int64Test{ "-9223372036854775807", -(1<<63-1), nil },
	Int64Test{ "9223372036854775808", 1<<63-1, os.ERANGE },
	Int64Test{ "-9223372036854775808", -1<<63, nil },
	Int64Test{ "9223372036854775809", 1<<63-1, os.ERANGE },
	Int64Test{ "-9223372036854775809", -1<<63, os.ERANGE },
}

type Uint32Test struct {
	in string;
	out uint32;
	err *os.Error;
}

var uint32tests = []Uint32Test {
	Uint32Test{ "0", 0, nil },
	Uint32Test{ "1", 1, nil },
	Uint32Test{ "12345", 12345, nil },
	Uint32Test{ "012345", 0, os.EINVAL },
	Uint32Test{ "12345x", 0, os.EINVAL },
	Uint32Test{ "987654321", 987654321, nil },
	Uint32Test{ "4294967295", 1<<32-1, nil },
	Uint32Test{ "4294967296", 1<<32-1, os.ERANGE },
}

type Int32Test struct {
	in string;
	out int32;
	err *os.Error;
}

var int32tests = []Int32Test {
	Int32Test{ "0", 0, nil },
	Int32Test{ "-0", 0, nil },
	Int32Test{ "1", 1, nil },
	Int32Test{ "-1", -1, nil },
	Int32Test{ "12345", 12345, nil },
	Int32Test{ "-12345", -12345, nil },
	Int32Test{ "012345", 0, os.EINVAL },
	Int32Test{ "-012345", 0, os.EINVAL },
	Int32Test{ "12345x", 0, os.EINVAL },
	Int32Test{ "-12345x", 0, os.EINVAL },
	Int32Test{ "987654321", 987654321, nil },
	Int32Test{ "-987654321", -987654321, nil },
	Int32Test{ "2147483647", 1<<31-1, nil },
	Int32Test{ "-2147483647", -(1<<31-1), nil },
	Int32Test{ "2147483648", 1<<31-1, os.ERANGE },
	Int32Test{ "-2147483648", -1<<31, nil },
	Int32Test{ "2147483649", 1<<31-1, os.ERANGE },
	Int32Test{ "-2147483649", -1<<31, os.ERANGE },
}

export func TestAtoui64() bool {
	ok := true;
	for i := 0; i < len(uint64tests); i++ {
		t := &uint64tests[i];
		out, err := strconv.atoui64(t.in);
		if t.out != out || t.err != err {
			fmt.printf("strconv.atoui64(%v) = %v, %v want %v, %v\n",
				t.in, out, err, t.out, t.err);
			ok = false;
		}
	}
	return ok;
}

export func TestAtoi64() bool {
	ok := true;
	for i := 0; i < len(int64tests); i++ {
		t := &int64tests[i];
		out, err := strconv.atoi64(t.in);
		if t.out != out || t.err != err {
			fmt.printf("strconv.atoi64(%v) = %v, %v want %v, %v\n",
				t.in, out, err, t.out, t.err);
			ok = false;
		}
	}
	return ok;
}

func IntSize1() uint {
	tmp := 1;
	if tmp<<16<<16 == 0 {
		return 32;
	}
println("tmp<<32 = ", tmp<<32);
	return 64;
}

export func TestAtoui() bool {
	ok := true;
	switch IntSize1() {
	case 32:
		for i := 0; i < len(uint32tests); i++ {
			t := &uint32tests[i];
			out, err := strconv.atoui(t.in);
			if t.out != uint32(out) || t.err != err {
				fmt.printf("strconv.atoui(%v) = %v, %v want %v, %v\n",
					t.in, out, err, t.out, t.err);
				ok = false;
			}
		}
	case 64:
		for i := 0; i < len(uint64tests); i++ {
			t := &uint64tests[i];
			out, err := strconv.atoui(t.in);
			if t.out != uint64(out) || t.err != err {
				fmt.printf("strconv.atoui(%v) = %v, %v want %v, %v\n",
					t.in, out, err, t.out, t.err);
				ok = false;
			}
		}
	}
	return ok;
}

export func TestAtoi() bool {
	ok := true;
	switch IntSize1() {
	case 32:
		for i := 0; i < len(int32tests); i++ {
			t := &int32tests[i];
			out, err := strconv.atoi(t.in);
			if t.out != int32(out) || t.err != err {
				fmt.printf("strconv.atoi(%v) = %v, %v want %v, %v\n",
					t.in, out, err, t.out, t.err);
				ok = false;
			}
		}
	case 64:
		for i := 0; i < len(int64tests); i++ {
			t := &int64tests[i];
			out, err := strconv.atoi(t.in);
			if t.out != int64(out) || t.err != err {
				fmt.printf("strconv.atoi(%v) = %v, %v want %v, %v\n",
					t.in, out, err, t.out, t.err);
				ok = false;
			}
		}
	}
	return ok;
}

