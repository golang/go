// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv
import (
	"fmt";
	"os";
	"strconv"
)

type Test struct {
	in string;
	out string;
	err *os.Error;
}

var tests = []Test {
	Test{ "", "0", os.EINVAL },
	Test{ "1", "1", nil },
	Test{ "+1", "1", nil },
	Test{ "1x", "0", os.EINVAL },
	Test{ "1.1.", "0", os.EINVAL },
	Test{ "1e23", "1e+23", nil },
	Test{ "100000000000000000000000", "1e+23", nil },
	Test{ "1e-100", "1e-100", nil },
	Test{ "123456700", "1.234567e+08", nil },
	Test{ "99999999999999974834176", "9.999999999999997e+22", nil },
	Test{ "100000000000000000000001", "1.0000000000000001e+23", nil },
	Test{ "100000000000000008388608", "1.0000000000000001e+23", nil },
	Test{ "100000000000000016777215", "1.0000000000000001e+23", nil },
	Test{ "100000000000000016777216", "1.0000000000000003e+23", nil },
	Test{ "-1", "-1", nil },
	Test{ "-0", "0", nil },
	Test{ "1e-20", "1e-20", nil },
	Test{ "625e-3", "0.625", nil },

	// largest float64
	Test{ "1.7976931348623157e308", "1.7976931348623157e+308", nil },
	Test{ "-1.7976931348623157e308", "-1.7976931348623157e+308", nil },
	// next float64 - too large
	Test{ "1.7976931348623159e308", "+Inf", os.ERANGE },
	Test{ "-1.7976931348623159e308", "-Inf", os.ERANGE },
	// the border is ...158079
	// borderline - okay
	Test{ "1.7976931348623158e308", "1.7976931348623157e+308", nil },
	Test{ "-1.7976931348623158e308", "-1.7976931348623157e+308", nil },
	// borderline - too large
	Test{ "1.797693134862315808e308", "+Inf", os.ERANGE },
	Test{ "-1.797693134862315808e308", "-Inf", os.ERANGE },

	// a little too large
	Test{ "1e308", "1e+308", nil },
	Test{ "2e308", "+Inf", os.ERANGE },
	Test{ "1e309", "+Inf", os.ERANGE },

	// way too large
	Test{ "1e310", "+Inf", os.ERANGE },
	Test{ "-1e310", "-Inf", os.ERANGE },
	Test{ "1e400", "+Inf", os.ERANGE },
	Test{ "-1e400", "-Inf", os.ERANGE },
	Test{ "1e400000", "+Inf", os.ERANGE },
	Test{ "-1e400000", "-Inf", os.ERANGE },

	// denormalized
	Test{ "1e-305", "1e-305", nil },
	Test{ "1e-306", "1e-306", nil },
	Test{ "1e-307", "1e-307", nil },
	Test{ "1e-308", "1e-308", nil },
	Test{ "1e-309", "1e-309", nil },
	Test{ "1e-310", "1e-310", nil },
	Test{ "1e-322", "1e-322", nil },
	// smallest denormal
	Test{ "5e-324", "5e-324", nil },
	// too small
	Test{ "4e-324", "0", nil },
	// way too small
	Test{ "1e-350", "0", nil },
	Test{ "1e-400000", "0", nil },

	// try to overflow exponent
	Test{ "1e-4294967296", "0", nil },
	Test{ "1e+4294967296", "+Inf", os.ERANGE },
	Test{ "1e-18446744073709551616", "0", nil },
	Test{ "1e+18446744073709551616", "+Inf", os.ERANGE },

	// Parse errors
	Test{ "1e", "0", os.EINVAL },
	Test{ "1e-", "0", os.EINVAL },
	Test{ ".e-1", "0", os.EINVAL },
}

func XTestAtof(opt bool) bool {
	oldopt := strconv.optimize;
	strconv.optimize = opt;
	ok := true;
	for i := 0; i < len(tests); i++ {
		t := &tests[i];
		out, err := strconv.atof64(t.in);
		outs := strconv.ftoa64(out, 'g', -1);
		if outs != t.out || err != t.err {
			fmt.printf("strconv.atof64(%v) = %v, %v want %v, %v\n",
				t.in, out, err, t.out, t.err);
			ok = false;
		}

		if float64(float32(out)) == out {
			out32, err := strconv.atof32(t.in);
			outs := strconv.ftoa32(out32, 'g', -1);
			if outs != t.out || err != t.err {
				fmt.printf("strconv.atof32(%v) = %v, %v want %v, %v  # %v\n",
					t.in, out32, err, t.out, t.err, out);
				ok = false;
			}
		}

		if floatsize == 64 || float64(float32(out)) == out {
			outf, err := strconv.atof(t.in);
			outs := strconv.ftoa(outf, 'g', -1);
			if outs != t.out || err != t.err {
				fmt.printf("strconv.ftoa(%v) = %v, %v want %v, %v  # %v\n",
					t.in, outf, err, t.out, t.err, out);
				ok = false;
			}
		}
	}
	strconv.optimize = oldopt;
	return ok;
}

export func TestAtof() bool {
	return XTestAtof(true);
}

export func TestAtofSlow() bool {
	return XTestAtof(false);
}
