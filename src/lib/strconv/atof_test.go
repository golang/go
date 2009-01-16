// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv
import (
	"fmt";
	"os";
	"strconv";
	"testing"
)

type AtofTest struct {
	in string;
	out string;
	err *os.Error;
}

var atoftests = []AtofTest {
	AtofTest{ "", "0", os.EINVAL },
	AtofTest{ "1", "1", nil },
	AtofTest{ "+1", "1", nil },
	AtofTest{ "1x", "0", os.EINVAL },
	AtofTest{ "1.1.", "0", os.EINVAL },
	AtofTest{ "1e23", "1e+23", nil },
	AtofTest{ "100000000000000000000000", "1e+23", nil },
	AtofTest{ "1e-100", "1e-100", nil },
	AtofTest{ "123456700", "1.234567e+08", nil },
	AtofTest{ "99999999999999974834176", "9.999999999999997e+22", nil },
	AtofTest{ "100000000000000000000001", "1.0000000000000001e+23", nil },
	AtofTest{ "100000000000000008388608", "1.0000000000000001e+23", nil },
	AtofTest{ "100000000000000016777215", "1.0000000000000001e+23", nil },
	AtofTest{ "100000000000000016777216", "1.0000000000000003e+23", nil },
	AtofTest{ "-1", "-1", nil },
	AtofTest{ "-0", "-0", nil },
	AtofTest{ "1e-20", "1e-20", nil },
	AtofTest{ "625e-3", "0.625", nil },

	// largest float64
	AtofTest{ "1.7976931348623157e308", "1.7976931348623157e+308", nil },
	AtofTest{ "-1.7976931348623157e308", "-1.7976931348623157e+308", nil },
	// next float64 - too large
	AtofTest{ "1.7976931348623159e308", "+Inf", os.ERANGE },
	AtofTest{ "-1.7976931348623159e308", "-Inf", os.ERANGE },
	// the border is ...158079
	// borderline - okay
	AtofTest{ "1.7976931348623158e308", "1.7976931348623157e+308", nil },
	AtofTest{ "-1.7976931348623158e308", "-1.7976931348623157e+308", nil },
	// borderline - too large
	AtofTest{ "1.797693134862315808e308", "+Inf", os.ERANGE },
	AtofTest{ "-1.797693134862315808e308", "-Inf", os.ERANGE },

	// a little too large
	AtofTest{ "1e308", "1e+308", nil },
	AtofTest{ "2e308", "+Inf", os.ERANGE },
	AtofTest{ "1e309", "+Inf", os.ERANGE },

	// way too large
	AtofTest{ "1e310", "+Inf", os.ERANGE },
	AtofTest{ "-1e310", "-Inf", os.ERANGE },
	AtofTest{ "1e400", "+Inf", os.ERANGE },
	AtofTest{ "-1e400", "-Inf", os.ERANGE },
	AtofTest{ "1e400000", "+Inf", os.ERANGE },
	AtofTest{ "-1e400000", "-Inf", os.ERANGE },

	// denormalized
	AtofTest{ "1e-305", "1e-305", nil },
	AtofTest{ "1e-306", "1e-306", nil },
	AtofTest{ "1e-307", "1e-307", nil },
	AtofTest{ "1e-308", "1e-308", nil },
	AtofTest{ "1e-309", "1e-309", nil },
	AtofTest{ "1e-310", "1e-310", nil },
	AtofTest{ "1e-322", "1e-322", nil },
	// smallest denormal
	AtofTest{ "5e-324", "5e-324", nil },
	// too small
	AtofTest{ "4e-324", "0", nil },
	// way too small
	AtofTest{ "1e-350", "0", nil },
	AtofTest{ "1e-400000", "0", nil },

	// try to overflow exponent
	AtofTest{ "1e-4294967296", "0", nil },
	AtofTest{ "1e+4294967296", "+Inf", os.ERANGE },
	AtofTest{ "1e-18446744073709551616", "0", nil },
	AtofTest{ "1e+18446744073709551616", "+Inf", os.ERANGE },

	// Parse errors
	AtofTest{ "1e", "0", os.EINVAL },
	AtofTest{ "1e-", "0", os.EINVAL },
	AtofTest{ ".e-1", "0", os.EINVAL },
}

func XTestAtof(t *testing.T, opt bool) {
	oldopt := strconv.optimize;
	strconv.optimize = opt;
	for i := 0; i < len(atoftests); i++ {
		test := &atoftests[i];
		out, err := strconv.atof64(test.in);
		outs := strconv.ftoa64(out, 'g', -1);
		if outs != test.out || err != test.err {
			t.Errorf("strconv.atof64(%v) = %v, %v want %v, %v\n",
				test.in, out, err, test.out, test.err);
		}

		if float64(float32(out)) == out {
			out32, err := strconv.atof32(test.in);
			outs := strconv.ftoa32(out32, 'g', -1);
			if outs != test.out || err != test.err {
				t.Errorf("strconv.atof32(%v) = %v, %v want %v, %v  # %v\n",
					test.in, out32, err, test.out, test.err, out);
			}
		}

		if FloatSize == 64 || float64(float32(out)) == out {
			outf, err := strconv.atof(test.in);
			outs := strconv.ftoa(outf, 'g', -1);
			if outs != test.out || err != test.err {
				t.Errorf("strconv.ftoa(%v) = %v, %v want %v, %v  # %v\n",
					test.in, outf, err, test.out, test.err, out);
			}
		}
	}
	strconv.optimize = oldopt;
}

export func TestAtof(t *testing.T) {
	XTestAtof(t, true);
}

export func TestAtofSlow(t *testing.T) {
	XTestAtof(t, false);
}
