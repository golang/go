// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	"fmt";
	"os";
	"reflect";
	. "strconv";
	"testing";
)

type atofTest struct {
	in string;
	out string;
	err os.Error;
}

var atoftests = []atofTest {
	atofTest{ "", "0", os.EINVAL },
	atofTest{ "1", "1", nil },
	atofTest{ "+1", "1", nil },
	atofTest{ "1x", "0", os.EINVAL },
	atofTest{ "1.1.", "0", os.EINVAL },
	atofTest{ "1e23", "1e+23", nil },
	atofTest{ "100000000000000000000000", "1e+23", nil },
	atofTest{ "1e-100", "1e-100", nil },
	atofTest{ "123456700", "1.234567e+08", nil },
	atofTest{ "99999999999999974834176", "9.999999999999997e+22", nil },
	atofTest{ "100000000000000000000001", "1.0000000000000001e+23", nil },
	atofTest{ "100000000000000008388608", "1.0000000000000001e+23", nil },
	atofTest{ "100000000000000016777215", "1.0000000000000001e+23", nil },
	atofTest{ "100000000000000016777216", "1.0000000000000003e+23", nil },
	atofTest{ "-1", "-1", nil },
	atofTest{ "-0", "-0", nil },
	atofTest{ "1e-20", "1e-20", nil },
	atofTest{ "625e-3", "0.625", nil },

	// largest float64
	atofTest{ "1.7976931348623157e308", "1.7976931348623157e+308", nil },
	atofTest{ "-1.7976931348623157e308", "-1.7976931348623157e+308", nil },
	// next float64 - too large
	atofTest{ "1.7976931348623159e308", "+Inf", os.ERANGE },
	atofTest{ "-1.7976931348623159e308", "-Inf", os.ERANGE },
	// the border is ...158079
	// borderline - okay
	atofTest{ "1.7976931348623158e308", "1.7976931348623157e+308", nil },
	atofTest{ "-1.7976931348623158e308", "-1.7976931348623157e+308", nil },
	// borderline - too large
	atofTest{ "1.797693134862315808e308", "+Inf", os.ERANGE },
	atofTest{ "-1.797693134862315808e308", "-Inf", os.ERANGE },

	// a little too large
	atofTest{ "1e308", "1e+308", nil },
	atofTest{ "2e308", "+Inf", os.ERANGE },
	atofTest{ "1e309", "+Inf", os.ERANGE },

	// way too large
	atofTest{ "1e310", "+Inf", os.ERANGE },
	atofTest{ "-1e310", "-Inf", os.ERANGE },
	atofTest{ "1e400", "+Inf", os.ERANGE },
	atofTest{ "-1e400", "-Inf", os.ERANGE },
	atofTest{ "1e400000", "+Inf", os.ERANGE },
	atofTest{ "-1e400000", "-Inf", os.ERANGE },

	// denormalized
	atofTest{ "1e-305", "1e-305", nil },
	atofTest{ "1e-306", "1e-306", nil },
	atofTest{ "1e-307", "1e-307", nil },
	atofTest{ "1e-308", "1e-308", nil },
	atofTest{ "1e-309", "1e-309", nil },
	atofTest{ "1e-310", "1e-310", nil },
	atofTest{ "1e-322", "1e-322", nil },
	// smallest denormal
	atofTest{ "5e-324", "5e-324", nil },
	// too small
	atofTest{ "4e-324", "0", nil },
	// way too small
	atofTest{ "1e-350", "0", nil },
	atofTest{ "1e-400000", "0", nil },

	// try to overflow exponent
	atofTest{ "1e-4294967296", "0", nil },
	atofTest{ "1e+4294967296", "+Inf", os.ERANGE },
	atofTest{ "1e-18446744073709551616", "0", nil },
	atofTest{ "1e+18446744073709551616", "+Inf", os.ERANGE },

	// Parse errors
	atofTest{ "1e", "0", os.EINVAL },
	atofTest{ "1e-", "0", os.EINVAL },
	atofTest{ ".e-1", "0", os.EINVAL },
}

func init() {
	// The atof routines return NumErrors wrapping
	// the error and the string.  Convert the table above.
	for i := range atoftests {
		test := &atoftests[i];
		if test.err != nil {
			test.err = &NumError{test.in, test.err}
		}
	}
}

func testAtof(t *testing.T, opt bool) {
	oldopt := SetOptimize(opt);
	for i := 0; i < len(atoftests); i++ {
		test := &atoftests[i];
		out, err := Atof64(test.in);
		outs := Ftoa64(out, 'g', -1);
		if outs != test.out || !reflect.DeepEqual(err, test.err) {
			t.Errorf("Atof64(%v) = %v, %v want %v, %v\n",
				test.in, out, err, test.out, test.err);
		}

		if float64(float32(out)) == out {
			out32, err := Atof32(test.in);
			outs := Ftoa32(out32, 'g', -1);
			if outs != test.out || !reflect.DeepEqual(err, test.err) {
				t.Errorf("Atof32(%v) = %v, %v want %v, %v  # %v\n",
					test.in, out32, err, test.out, test.err, out);
			}
		}

		if FloatSize == 64 || float64(float32(out)) == out {
			outf, err := Atof(test.in);
			outs := Ftoa(outf, 'g', -1);
			if outs != test.out || !reflect.DeepEqual(err, test.err) {
				t.Errorf("Ftoa(%v) = %v, %v want %v, %v  # %v\n",
					test.in, outf, err, test.out, test.err, out);
			}
		}
	}
	SetOptimize(oldopt);
}

func TestAtof(t *testing.T) {
	testAtof(t, true);
}

func TestAtofSlow(t *testing.T) {
	testAtof(t, false);
}
