// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

import (
	"strconv";
	"testing"
)

type Test struct {
	f float64;
	fmt byte;
	prec int;
	s string;
}

func fdiv(a, b float64) float64 { return a / b }	// keep compiler in the dark

// TODO: Should be able to call this tests but it conflicts with testatof.go
var ftests = []Test {
	Test{ 1, 'e', 5, "1.00000e+00" },
	Test{ 1, 'f', 5, "1.00000" },
	Test{ 1, 'g', 5, "1" },
	Test{ 1, 'g', -1, "1" },

	Test{ 0, 'e', 5, "0.00000e+00" },
	Test{ 0, 'f', 5, "0.00000" },
	Test{ 0, 'g', 5, "0" },
	Test{ 0, 'g', -1, "0" },

	Test{ -1, 'e', 5, "-1.00000e+00" },
	Test{ -1, 'f', 5, "-1.00000" },
	Test{ -1, 'g', 5, "-1" },
	Test{ -1, 'g', -1, "-1" },

	Test{ 12, 'e', 5, "1.20000e+01" },
	Test{ 12, 'f', 5, "12.00000" },
	Test{ 12, 'g', 5, "12" },
	Test{ 12, 'g', -1, "12" },

	Test{ 123456700, 'e', 5, "1.23457e+08" },
	Test{ 123456700, 'f', 5, "123456700.00000" },
	Test{ 123456700, 'g', 5, "1.2346e+08" },
	Test{ 123456700, 'g', -1, "1.234567e+08" },

	Test{ 1.2345e6, 'e', 5, "1.23450e+06" },
	Test{ 1.2345e6, 'f', 5, "1234500.00000" },
	Test{ 1.2345e6, 'g', 5, "1.2345e+06" },

	Test{ 1e23, 'e', 17, "9.99999999999999916e+22" },
	Test{ 1e23, 'f', 17, "99999999999999991611392.00000000000000000" },
	Test{ 1e23, 'g', 17, "9.9999999999999992e+22" },

	Test{ 1e23, 'e', -1, "1e+23" },
	Test{ 1e23, 'f', -1, "100000000000000000000000" },
	Test{ 1e23, 'g', -1, "1e+23" },

	Test{ 1e23-8.5e6, 'e', 17, "9.99999999999999748e+22" },
	Test{ 1e23-8.5e6, 'f', 17, "99999999999999974834176.00000000000000000" },
	Test{ 1e23-8.5e6, 'g', 17, "9.9999999999999975e+22" },

	Test{ 1e23-8.5e6, 'e', -1, "9.999999999999997e+22" },
	Test{ 1e23-8.5e6, 'f', -1, "99999999999999970000000" },
	Test{ 1e23-8.5e6, 'g', -1, "9.999999999999997e+22" },

	Test{ 1e23+8.5e6, 'e', 17, "1.00000000000000008e+23" },
	Test{ 1e23+8.5e6, 'f', 17, "100000000000000008388608.00000000000000000" },
	Test{ 1e23+8.5e6, 'g', 17, "1.0000000000000001e+23" },

	Test{ 1e23+8.5e6, 'e', -1, "1.0000000000000001e+23" },
	Test{ 1e23+8.5e6, 'f', -1, "100000000000000010000000" },
	Test{ 1e23+8.5e6, 'g', -1, "1.0000000000000001e+23" },

	Test{ fdiv(5e-304, 1e20), 'g', -1, "5e-324" },
	Test{ fdiv(-5e-304, 1e20), 'g', -1, "-5e-324" },

	Test{ 32, 'g', -1, "32" },
	Test{ 32, 'g', 0, "3e+01" },

	Test{ 100, 'x', -1, "%x" },

	Test{ sys.NaN(), 'g', -1, "NaN" },
	Test{ -sys.NaN(), 'g', -1, "NaN" },
	Test{ sys.Inf(0), 'g', -1, "+Inf" },
	Test{ sys.Inf(-1), 'g', -1,  "-Inf" },
	Test{ -sys.Inf(0), 'g', -1, "-Inf" },

	Test{ -1, 'b', -1, "-4503599627370496p-52" },
}

export func TestFtoa(t *testing.T) {
	if strconv.floatsize != 32 {
		panic("floatsize: ", strconv.floatsize);
	}
	for i := 0; i < len(ftests); i++ {
		test := &ftests[i];
		s := strconv.ftoa64(test.f, test.fmt, test.prec);
		if s != test.s {
			t.Error("test", test.f, string(test.fmt), test.prec, "want", test.s, "got", s);
		}
		if float64(float32(test.f)) == test.f && test.fmt != 'b' {
			s := strconv.ftoa32(float32(test.f), test.fmt, test.prec);
			if s != test.s {
				t.Error("test32", test.f, string(test.fmt), test.prec, "want", test.s, "got", s);
			}
		}
	}
}
