// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "strconv"

type Test struct {
	in string;
	out string;
}

var tests = []Test {
	Test{ "1", "1" },
	Test{ "1e23", "1e+23" },
	Test{ "100000000000000000000000", "1e+23" },
	Test{ "1e-100", "1e-100" },
	Test{ "123456700", "1.234567e+08" },
	Test{ "99999999999999974834176", "9.999999999999997e+22" },
	Test{ "100000000000000000000001", "1.0000000000000001e+23" },
	Test{ "100000000000000008388608", "1.0000000000000001e+23" },
	Test{ "100000000000000016777215", "1.0000000000000001e+23" },
	Test{ "100000000000000016777216", "1.0000000000000003e+23" },
	Test{ "-1", "-1" },
	Test{ "-0", "0" },
	Test{ "1e-20", "1e-20" },

	// largest float64
	Test{ "1.7976931348623157e308", "1.7976931348623157e+308" },
	Test{ "-1.7976931348623157e308", "-1.7976931348623157e+308" },
	// next float64 - too large
	Test{ "1.7976931348623159e308", "+Inf" },
	Test{ "-1.7976931348623159e308", "-Inf" },
	// the border is ...158079
	// borderline - okay
	Test{ "1.7976931348623158e308", "1.7976931348623157e+308" },
	Test{ "-1.7976931348623158e308", "-1.7976931348623157e+308" },
	// borderline - too large
	Test{ "1.797693134862315808e308", "+Inf" },
	Test{ "-1.797693134862315808e308", "-Inf" },

	// a little too large
	Test{ "1e308", "1e+308" },
	Test{ "2e308", "+Inf" },
	Test{ "1e309", "+Inf" },

	// way too large
	Test{ "1e310", "+Inf" },
	Test{ "-1e310", "-Inf" },
	Test{ "1e400", "+Inf" },
	Test{ "-1e400", "-Inf" },
	Test{ "1e400000", "+Inf" },
	Test{ "-1e400000", "-Inf" },

	// denormalized
	Test{ "1e-305", "1e-305" },
	Test{ "1e-306", "1e-306" },
	Test{ "1e-307", "1e-307" },
	Test{ "1e-308", "1e-308" },
	Test{ "1e-309", "1e-309" },
	Test{ "1e-310", "1e-310" },
	Test{ "1e-322", "1e-322" },
	// smallest denormal
	Test{ "5e-324", "5e-324" },
	// too small
	Test{ "4e-324", "0" },
	// way too small
	Test{ "1e-350", "0" },
	Test{ "1e-400000", "0" },

	// try to overflow exponent
	Test{ "1e-4294967296", "0" },
	Test{ "1e+4294967296", "+Inf" },
	Test{ "1e-18446744073709551616", "0" },
	Test{ "1e+18446744073709551616", "+Inf" },

	// Parse errors
	Test{ "1e", "error" },
	Test{ "1e-", "error" },
	Test{ ".e-1", "error" },
}

func main() {
	bad := 0;
	for i := 0; i < len(tests); i++ {
		t := &tests[i];
		f, overflow, ok := strconv.atof64(t.in);
		if !ok && t.out == "error" {
			continue;
		}
		if !ok {
			panicln("test:", t.in, "failed to parse");
		}
		if overflow && !sys.isInf(f, 0) {
			panicln("overflow but not inf:", t.in, f);
		}
		if sys.isInf(f, 0) && !overflow {
			panicln("inf but not overflow:", t.in, f);
		}
		s := strconv.ftoa64(f, 'g', -1);
		if s != t.out {
			println("test", t.in, "want", t.out, "got", s);
			bad++;
		}
	}
	if bad != 0 {
		panic("failed");
	}
}
