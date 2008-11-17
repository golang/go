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
}

func main() {
	bad := 0;
	for i := 0; i < len(tests); i++ {
		t := &tests[i];
		f, overflow, ok := strconv.atof64(t.in);
		if !ok {
			panicln("test", t.in);
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
