// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG: bug120

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "strconv";

type Test struct {
	f float64;
	in string;
	out string;
}

var tests = []Test {
	Test{ 123.5, "123.5", "123.5" },
	Test{ 456.7, "456.7", "456.7" },
	Test{ 1e23+8.5e6, "1e23+8.5e6", "1.0000000000000001e+23" },
	Test{ 100000000000000008388608, "100000000000000008388608", "1.0000000000000001e+23" },
	Test{ 1e23+8.388608e6, "1e23+8.388608e6", "1.0000000000000001e+23" },
	Test{ 1e23+8.388609e6, "1e23+8.388609e6", "1.0000000000000001e+23" },
}

func main() {
	ok := true;
	for i := 0; i < len(tests); i++ {
		t := tests[i];
		v := strconv.ftoa64(t.f, 'g', -1);
		if v != t.out {
			println("Bad float64 const:", t.in, "want", t.out, "got", v);
			ok = false;
		}
	}
	if !ok {
		panicln("bug120");
	}
}
