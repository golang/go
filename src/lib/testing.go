// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

export type Test struct {
	name string;
	f *() bool;
}

export func Main(tests *[]Test) {
	ok := true;
	for i := 0; i < len(tests); i++ {
		ok1 := tests[i].f();
		status := "FAIL";
		if ok1 {
			status = "PASS"
		}
		ok = ok && ok1;
		println(status, tests[i].name);
	}
	if !ok {
		sys.exit(1);
	}
}
