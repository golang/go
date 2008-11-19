// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"flag"
)

var chatty bool;
func init() {
	flag.Bool("chatty", false, &chatty, "chatty");
}

export type Test struct {
	name string;
	f *() bool;
}

export func Main(tests *[]Test) {
	flag.Parse();
	ok := true;
	for i := 0; i < len(tests); i++ {
		if chatty {
			println("=== RUN ", tests[i].name);
		}
		ok1 := tests[i].f();
		if !ok1 {
			ok = false;
			println("--- FAIL", tests[i].name);
		} else if chatty {
			println("--- PASS", tests[i].name);
		}
	}
	if !ok {
		sys.exit(1);
	}
	println("PASS");
}
