// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func assertequal(is, shouldbe int, msg string) {
	if is != shouldbe {
		print("assertion fail" + msg + "\n");
		panic(1);
	}
}

func main() {
	i5 := 5;
	i7 := 7;

	var count int;

	count = 0;
	if true {
		count = count + 1;
	}
	assertequal(count, 1, "if true");

	count = 0;
	if false {
		count = count + 1;
	}
	assertequal(count, 0, "if false");

	count = 0;
	if one := 1; true {
		count = count + one;
	}
	assertequal(count, 1, "if true one");

	count = 0;
	if one := 1; false {
		_ = one;
		count = count + 1;
	}
	assertequal(count, 0, "if false one");

	count = 0;
	if {
		count = count + 1;
	}
	assertequal(count, 1, "if empty");

	count = 0;
	if one := 1; {
		count = count + one;
	}
	assertequal(count, 1, "if empty one");

	count = 0;
	if i5 < i7 {
		count = count + 1;
	}
	assertequal(count, 1, "if cond");

	count = 0;
	if true {
		count = count + 1;
	} else
		count = count - 1;
	assertequal(count, 1, "if else true");

	count = 0;
	if false {
		count = count + 1;
	} else
		count = count - 1;
	assertequal(count, -1, "if else false");

	count = 0;
	if t:=1; false {
		count = count + 1;
		t := 7;
		_ = t;
	} else
		count = count - t;
	assertequal(count, -1, "if else false var");

	count = 0;
	t := 1;
	if false {
		count = count + 1;
		t := 7;
		_ = t;
	} else
		count = count - t;
	assertequal(count, -1, "if else false var outside");
}
