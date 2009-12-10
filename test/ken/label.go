// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


package main

func
main() {
	i := 0;
	if false {
		goto gogoloop;
	}
	if false {
		goto gogoloop;
	}
	if false {
		goto gogoloop;
	}
	goto gogoloop;

// backward declared
loop:
	i = i+1;
	if i < 100 {
		goto loop;
	}
	print(i);
	print("\n");
	return;

gogoloop:
	goto loop;
}
