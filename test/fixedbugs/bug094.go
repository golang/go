// compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f0() {
	const x = 0;
}


func f1() {
	x := 0;
	_ = x;
}


func main() {
	f0();
	f1();
}

/*
uetli:~/Source/go1/test/bugs gri$ 6g bug094.go && 6l bug094.6 && 6.out
bug094.go:11: left side of := must be a name
bad top
.   LITERAL-I0 l(343)
bug094.go:11: fatal error: walktype: top=3 LITERAL
uetli:~/Source/go1/test/bugs gri$
*/
