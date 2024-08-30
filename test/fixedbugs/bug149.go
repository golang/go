// compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var b1 []byte;
	s1 := string(b1);
	println(len(s1));  // prints 0

	b2 := ([]byte)(nil);
	s2 := string(b2);
	println(len(s2));  // prints 0

	s3 := string(([]byte)(nil));  // does not compile (literal substitution of b2)
	println(len(s3));
}

/*
bug149.go:14: cannot convert []uint8 constant to string
*/
