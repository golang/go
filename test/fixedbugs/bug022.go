// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func putint(digits *string) {
	var i byte;
	i = (*digits)[7];  // compiles
	i = digits[7];  // ERROR "illegal|is not|invalid"
	_ = i;
}

func main() {
	s := "asdfasdfasdfasdf";
	putint(&s);
}

/*
bug022.go:8: illegal types for operand
	(*<string>*STRING) INDEXPTR (<int32>INT32)
bug022.go:8: illegal types for operand
	(<uint8>UINT8) AS
*/
