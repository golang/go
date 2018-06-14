// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Box struct {};
var m map[string] *Box;

func main() {
	m := make(map[string] *Box);
	s := "foo";
	var x *Box = nil;
	m[s] = x;
}

/*
bug058.go:9: illegal types for operand: INDEX
	(MAP[<string>*STRING]*<Box>{})
	(<string>*STRING)
*/
