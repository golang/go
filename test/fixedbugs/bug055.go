// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var i int;
	var j int;
	if true {}
	{ return }
	i = 0;
	if true {} else i++;
	type s struct {};
	i = 0;
	type s2 int;
	var k = func (a int) int { return a+1 }(3);
	_, _ = j, k;
ro: ;
}
