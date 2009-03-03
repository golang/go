// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {

	type T struct {
		s string;
		f float;
	};
	var s string = "hello";
	var f float = 0.2;
	t := T{s, f};

	type M map[int] int;
	m0 := M{7:8};
}
