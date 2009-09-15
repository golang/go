// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	s := 0;
	for _, v := range []int{1} {
		s += v;
	}
	if s != 1 {
		println("BUG: s =", s);
	}
}
