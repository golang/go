// $G $D/$F.go || echo BUG: compilation should succeed

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func
swap(x, y int) (u, v int) {
	return y, x
}

func
main() {
	a := 1;
	b := 2;
	a, b = swap(swap(a, b));
	if a != 2 || b != 1 {
		panic("bad swap");
	}
}
