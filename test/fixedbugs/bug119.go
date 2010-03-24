// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG: should not fail

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func foo(a []int) int {
	return a[0] // this seems to do the wrong thing
}

func main() {
	a := &[]int{12}
	if x := (*a)[0]; x != 12 {
		panic(2)
	}
	if x := foo(*a); x != 12 {
		// fails (x is incorrect)
		panic(3)
	}
}

/*
uetli:~/Source/go1/test/bugs gri$ 6go bug119
3 70160

panic on line 83 PC=0x14d6
0x14d6?zi
	main·main(23659, 0, 1, ...)
	main·main(0x5c6b, 0x1, 0x7fff5fbff830, ...)
0x52bb?zi
	mainstart(1, 0, 1606416432, ...)
	mainstart(0x1, 0x7fff5fbff830, 0x0, ...)
uetli:~/Source/go1/test/bugs gri$
*/
