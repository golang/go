// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG wrong result

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type A []int

func main() {
	var a [3]A
	for i := 0; i < 3; i++ {
		a[i] = A{i}
	}
	if a[0][0] != 0 {
		panic("fail a[0][0]")
	}
	if a[1][0] != 1 {
		panic("fail a[1][0]")
	}
	if a[2][0] != 2 {
		panic("fail a[2][0]")
	}
}

/*
uetli:~/Source/go1/test/bugs gri$ 6g bug097.go && 6l bug097.6 && 6.out

panic on line 342 PC=0x13c2
0x13c2?zi
	main·main(1, 0, 1606416416, ...)
	main·main(0x1, 0x7fff5fbff820, 0x0, ...)
SIGTRAP: trace trap
Faulting address: 0x4558
pc: 0x4558

0x4558?zi
	sys·Breakpoint(40960, 0, 45128, ...)
	sys·Breakpoint(0xa000, 0xb048, 0xa000, ...)
0x156a?zi
	sys·panicl(342, 0, 0, ...)
	sys·panicl(0x156, 0x300000000, 0xb024, ...)
0x13c2?zi
	main·main(1, 0, 1606416416, ...)
	main·main(0x1, 0x7fff5fbff820, 0x0, ...)
*/

/* An array composite literal needs to be created freshly every time.
It is a "construction" of an array after all. If I pass the address
of the array to some function, it may store it globally. Same applies
to struct literals.
*/
