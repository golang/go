// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func shift(x int) int { return 1 << (1 << (1 << (uint(x)))) }

func main() {
	if n := shift(2); n != 1<<(1<<(1<<2)) {
		println("bad shift", n)
		panic("fail")
	}
}
