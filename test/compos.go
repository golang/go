// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG: compos

// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T struct {
	int
}

func f() *T {
	return &T{1}
}

func main() {
	x := f()
	y := f()
	if x == y {
		panic("not allocating & composite literals")
	}
}
