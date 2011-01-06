// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG: bug317

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	x := []uint{0}
	x[0] &^= f()
}

func f() uint {
	return 1<<31 // doesn't panic with 1<<31 - 1
}
