// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG: bug254

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var a [10]int
var b [1e1]int

func main() {
	if len(a) != 10 || len(b) != 10 {
		println("len", len(a), len(b))
		panic("fail")
	}
}
