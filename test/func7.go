// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var calledf = false

func f() int {
	calledf = true
	return 1
}

func g() int {
	if !calledf {
		println("BUG: func7 - called g before f")
	}
	return 0
}

func main() {
	// 6g, 8g, 5g all used to evaluate g() before f().
	if f() < g() {
		panic("wrong answer")
	}
}

