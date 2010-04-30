// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// http://code.google.com/p/go/issues/detail?id=589

package main

func main() {	
	n := int64(100)
	x := make([]int, n)
	x[99] = 234;	
	z := x[n-1]
	if z != 234 {
		println("BUG")
	}
	n |= 1<<32
	defer func() {
		recover()
	}()
	z = x[n-1]
	println("BUG2")
}
