// $G $D/$F.go || echo BUG

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T []int

var a []bool

func _() {
	if a[T{42}[0]] {
	}
	// if (a[T{42}[0]]) {}  // this compiles
}

/*
6g bugs/bug267.go
bugs/bug267.go:14: syntax error: unexpected {, expecting :
*/
