// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T struct { a int }
var t *T

type I interface { M() }
var i I

func main() {
	// neither of these can work,
	// because i has an extra method
	// that t does not, so i cannot contain a t.
	i = t;	// ERROR "missing|incompatible"
	t = i;	// ERROR "missing|incompatible"
}
