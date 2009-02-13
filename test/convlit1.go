// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var a = []int ( "a" );	// ERROR "conver|incompatible"


func f() int

func main() {
	if f < 1 { }	// ERROR "conver|incompatible"
}
