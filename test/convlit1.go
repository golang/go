// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var a = []int { "a" };	// ERROR "conver|incompatible"
var b = int { 1 };	// ERROR "compos"


func main() {
	if sys.argc < 1 { }	// ERROR "conver|incompatible"
}
