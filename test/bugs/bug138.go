// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// $G $D/$F.go || echo BUG should compile

package main

func main() {
	const c int = -1;
	var i int = -1;
	var xc uint = uint(c);  // this does not work
	var xi uint = uint(i);  // this works
}

/*
bug138.go:8: overflow converting constant to uint
bug138.go:8: illegal combination of literals CONV 7
*/
