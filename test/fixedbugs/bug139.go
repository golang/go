// $G $D/$F.go || echo BUG should compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	x := false;
	func () { if x          { println(1); } }();  // this does not compile
	func () { if x == false { println(2); } }();  // this works as expected
}

/*
bug139.go:7: fatal error: naddr: ONAME class x 5
*/
