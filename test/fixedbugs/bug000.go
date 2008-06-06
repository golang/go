// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var x int;
	switch x {
	case 0:
		{}
	case 1:
		x = 0;
	}
}
/*
bug0.go:8: case statement out of place
*/
