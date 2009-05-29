// $G $D/$F.go || echo BUG: bug156

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f(a int64) int64 {
	const b int64 = 0;
	n := a &^ b;
	return n;
}

func main() {
	f(1)
}

/*
bug156.go:7: constant 18446744073709551615 overflows int64
*/
