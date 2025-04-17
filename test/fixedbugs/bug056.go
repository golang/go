// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func frexp() (a int, b float64) {
	return 1, 2.0
}

func main() {
	a, b := frexp();
	_, _ = a, b;
}

/*
bug056.go:8: illegal types for operand: AS
	(<int32>INT32)
	(<int32>INT32)
*/
