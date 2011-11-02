// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG: bug331

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "io"

func f() (_ string, x float64, err error) {
	return
}

func g() (_ string, x float64, err error) {
	return "hello", 3.14, io.EOF
}

var _ func() (string, float64, error) = f
var _ func() (string, float64, error) = g

func main() {
	x, y, z := g()
	if x != "hello" || y != 3.14 || z != io.EOF {
		println("wrong", x, len(x), y, z)
	}
}

/*
issue 1712

bug331.go:12: cannot use "hello" (type string) as type float64 in assignment
bug331.go:12: cannot use 0 (type float64) as type os.Error in assignment:
	float64 does not implement os.Error (missing String method)
bug331.go:12: error in shape across RETURN
*/
