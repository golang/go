// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// $G $D/$F.go || echo BUG: should compile

package P

var x int

func f() int {
	return P.x  // P should be visible
}

/*
uetli:~/Source/go1/test/bugs gri$ 6g bug105.go
bug105.go:8: P: undefined
bug105.go:9: illegal types for operand: RETURN
	(<int32>INT32)
*/
