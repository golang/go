// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import P "./bug0"

func main() {
	a0 := P.V0();  // works
	a1 := P.V1();  // works
	a2, b2 := P.V2();  // doesn't work
	_, _, _, _ = a0, a1, a2, b2;
}

/*
uetli:~/Source/go1/test/bugs/bug088.dir gri$ 6g bug0.go && 6g bug1.go
bug1.go:8: shape error across :=
bug1.go:8: a2: undefined
bug1.go:8: b2: undefined
bug1.go:8: illegal types for operand: AS
	(<(bug0)P.int32>INT32)
*/
