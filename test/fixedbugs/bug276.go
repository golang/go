// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG code should run

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test case for issue 789. The bug only appeared for GOARCH=386.

package main

func main() {
	i := 0
	x := 0

	a := (x & 1) << uint(1-i)
	
	s := uint(1-i)
	b := (x & 1) << s
	
	if a != b {
		panic(0)
	}
}
