// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// 5g bug used to set up the 0 for -f() before calling f,
// and the call to f smashed the register.

func f(n int) int {
	s := 0
	for i := 0; i < n; i++ {
		s += i>>1
	}
	return s
}

func main() {
	x := -f(100)
	if x != -2450 {
		println(x)
		panic("broken")
	}
}
