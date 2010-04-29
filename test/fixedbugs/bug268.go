// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// http://code.google.com/p/go/issues/detail?id=745

package main

type T1 struct {
	T2 *T2
}

type T2 struct {
	T3 *T3
}

type T3 struct {
	T4 []*T4
}

type T4 struct {
	X int
}

func f() *T1 {
	x := &T1{
		&T2{
			&T3{
				[1]*T4{
					&T4{5},
				}[0:],
			},
		},
	}
	return x
}

func g(x int) {
	if x == 0 {
		return
	}
	g(x-1)
}

func main() {
	x := f()
	g(100) // smash temporaries left over on stack
	if x.T2.T3.T4[0].X != 5 {
		println("BUG", x.T2.T3.T4[0].X)
	}
}
