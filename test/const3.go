// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

type T int

func (t T) String() string { return fmt.Sprintf("T%d", int(t)) }

const (
	A T = 1 << (1 << iota)
	B
	C
	D
	E
)

func main() {
	s := fmt.Sprintf("%v %v %v %v %v", A, B, C, D, E)
	if s != "T2 T4 T16 T256 T65536" {
		println("type info didn't propagate in const: got", s)
		panic("fail")
	}
	x := uint(5)
	y := float64(uint64(1)<<x)	// used to fail to compile
	if y != 32 {
		println("wrong y", y)
		panic("fail")
	}
}
