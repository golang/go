// compile

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 11790: Incorrect error following named pointer dereference on field

package main

import "fmt"

type T0 struct {
	x int
}

func (*T0) M0() {
	fmt.Println("M0")
}

type T2 struct {
	*T0
}

type Q *T2

func main() {
	// If run, expected output is
	// 42
	// M0
	t0 := T0{42}
	t2 := T2{&t0}
	var q Q = &t2
	fmt.Println(q.x) // Comment out either this line or the next line and the program works
	(*q).T0.M0()
}
