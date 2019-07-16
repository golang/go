// run

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Used to be miscompiled by gccgo, due to a bug in handling
// initialization ordering.

package main

func F(a ...interface{}) interface{} {
	s := 0
	for _, v := range a {
		s += v.(int)
	}
	return s
}

var V1 = F(V10, V4, V3, V11)

var V2 = F(V1)

var V3 = F(1)

var V4 = F(2)

var V5 = F(3)

var V6 = F(4)

var V7 = F(5)

var V8 = F(V14, V7, V3, V6, V5)

var V9 = F(V4, F(V12))

var V10 = F(V4, V9)

var V11 = F(6)

var V12 = F(V5, V3, V8)

var V13 = F(7)

var V14 = F(8)

func expect(name string, a interface{}, b int) {
	if a.(int) != b {
		panic(name)
	}
}

func main() {
	expect("V1", V1, 38)
	expect("V2", V2, 38)
	expect("V3", V3, 1)
	expect("V4", V4, 2)
	expect("V5", V5, 3)
	expect("V6", V6, 4)
	expect("V7", V7, 5)
	expect("V8", V8, 21)
	expect("V9", V9, 27)
	expect("V10", V10, 29)
	expect("V11", V11, 6)
	expect("V12", V12, 25)
	expect("V13", V13, 7)
	expect("V14", V14, 8)
}
