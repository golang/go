// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var v1 = T1(1)
var v2 = T2{2}
var v3 = T3{0: 3, 1: 4}
var v4 = T4{0: 5, 1: 6}
var v5 = T5{0: 7, 1: 8}
var v6 = T2{f: 9}
var v7 = T4{f: 10}
var v8 = T5{f: 11}
var pf func(T1)

func main() {
	if v1 != 1 || v2.f != 2 || v3[0] != 3 || v3[1] != 4 ||
		v4[0] != 5 || v4[1] != 6 || v5[0] != 7 || v5[1] != 8 ||
		v6.f != 9 || v7[0] != 10 || v8[0] != 11 {
		panic("fail")
	}
}

type T1 int
type T2 struct {
	f int
}
type T3 []int
type T4 [2]int
type T5 map[int]int

const f = 0
