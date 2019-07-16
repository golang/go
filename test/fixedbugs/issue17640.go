// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

var results string

type TwoInts struct {
	x, y int
}

func f(x int) int { results = results + fmt.Sprintf("_%d", x); return x }

func main() {
	_ = [19]int{1: f(1), 0: f(0), 2: f(2), 6, 7}
	_ = [2]int{1: f(4), 0: f(3)}
	_ = TwoInts{y: f(6), x: f(5)}
	_ = map[int]int{f(f(9) + 1): f(8), 0: f(7), f(22): -1}
	if results != "_1_0_2_4_3_6_5_9_10_8_7_22" {
		fmt.Printf("unexpected: %s\n", results)
		panic("fail")
	}
}
