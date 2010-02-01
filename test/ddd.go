// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func sum(args ...int) int {
	s := 0
	for _, v := range args {
		s += v
	}
	return s
}

func sumA(args []int) int {
	s := 0
	for _, v := range args {
		s += v
	}
	return s
}

func sum2(args ...int) int { return 2 * sum(args) }

func sum3(args ...int) int { return 3 * sumA(args) }

func intersum(args ...interface{}) int {
	s := 0
	for _, v := range args {
		s += v.(int)
	}
	return s
}

type T []T

func ln(args ...T) int { return len(args) }

func ln2(args ...T) int { return 2 * ln(args) }

func main() {
	if x := sum(1, 2, 3); x != 6 {
		panicln("sum 6", x)
	}
	if x := sum(); x != 0 {
		panicln("sum 0", x)
	}
	if x := sum(10); x != 10 {
		panicln("sum 10", x)
	}
	if x := sum(1, 8); x != 9 {
		panicln("sum 9", x)
	}
	if x := sum2(1, 2, 3); x != 2*6 {
		panicln("sum 6", x)
	}
	if x := sum2(); x != 2*0 {
		panicln("sum 0", x)
	}
	if x := sum2(10); x != 2*10 {
		panicln("sum 10", x)
	}
	if x := sum2(1, 8); x != 2*9 {
		panicln("sum 9", x)
	}
	if x := sum3(1, 2, 3); x != 3*6 {
		panicln("sum 6", x)
	}
	if x := sum3(); x != 3*0 {
		panicln("sum 0", x)
	}
	if x := sum3(10); x != 3*10 {
		panicln("sum 10", x)
	}
	if x := sum3(1, 8); x != 3*9 {
		panicln("sum 9", x)
	}
	if x := intersum(1, 2, 3); x != 6 {
		panicln("intersum 6", x)
	}
	if x := intersum(); x != 0 {
		panicln("intersum 0", x)
	}
	if x := intersum(10); x != 10 {
		panicln("intersum 10", x)
	}
	if x := intersum(1, 8); x != 9 {
		panicln("intersum 9", x)
	}

	if x := ln(nil, nil, nil); x != 3 {
		panicln("ln 3", x)
	}
	if x := ln([]T{}); x != 1 {
		panicln("ln 1", x)
	}
	if x := ln2(nil, nil, nil); x != 2*3 {
		panicln("ln2 3", x)
	}
	if x := ln2([]T{}); x != 2*1 {
		panicln("ln2 1", x)
	}
}
