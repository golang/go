// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test for loops.

package main

func assertequal(is, shouldbe int, msg string) {
	if is != shouldbe {
		print("assertion fail", msg, "\n")
		panic(1)
	}
}

func main() {
	var i, sum int

	i = 0
	for {
		i = i + 1
		if i > 5 {
			break
		}
	}
	assertequal(i, 6, "break")

	sum = 0
	for i := 0; i <= 10; i++ {
		sum = sum + i
	}
	assertequal(sum, 55, "all three")

	sum = 0
	for i := 0; i <= 10; {
		sum = sum + i
		i++
	}
	assertequal(sum, 55, "only two")

	sum = 0
	for sum < 100 {
		sum = sum + 9
	}
	assertequal(sum, 99+9, "only one")

	sum = 0
	for i := 0; i <= 10; i++ {
		if i%2 == 0 {
			continue
		}
		sum = sum + i
	}
	assertequal(sum, 1+3+5+7+9, "continue")

	i = 0
	for i = range [5]struct{}{} {
	}
	assertequal(i, 4, " incorrect index value after range loop")

	i = 0
	var a1 [5]struct{}
	for i = range a1 {
		a1[i] = struct{}{}
	}
	assertequal(i, 4, " incorrect index value after array with zero size elem range clear")

	i = 0
	var a2 [5]int
	for i = range a2 {
		a2[i] = 0
	}
	assertequal(i, 4, " incorrect index value after array range clear")
}
