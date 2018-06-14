// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test simple functions.

package main

func assertequal(is, shouldbe int, msg string) {
	if is != shouldbe {
		print("assertion fail", msg, "\n")
		panic(1)
	}
}

func f1() {
}

func f2(a int) {
}

func f3(a, b int) int {
	return a + b
}

func f4(a, b int, c float32) int {
	return (a+b)/2 + int(c)
}

func f5(a int) int {
	return 5
}

func f6(a int) (r int) {
	return 6
}

func f7(a int) (x int, y float32) {
	return 7, 7.0
}


func f8(a int) (x int, y float32) {
	return 8, 8.0
}

type T struct {
	x, y int
}

func (t *T) m10(a int, b float32) int {
	return (t.x + a) * (t.y + int(b))
}


func f9(a int) (i int, f float32) {
	i = 9
	f = 9.0
	return
}


func main() {
	f1()
	f2(1)
	r3 := f3(1, 2)
	assertequal(r3, 3, "3")
	r4 := f4(0, 2, 3.0)
	assertequal(r4, 4, "4")
	r5 := f5(1)
	assertequal(r5, 5, "5")
	r6 := f6(1)
	assertequal(r6, 6, "6")
	r7, s7 := f7(1)
	assertequal(r7, 7, "r7")
	assertequal(int(s7), 7, "s7")
	r8, s8 := f8(1)
	assertequal(r8, 8, "r8")
	assertequal(int(s8), 8, "s8")
	r9, s9 := f9(1)
	assertequal(r9, 9, "r9")
	assertequal(int(s9), 9, "s9")
	var t *T = new(T)
	t.x = 1
	t.y = 2
	r10 := t.m10(1, 3.0)
	assertequal(r10, 10, "10")
}
