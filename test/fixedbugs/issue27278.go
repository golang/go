// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 27278: dead auto elim deletes an auto and its
// initialization, but it is live because of a nil check.

package main

type T struct {
	_ [3]string
	T2
}

func (t *T) M() []string {
	return t.T2.M()
}

type T2 struct {
	T3
}

func (t *T2) M() []string {
	return t.T3.M()
}

type T3 struct {
	a string
}

func (t *T3) M() []string {
	return []string{}
}

func main() {
	poison()
	f()
}

//go:noinline
func f() {
	(&T{}).M()
	grow(10000)
}

// grow stack, triggers stack copy
func grow(n int) {
	if n == 0 {
		return
	}
	grow(n-1)
}

// put some junk on stack, which cannot be valid address
//go:noinline
func poison() {
	x := [10]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	g = x
}

var g [10]int
