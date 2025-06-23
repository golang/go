// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test to make sure RHS is evaluated before map insert is started.
// The RHS panics in all of these cases.

package main

import "fmt"

func main() {
	for i, f := range []func(map[int]int){
		f0, f1, f2, f3, f4, f5, f6, f7, f8,
	} {
		m := map[int]int{}
		func() { // wrapper to scope the defer.
			defer func() {
				recover()
			}()
			f(m) // Will panic. Shouldn't modify m.
			fmt.Printf("RHS didn't panic, case f%d\n", i)
		}()
		if len(m) != 0 {
			fmt.Printf("map insert happened, case f%d\n", i)
		}
	}

	// Append slice.
	for i, f := range []func(map[int][]int){
		fa0, fa1, fa2, fa3,
	} {
		m := map[int][]int{}
		func() { // wrapper to scope the defer.
			defer func() {
				recover()
			}()
			f(m) // Will panic. Shouldn't modify m.
			fmt.Printf("RHS didn't panic, case fa%d\n", i)
		}()
		if len(m) != 0 {
			fmt.Printf("map insert happened, case fa%d\n", i)
		}
	}
}

func f0(m map[int]int) {
	var p *int
	m[0] = *p
}

func f1(m map[int]int) {
	var p *int
	m[0] += *p
}

func f2(m map[int]int) {
	var p *int
	sink, m[0] = sink, *p
}

func f3(m map[int]int) {
	var p *chan int
	m[0], sink = <-(*p)
}

func f4(m map[int]int) {
	var p *interface{}
	m[0], sink = (*p).(int)
}

func f5(m map[int]int) {
	var p *map[int]int
	m[0], sink = (*p)[0]
}

func f6(m map[int]int) {
	var z int
	m[0] /= z
}

func f7(m map[int]int) {
	var a []int
	m[0] = a[0]
}

func f8(m map[int]int) {
	var z int
	m[0] %= z
}

func fa0(m map[int][]int) {
	var p *int
	m[0] = append(m[0], *p)
}

func fa1(m map[int][]int) {
	var p *int
	sink, m[0] = !sink, append(m[0], *p)
}

func fa2(m map[int][]int) {
	var p *int
	m[0], _ = append(m[0], 0), *p
}

func fa3(m map[int][]int) {
	// OSLICE has similar in-place-reassignment
	// optimizations as OAPPEND, but we need to make sure
	// to *not* optimize them, because we can't guarantee
	// the slice indices are within bounds.
	m[0] = m[0][:1]
}

var sink bool
