// errorcheck -0 -m -l

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test, using compiler diagnostic flags, that the escape analysis is working.
// Compiles but does not run.  Inlining is disabled.
// Registerization is disabled too (-N), which should
// have no effect on escape analysis.

package main

import "fmt"

func main() {
	// Just run test over and over again. This main func is just for
	// convenience; if test were the main func, we could also trigger
	// the panic just by running the program over and over again
	// (sometimes it takes 1 time, sometimes it takes ~4,000+).
	for iter := 0; ; iter++ {
		if iter%50 == 0 {
			fmt.Println(iter) // ERROR "iter escapes to heap$" "... argument does not escape$"
		}
		test1(iter)
		test2(iter)
		test3(iter)
		test4(iter)
		test5(iter)
		test6(iter)
	}
}

func test1(iter int) {

	const maxI = 500
	m := make(map[int][]int) // ERROR "make\(map\[int\]\[\]int\) escapes to heap$"

	// The panic seems to be triggered when m is modified inside a
	// closure that is both recursively called and reassigned to in a
	// loop.

	// Cause of bug -- escape of closure failed to escape (shared) data structures
	// of map.  Assign to fn declared outside of loop triggers escape of closure.
	// Heap -> stack pointer eventually causes badness when stack reallocation
	// occurs.

	var fn func() // ERROR "moved to heap: fn$"
	i := 0        // ERROR "moved to heap: i$"
	for ; i < maxI; i++ {
		// var fn func() // this makes it work, because fn stays off heap
		j := 0        // ERROR "moved to heap: j$"
		fn = func() { // ERROR "func literal escapes to heap$"
			m[i] = append(m[i], 0)
			if j < 25 {
				j++
				fn()
			}
		}
		fn()
	}

	if len(m) != maxI {
		panic(fmt.Sprintf("iter %d: maxI = %d, len(m) = %d", iter, maxI, len(m))) // ERROR "iter escapes to heap$" "len\(m\) escapes to heap$" "500 escapes to heap$" "... argument does not escape$" "fmt.Sprintf\(.*\) escapes to heap"
	}
}

func test2(iter int) {

	const maxI = 500
	m := make(map[int][]int) // ERROR "make\(map\[int\]\[\]int\) does not escape$"

	// var fn func()
	for i := 0; i < maxI; i++ {
		var fn func() // this makes it work, because fn stays off heap
		j := 0
		fn = func() { // ERROR "func literal does not escape$"
			m[i] = append(m[i], 0)
			if j < 25 {
				j++
				fn()
			}
		}
		fn()
	}

	if len(m) != maxI {
		panic(fmt.Sprintf("iter %d: maxI = %d, len(m) = %d", iter, maxI, len(m))) // ERROR "iter escapes to heap$" "len\(m\) escapes to heap$" "500 escapes to heap$" "... argument does not escape$" "fmt.Sprintf\(.*\) escapes to heap"
	}
}

func test3(iter int) {

	const maxI = 500
	var x int // ERROR "moved to heap: x$"
	m := &x

	var fn func() // ERROR "moved to heap: fn$"
	for i := 0; i < maxI; i++ {
		// var fn func() // this makes it work, because fn stays off heap
		j := 0        // ERROR "moved to heap: j$"
		fn = func() { // ERROR "func literal escapes to heap$"
			if j < 100 {
				j++
				fn()
			} else {
				*m = *m + 1
			}
		}
		fn()
	}

	if *m != maxI {
		panic(fmt.Sprintf("iter %d: maxI = %d, *m = %d", iter, maxI, *m)) // ERROR "\*m escapes to heap$" "iter escapes to heap$" "500 escapes to heap$" "... argument does not escape$" "fmt.Sprintf\(.*\) escapes to heap"
	}
}

func test4(iter int) {

	const maxI = 500
	var x int
	m := &x

	// var fn func()
	for i := 0; i < maxI; i++ {
		var fn func() // this makes it work, because fn stays off heap
		j := 0
		fn = func() { // ERROR "func literal does not escape$"
			if j < 100 {
				j++
				fn()
			} else {
				*m = *m + 1
			}
		}
		fn()
	}

	if *m != maxI {
		panic(fmt.Sprintf("iter %d: maxI = %d, *m = %d", iter, maxI, *m)) // ERROR "\*m escapes to heap$" "iter escapes to heap$" "500 escapes to heap$" "... argument does not escape$" "fmt.Sprintf\(.*\) escapes to heap"
	}
}

type str struct {
	m *int
}

func recur1(j int, s *str) { // ERROR "s does not escape"
	if j < 100 {
		j++
		recur1(j, s)
	} else {
		*s.m++
	}
}

func test5(iter int) {

	const maxI = 500
	var x int // ERROR "moved to heap: x$"
	m := &x

	var fn *str
	for i := 0; i < maxI; i++ {
		// var fn *str // this makes it work, because fn stays off heap
		fn = &str{m} // ERROR "&str{...} escapes to heap"
		recur1(0, fn)
	}

	if *m != maxI {
		panic(fmt.Sprintf("iter %d: maxI = %d, *m = %d", iter, maxI, *m)) // ERROR "\*m escapes to heap$" "iter escapes to heap$" "500 escapes to heap$" "... argument does not escape$" "fmt.Sprintf\(.*\) escapes to heap"
	}
}

func test6(iter int) {

	const maxI = 500
	var x int
	m := &x

	// var fn *str
	for i := 0; i < maxI; i++ {
		var fn *str  // this makes it work, because fn stays off heap
		fn = &str{m} // ERROR "&str{...} does not escape"
		recur1(0, fn)
	}

	if *m != maxI {
		panic(fmt.Sprintf("iter %d: maxI = %d, *m = %d", iter, maxI, *m)) // ERROR "\*m escapes to heap$" "iter escapes to heap$" "500 escapes to heap$" "... argument does not escape$" "fmt.Sprintf\(.*\) escapes to heap"
	}
}
