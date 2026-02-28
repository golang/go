// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

// test expects f to panic, but not to run out of memory,
// which is a non-panic fatal error.  OOM results from failure
// to properly check negative limit.
func test(f func()) {
	defer func() {
		r := recover()
		if r == nil {
			panic("panic wasn't recoverable")
		}
	}()
	f()
}

//go:noinline
func id(x int) int {
	return x
}

func main() {
	test(foo)
	test(bar)
}

func foo() {
	b := make([]byte, 0)
	b = append(b, 1)
	id(len(b))
	id(len(b) - 2)
	s := string(b[1 : len(b)-2])
	fmt.Println(s)
}

func bar() {
	b := make([]byte, 1)
	b = append(b, 1)
	i := id(-1)
	if i < len(b) { // establish value is not too large.
		s := string(b[1:i]) // should check for negative also.
		fmt.Println(s)
	}
}
