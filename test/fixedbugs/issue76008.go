// run

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "runtime"

func main() {
	shouldPanic(func() {
		g = any(func() {}) == any(func() {})
	})
	shouldPanic(func() {
		g = any(map[int]int{}) == any(map[int]int{})
	})
	shouldPanic(func() {
		g = any([]int{}) == any([]int{})
	})
}

var g bool

func shouldPanic(f func()) {
	defer func() {
		err := recover()
		if err == nil {
			_, _, line, _ := runtime.Caller(2)
			println("did not panic at line", line+1)
		}
	}()

	f()
}
