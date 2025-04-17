// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 20029: make sure we zero at VARKILLs of
// ambiguously live variables.
// The ambiguously live variable here is the hiter
// for the inner range loop.

package main

import "runtime"

func f(m map[int]int) {
outer:
	for i := 0; i < 10; i++ {
		for k := range m {
			if k == 5 {
				continue outer
			}
		}
		runtime.GC()
		break
	}
	runtime.GC()
}
func main() {
	m := map[int]int{1: 2, 2: 3, 3: 4}
	f(m)
}
