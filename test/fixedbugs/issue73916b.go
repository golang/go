// run

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func callRecover() {
	func() {
		if recover() != nil {
			println("recovered")
		}
	}()
}

func F() int { callRecover(); return 0 }

func main() {
	mustPanic(func() {
		defer F()
		panic("XXX")
	})
}

func mustPanic(f func()) {
	defer func() {
		r := recover()
		if r == nil {
			panic("didn't panic")
		}
	}()
	f()
}
