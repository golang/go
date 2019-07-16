// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	m := map[interface{}]int{}
	k := []int{}

	mustPanic(func() {
		_ = m[k]
	})
	mustPanic(func() {
		_, _ = m[k]
	})
	mustPanic(func() {
		delete(m, k)
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
