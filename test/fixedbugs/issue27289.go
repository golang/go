// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure we don't prove that the bounds check failure branch is unreachable.

package main

//go:noinline
func f(a []int) {
	_ = a[len(a)-1]
}

func main() {
	defer func() {
		if err := recover(); err != nil {
			return
		}
		panic("f should panic")
	}()
	f(nil)
}
