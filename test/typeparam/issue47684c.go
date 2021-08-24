// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f[G any]() func()func()int {
	return func() func()int {
		return func() int {
			return 0
		}
	}
}

func main() {
	f[int]()()()
}
