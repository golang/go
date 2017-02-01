// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

const (
	_ = unsafe.Sizeof(func() int {
		const (
			_ = 1
			_
			_
		)
		return 0
	}())

	y = iota
)

func main() {
	if y != 1 {
		panic(y)
	}
}
