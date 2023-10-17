// errorcheck

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f[T interface{ ~[]P }, P any](t T) { // ERROR "instantiation cycle"
	if t == nil {
		return
	}
	f[[]T, T]([]T{t})
}

func main() {
	f[[]int](nil)
}
