// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T[_ any] struct{}

var m = map[interface{}]int{
	T[struct{ int }]{}: 0,
	T[struct {
		int "x"
	}]{}: 0,
}

func main() {
	if len(m) != 2 {
		panic(len(m))
	}
}
