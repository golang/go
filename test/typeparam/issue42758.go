// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func F[T, U int]() interface{} {
	switch interface{}(nil) {
	case int(0), T(0), U(0):
	}

	return map[interface{}]int{int(0): 0, T(0): 0, U(0): 0}
}

func main() {
	F[int, int]()
}
