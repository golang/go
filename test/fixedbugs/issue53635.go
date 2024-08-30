// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	f[int]()
}

func f[T any]() {
	switch []T(nil) {
	case nil:
	default:
		panic("FAIL")
	}

	switch (func() T)(nil) {
	case nil:
	default:
		panic("FAIL")
	}

	switch (map[int]T)(nil) {
	case nil:
	default:
		panic("FAIL")
	}
}
