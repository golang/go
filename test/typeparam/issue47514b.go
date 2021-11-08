// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func Do[T any](do func() (T, string)) {
	_ = func() (T, string) {
		return do()
	}
}

func main() {
	Do[int](func() (int, string) {
		return 3, "3"
	})
}
