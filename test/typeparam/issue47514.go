// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that closures inside a generic function are not exported,
// even though not themselves generic.

package main

func Do[T any]() {
	_ = func() string {
		return ""
	}
}

func main() {
	Do[int]()
}
