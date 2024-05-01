// run

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var (
	v0 = initv0()
	v1 = initv1()
)

const c = "c"

func initv0() string {
	println("initv0")
	if c != "" { // have a dependency on c
		return ""
	}
	return ""
}

func initv1() string {
	println("initv1")
	return ""
}

func main() {
	// do nothing
}
