// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Gccgo used to miscompile passing a global variable with a
// zero-sized type to a function.

package main

type T struct {
	field s
}

type s struct{}

var X T

func F(_ T, c interface{}) int {
	return len(c.(string))
}

func main() {
	if v := F(X, "hi"); v != 2 {
		panic(v)
	}
}
