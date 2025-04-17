// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:noinline
func f[T any]() {
	x := 5
	g := func() int { return x }
	g()
}

func main() {
	f[int]()
}
