// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 47317: ICE when calling ABI0 function via func value.

package main

func main() { F() }

func F() interface{} {
	g := G
	g(1)
	return G
}

func G(x int) [2]int
