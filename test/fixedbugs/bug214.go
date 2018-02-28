// compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Used to crash the compiler.
// https://golang.org/issue/88

package main

func main() {
	x := make(map[int]int, 10);
	x[0], x[1] = 2, 6;
}
