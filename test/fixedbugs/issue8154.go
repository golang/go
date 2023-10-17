// compile

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 8154: cmd/5g: ICE in walkexpr walk.c

package main

func main() {
	c := make(chan int)
	_ = [1][]func(){[]func(){func() { <-c }}}
}
