// errorcheck -newparser=0

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(mdempsky): Update for new parser or delete.
// Like go/parser, the new parser doesn't specially recognize
// send statements misused in an expression context.

package main

var c chan int
var v int

func main() {
	if c <- v { // ERROR "used as value"
	}
}

var _ = c <- v // ERROR "used as value"
