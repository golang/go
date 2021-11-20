// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

var x any
var y interface{}

var _ = &x == &y // assert x and y have identical types

func main() {
	fmt.Printf("%T\n%T\n", &x, &y)
}
