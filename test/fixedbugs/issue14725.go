// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

func f1() (x int) {
	for {
		defer func() {
			recover()
			x = 1
		}()
		panic(nil)
	}
}

var sink *int

func f2() (x int) {
	sink = &x
	defer func() {
		recover()
		x = 1
	}()
	panic(nil)
}

func f3(b bool) (x int) {
	sink = &x
	defer func() {
		recover()
		x = 1
	}()
	if b {
		panic(nil)
	}
	return
}

func main() {
	if x := f1(); x != 1 {
		panic(fmt.Sprintf("f1 returned %d, wanted 1", x))
	}
	if x := f2(); x != 1 {
		panic(fmt.Sprintf("f2 returned %d, wanted 1", x))
	}
	if x := f3(true); x != 1 {
		panic(fmt.Sprintf("f3(true) returned %d, wanted 1", x))
	}
	if x := f3(false); x != 1 {
		panic(fmt.Sprintf("f3(false) returned %d, wanted 1", x))
	}
}
