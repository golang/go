// run

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f() int {
	defer func() {
		recover()
	}()
	panic("oops")
}

func g() int {	
	return 12345
}

func main() {
	g()	// leave 12345 on stack
	x := f()
	if x != 0 {
		panic(x)
	}
}
