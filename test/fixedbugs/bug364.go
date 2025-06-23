// run

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

var s string

func accum(args ...interface{}) {
	s += fmt.Sprintln(args...)
}

func f(){
	v := 0.0
	for i := 0; i < 3; i++ {
		v += 0.1
		defer accum(v)
	}
}

func main() {
	f()
	if s != "0.30000000000000004\n0.2\n0.1\n" {
		println("BUG: defer")
		print(s)
	}
}
