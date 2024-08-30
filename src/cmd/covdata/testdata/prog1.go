// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"prog/dep"
)

//go:noinline
func first() {
	println("whee")
}

//go:noinline
func second() {
	println("oy")
}

//go:noinline
func third(x int) int {
	if x != 0 {
		return 42
	}
	println("blarg")
	return 0
}

//go:noinline
func fourth() int {
	return 99
}

func main() {
	println(dep.Dep1())
	dep.PDep(2)
	if len(os.Args) > 1 {
		second()
		third(1)
	} else if len(os.Args) > 2 {
		fourth()
	} else {
		first()
		third(0)
	}
}
