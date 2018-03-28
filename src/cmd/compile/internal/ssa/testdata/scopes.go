// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

func main() {
	test()
}

//go:noinline
func id(x int) int {
	return x
}

func test() {
	x := id(0)
	y := id(0)
	fmt.Println(x)
	for i := x; i < 3; i++ {
		x := i * i
		y += id(x) //gdb-dbg=(x,y)//gdb-opt=(x,y)
	}
	y = x + y //gdb-dbg=(x,y)//gdb-opt=(x,y)
	fmt.Println(x, y)
}
