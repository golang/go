// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"time"
)

func main() {
	growstack() // Use stack early to prevent growth during test, which confuses gdb
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

	for x := 0; x <= 1; x++ { // From delve scopetest.go
		a := y
		f1(a)
		{
			b := 0
			f2(b)
			if gretbool() {
				c := 0
				f3(c)
			} else {
				c := 1.1
				f4(int(c))
			}
			f5(b)
		}
		f6(a)
	}

	{ // From delve testnextprog.go
		var (
			j = id(1)
			f = id(2)
		)
		for i := 0; i <= 5; i++ {
			j += j * (j ^ 3) / 100
			if i == f {
				fmt.Println("foo")
				break
			}
			sleepytime()
		}
		helloworld()
	}
}

func sleepytime() {
	time.Sleep(5 * time.Millisecond)
}

func helloworld() {
	fmt.Println("Hello, World!")
}

//go:noinline
func f1(x int) {}

//go:noinline
func f2(x int) {}

//go:noinline
func f3(x int) {}

//go:noinline
func f4(x int) {}

//go:noinline
func f5(x int) {}

//go:noinline
func f6(x int) {}

var boolvar = true

func gretbool() bool {
	x := boolvar
	boolvar = !boolvar
	return x
}

var sink string

//go:noinline
func growstack() {
	sink = fmt.Sprintf("%#v,%#v,%#v", 1, true, "cat")
}
