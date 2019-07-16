// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

func main() {
	var i, k int
	var r string
outer:
	for k = 0; k < 2; k++ {
		r += fmt.Sprintln("outer loop top k", k)
		if k != 0 {
			panic("k not zero")
		} // inner loop breaks this one every time
		for i = 0; i < 2; i++ {
			if i != 0 {
				panic("i not zero")
			} // loop breaks every time
			r += fmt.Sprintln("inner loop top i", i)
			if true {
				r += "do break\n"
				break outer
			}
		}
	}
	r += "broke\n"
	expect := `outer loop top k 0
inner loop top i 0
do break
broke
`
	if r != expect {
		panic(r)
	}
}
