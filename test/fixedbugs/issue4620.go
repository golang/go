// run

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4620: map indexes are not evaluated before assignment of other elements

package main

import "fmt"

func main() {
	m := map[int]int{0:1}
	i := 0
	i, m[i] = 1, 2
	if m[0] != 2 {
		fmt.Println(m)
		panic("m[i] != 2")
	}
}
