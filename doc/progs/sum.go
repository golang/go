// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

func sum(a []int) int { // returns an int
	s := 0
	for i := 0; i < len(a); i++ {
		s += a[i]
	}
	return s
}

func main() {
	x := [3]int{1, 2, 3}
	s := sum(x[:]) // a slice of the array is passed to sum
	fmt.Print(s, "\n")
}
