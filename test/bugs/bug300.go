// errchk $G $D/$F.go

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T struct {
	x, y *T
}

func main() {
	// legal composite literals
	_ = struct{}{}
	_ = [42]int{}
	_ = [...]int{}
	_ = []int{}
	_ = map[int]int{}
	_ = T{}

	// illegal composite literals: parentheses not allowed around literal type
	_ = (struct{}){}	// ERROR "xxx"
	_ = ([42]int){}		// ERROR "xxx"
	_ = ([...]int){}	// ERROR "xxx"
	_ = ([]int){}		// ERROR "xxx"
	_ = (map[int]int){}	// ERROR "xxx"
	_ = (T){}		// ERROR "xxx"
}
