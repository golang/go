// run

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4173

package main

func main() {
	var arr *[10]int
	s := 0
	for i, _ := range arr {
		// used to panic trying to access arr[i]
		s += i
	}
	if s != 45 {
		println("BUG")
	}
}
