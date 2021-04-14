// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	s := []int{0, 1, 2}
	i := 1
	for i > 0 && s[i] != 2 {
		i++
	}
	if i != 2 {
		panic("loop didn't run")
	}
}
