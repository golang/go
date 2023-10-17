// run

// Copyright 2020 The Go Authors. All rights reserved.  Use of this
// source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package main

import "fmt"

func main() {
	const N = 1024
	var a [N]int
	x := cap(append(a[:N-1:N], 9, 9))
	y := cap(append(a[:N:N], 9))
	if x != y {
		panic(fmt.Sprintf("different capacity on append: %d vs %d", x, y))
	}
}
