// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var x = [4]int32{-0x7fffffff, 0x7fffffff, 2, 4}

func main() {
	if x[0] > x[1] {
		panic("fail 1")
	}
	if x[2]&x[3] < 0 {
		panic("fail 2") // Fails here
	}
}
