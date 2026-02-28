// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f(j int) {
loop:
	switch j {
	case 1:
		break loop
	default:
		println(j)
	}
}

func main() {
loop:
	for j := 0; j < 5; j++ {
		f(j)
		if j == 3 {
			break loop
		}
	}
}
