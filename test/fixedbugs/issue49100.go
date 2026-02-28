// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f(j int) {
loop:
	for i := 0; i < 4; i++ {
		if i == 1 {
			continue loop
		}
		println(j, i)
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
