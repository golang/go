// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func r(j int) {
loop:
	for i, c := range "goclang" {
		if i == 2 {
			continue loop
		}
		println(string(c))
	}
}

func main() {
loop:
	for j := 0; j < 4; j++ {
		r(j)
		if j == 0 {
			break loop
		}
	}
}
