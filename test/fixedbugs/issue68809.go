// run

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	cnt := 0
	for i := 1; i <= 11; i++ {
		if i-6 > 4 {
			cnt++
		}
	}
	if cnt != 1 {
		panic("bad")
	}
}
