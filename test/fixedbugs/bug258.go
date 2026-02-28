// run

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "math"

func f() float64 {
	math.Pow(2, 2)
	return 1
}

func main() {
	for i := 0; i < 10; i++ {
		// 386 float register bug used to load constant before call
		if -5 < f() {
		} else {
			println("BUG 1")
			return
		}
		if f() > -7 {
		} else {
			println("BUG 2")
		}
		
		if math.Pow(2, 3) != 8 {
			println("BUG 3")
		}
	}
}
