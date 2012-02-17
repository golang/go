// run

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2206.  Incorrect sign extension of div arguments.

package main

func five(x int64) {
	if x != 5 {
		panic(x)
	}
}

func main() {
       // 5
       five(int64(5 / (5 / 3)))

       // 5
       five(int64(byte(5) / (byte(5) / byte(3))))

       // 5
       var a, b byte = 5, 3
       five(int64(a / (a / b)))
       
       // integer divide by zero in golang.org sandbox
       // 0 on windows/amd64
       x := [3]byte{2, 3, 5}
       five(int64(x[2] / (x[2] / x[1])))

       // integer divide by zero in golang.org sandbox
       // crash on windows/amd64
       y := x[1:3]
       five(int64(y[1] / (y[1] / y[0])))
}