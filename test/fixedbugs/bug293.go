// run

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// https://golang.org/issue/846

package main

func x() (a int, b bool) {
	defer func(){
		a++
	}()
	a, b = y()
	return
}

func x2() (a int, b bool) {
	defer func(){
		a++
	}()
	return y()
}

func y() (int, bool) {
	return 4, false
}

func main() {
	if a, _ := x(); a != 5 {
		println("BUG", a)
	}
	if a, _ := x2(); a != 5 {
		println("BUG", a)
	}
}
