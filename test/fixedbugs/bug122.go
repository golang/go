// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	// should allow at most 2 sizes
	a := make([]int, 10, 20, 30, 40); // ERROR "too many|expects 2 or 3 arguments; found 5"
	_ = a
}
