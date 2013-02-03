// errorcheck

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T []int

func main() {
	_ = make(T, -1)       // ERROR "negative"
	_ = make(T, 0.5)       // ERROR "constant 0.5 truncated to integer"
	_ = make(T, 1.0)       // ok
	_ = make(T, 1<<63)  // ERROR "len argument too large"
	_ = make(T, 0, -1)    // ERROR "negative cap"
	_ = make(T, 10, 0) // ERROR "len larger than cap"
}
