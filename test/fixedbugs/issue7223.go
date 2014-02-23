// errorcheck

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var bits1 uint = 10
const bits2 uint = 10

func main() {
	_ = make([]byte, 1<<bits1)
	_ = make([]byte, 1<<bits2)
	_ = make([]byte, nil) // ERROR "non-integer.*len"
	_ = make([]byte, nil, 2) // ERROR "non-integer.*len"
	_ = make([]byte, 1, nil) // ERROR "non-integer.*cap"
	_ = make([]byte, true) // ERROR "non-integer.*len"
	_ = make([]byte, "abc") // ERROR "non-integer.*len"
}
