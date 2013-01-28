// +build amd64
// errorcheck

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2444
// Issue 4666: issue with arrays of exactly 4GB.

package main

func main() { // ERROR "stack frame too large"
	var arr [1000200030]int32
	arr_bkup := arr
	_ = arr_bkup
}

func F() { // ERROR "stack frame too large"
	var arr [1 << 30]int32
	_ = arr[42]
}
