// +build 386 arm
// errorcheck

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2444

package main
func main() {
	var arr [1000200030]int   // GC_ERROR "type .* too large"
	arr_bkup := arr
	_ = arr_bkup
}
