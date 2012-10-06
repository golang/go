// run

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4197: growing a slice of zero-width elements
// panics on a division by zero.

package main

func main() {
	var x []struct{}
	x = append(x, struct{}{})
}
