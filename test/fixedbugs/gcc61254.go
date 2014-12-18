// compile

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// PR61254: gccgo failed to compile a slice expression with missing indices.

package main

func main() {
	[][]int{}[:][0][0]++
}
