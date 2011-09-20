// errchk $G $D/$F.go

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 1722.

// Check that the error messages says 
//	bug337.go:16: len("foo") not used
// and not
//	bug337.go:16: 3 not used

package main

func main() {
	len("foo")	// ERROR "len|value computed is not used"
}

