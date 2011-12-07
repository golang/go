// errchk $G $D/$F.go

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2276.

// Check that the error messages says 
//	bug378.go:19: unsafe.Alignof(0) not used
// and not
//	bug378.go:19: 4 not used

package main

import "unsafe"

func main() {
	unsafe.Alignof(0) // ERROR "unsafe\.Alignof|value computed is not used"
}
