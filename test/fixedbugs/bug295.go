// run

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import . "testing"  // defines file-level T

type _ B // make use of package "testing" (but don't refer to T)

type S struct {
	T int
}

func main() {
	_ = &S{T: 1}	// should work
}
