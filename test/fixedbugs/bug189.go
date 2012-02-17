// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type S struct {
	a, b int
}

func main() {
	s1 := S{a: 7};	// ok - field is named
	s3 := S{7, 11};	// ok - all fields have values
	s2 := S{7};	// ERROR "too few"
	_, _, _ = s1, s3, s2;
}
