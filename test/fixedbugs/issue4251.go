// errorcheck

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4251: slice with inverted range is an error.

package p

func F1(s []byte) []byte {
	return s[2:1]		// ERROR "invalid slice index|inverted slice range"
}

func F2(a [10]byte) []byte {
	return a[2:1]		// ERROR "invalid slice index|inverted slice range"
}

func F3(s string) string {
	return s[2:1]		// ERROR "invalid slice index|inverted slice range"
}
