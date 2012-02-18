// compile

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is a test case for issue 788.

package main

func main() {
	var a [1]complex64

	t := a[0]
	_ = real(t) // this works

	_ = real(a[0]) // this doesn't
}

// bug275.go:17: internal compiler error: subnode not addable
