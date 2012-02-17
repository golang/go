// errorcheck

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f(args ...int) {
	g(args)
}

func g(args ...interface{}) {
	f(args)	// ERROR "cannot use|incompatible"
}
