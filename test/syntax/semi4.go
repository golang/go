// errorcheck

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	for x		// GCCGO_ERROR "undefined"
	{		// ERROR "missing .*{.* after for clause"
		z	// GCCGO_ERROR "undefined"


