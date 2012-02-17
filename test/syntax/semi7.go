// errorcheck

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	if x { }	// GCCGO_ERROR "undefined"
	else { }	// ERROR "unexpected semicolon or newline before .?else.?"
}


