// errchk $G $D/$F.go

// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// make sure that even if a file imports runtime,
// it cannot get at the low-level runtime definitions
// known to the compiler.  for normal packages
// the compiler doesn't even record the lower case
// functions in its symbol table, but some functions
// in runtime are hard-coded into the compiler.

package main

import "runtime"

func main() {
	runtime.printbool(true)	// ERROR "unexported"
}
