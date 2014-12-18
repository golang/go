// compile

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// PR61273: gccgo failed to compile a SendStmt in the PostStmt of a ForClause
// that involved predefined constants.

package main

func main() {
	c := make(chan bool, 1)
	for ; false; c <- false {
	}
}
