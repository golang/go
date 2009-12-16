// errchk $G -e $D/$F.go

// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "testing"

func main() {
	var t testing.T
	
	// make sure error mentions that
	// ch is unexported, not just "ch not found".

	t.ch = nil	// ERROR "unexported"
	
	println(testing.anyLowercaseName("asdf"))	// ERROR "unexported"
}
