// errorcheck

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// 6g accepts the program below even though it is syntactically incorrect:
// Each statement in the list of statements for each case clause must be
// terminated with a semicolon. No semicolon is present for the labeled
// statements and because the last token is a colon ":", no semicolon is
// inserted automatically.
//
// Both gccgo and gofmt correctly refuse this program as is and accept it
// when the semicolons are present.

// This is a test case for issue 777 ( http://golang.org/issue/777 ).

package main

func main() {
	switch 0 {
	case 0:
		L0:  // ERROR "statement"
	case 1:
		L1:  // ERROR "statement"
	default:
		     // correct since no semicolon is required before a '}'
		goto L2
		L2:
	}
}
