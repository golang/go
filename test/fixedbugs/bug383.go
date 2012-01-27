// errchk $G $D/$F.go

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2520

package main
func main() {
	if 2e9 { }      // ERROR "2e.09|expected bool"
	if 3.14+1i { }  // ERROR "3.14 . 1i|expected bool"
}
