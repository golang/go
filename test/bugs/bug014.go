// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var c00 uint8 = '\0';  // three octal required; should not compile
	var c01 uint8 = '\07';  // three octal required; should not compile
	var cx0 uint8 = '\x0';  // two hex required; should not compile
	var cx1 uint8 = '\x';  // two hex required; REALLY should not compile
}
