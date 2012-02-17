// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var c00 uint8 = '\0';  // ERROR "oct|char"
	var c01 uint8 = '\07';  // ERROR "oct|char"
	var cx0 uint8 = '\x0';  // ERROR "hex|char"
	var cx1 uint8 = '\x';  // ERROR "hex|char"
	_, _, _, _ = c00, c01, cx0, cx1
}
