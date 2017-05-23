// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 13129: used to output error about C.unsignedshort with CC=clang

package main

import "C"

func main() {
	var x C.ushort
	x = int(0) // ERROR HERE
}
