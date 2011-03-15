// errchk $G -e $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
L:
	for {
		for {
			break L2    // ERROR "L2"
			continue L2 // ERROR "L2"
		}
	}

L1:
	x := 1
	_ = x
	for {
		break L1    // ERROR "L1"
		continue L1 // ERROR "L1"
	}

	goto L
}
