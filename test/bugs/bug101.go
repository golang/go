// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var a = []int { 1, 2, 3 }

func main() {
	if len(a) != 3 { panic("array len") }
	// print(a[0], " ", a[1], " ", a[2], "\n")
	if a[0] != 1 || a[1] != 2 || a[2] != 3 { panic("array contents") }
}
