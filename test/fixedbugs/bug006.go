// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "os"

const (
	x float64 = iota
	g float64 = 4.5 * iota
)

func main() {
	if g == 0.0 {
		print("zero\n")
	}
	if g != 4.5 {
		print(" fail\n")
		os.Exit(1)
	}
}
