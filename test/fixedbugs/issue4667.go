// run

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"testing"
)

var globl *int

func G() {
	F()
}

func F() {
	var x int
	globl = &x
}

func main() {
	nf := testing.AllocsPerRun(100, F)
	ng := testing.AllocsPerRun(100, G)
	if int(nf) != 1 {
		fmt.Printf("AllocsPerRun(100, F) = %v, want 1\n", nf)
		os.Exit(1)
	}
	if int(ng) != 1 {
		fmt.Printf("AllocsPerRun(100, G) = %v, want 1\n", ng)
		os.Exit(1)
	}
}
