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

type T struct { int }

var globl *T

func F() {
	t := &T{}
	globl = t
}

func G() {
	t := &T{}
	_ = t
}

func main() {
	nf := testing.AllocsPerRun(100, F)
	ng := testing.AllocsPerRun(100, G)
	if int(nf) != 1 {
		fmt.Printf("AllocsPerRun(100, F) = %v, want 1\n", nf)
		os.Exit(1)
	}
	if int(ng) != 0 {
		fmt.Printf("AllocsPerRun(100, G) = %v, want 0\n", ng)
		os.Exit(1)
	}
}
