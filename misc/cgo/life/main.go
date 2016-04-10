// cmpout

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build test_run

// Run the game of life in C using Go for parallelization.

package main

import (
	"."
	"flag"
	"fmt"
)

const MAXDIM = 100

var dim = flag.Int("dim", 16, "board dimensions")
var gen = flag.Int("gen", 10, "generations")

func main() {
	flag.Parse()

	var a [MAXDIM * MAXDIM]int32
	for i := 2; i < *dim; i += 8 {
		for j := 2; j < *dim-3; j += 8 {
			for y := 0; y < 3; y++ {
				a[i**dim+j+y] = 1
			}
		}
	}

	life.Run(*gen, *dim, *dim, a[:])

	for i := 0; i < *dim; i++ {
		for j := 0; j < *dim; j++ {
			if a[i**dim+j] == 0 {
				fmt.Print(" ")
			} else {
				fmt.Print("X")
			}
		}
		fmt.Print("\n")
	}
}
