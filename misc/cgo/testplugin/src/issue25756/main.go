// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Run the game of life in C using Go for parallelization.

package main

import (
	"flag"
	"fmt"
	"plugin"
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

	p, err := plugin.Open("life.so")
	if err != nil {
		panic(err)
	}
	f, err := p.Lookup("Run")
	if err != nil {
		panic(err)
	}
	f.(func(int, int, int, []int32))(*gen, *dim, *dim, a[:])

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
