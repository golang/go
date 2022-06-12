// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
)

func main() {
	x := f(60)
	fmt.Println(x)
	if x != 54 {
		os.Exit(11)
	}
}

var escape *int

func f(i int) int {
	a := 0
outer:
	for {
		switch {
		case i > 55:
			i--
			continue
		case i == 55:
			for j := i; j != 1; j = j / 2 {
				a++
				if j == 4 {
					escape = &j
					i--
					continue outer
				}
				if j&1 == 1 {
					j = 2 * (3*j + 1)
				}
			}
			return a
		case i < 55:
			return i
		}
	}
}
