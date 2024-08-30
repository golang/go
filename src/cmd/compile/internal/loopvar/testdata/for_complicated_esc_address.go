// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
)

func main() {
	ss, sa := shared(23)
	ps, pa := private(23)
	es, ea := experiment(23)

	fmt.Printf("shared s, a; private, s, a; experiment s, a = %d, %d;  %d, %d;  %d, %d\n", ss, sa, ps, pa, es, ea)

	if ss != ps || ss != es || ea != pa || sa == pa {
		os.Exit(11)
	} else {
		fmt.Println("PASS")
	}
}

func experiment(x int) (int, int) {
	sum := 0
	var is []*int
	for i := x; i != 1; i = i / 2 {
		for j := 0; j < 10; j++ {
			if i == j { // 10 skips
				continue
			}
			sum++
		}
		i = i*3 + 1
		if i&1 == 0 {
			is = append(is, &i)
			for i&2 == 0 {
				i = i >> 1
			}
		} else {
			i = i + i
		}
	}

	asum := 0
	for _, pi := range is {
		asum += *pi
	}

	return sum, asum
}

func private(x int) (int, int) {
	sum := 0
	var is []*int
	I := x
	for ; I != 1; I = I / 2 {
		i := I
		for j := 0; j < 10; j++ {
			if i == j { // 10 skips
				I = i
				continue
			}
			sum++
		}
		i = i*3 + 1
		if i&1 == 0 {
			is = append(is, &i)
			for i&2 == 0 {
				i = i >> 1
			}
		} else {
			i = i + i
		}
		I = i
	}

	asum := 0
	for _, pi := range is {
		asum += *pi
	}

	return sum, asum
}

func shared(x int) (int, int) {
	sum := 0
	var is []*int
	i := x
	for ; i != 1; i = i / 2 {
		for j := 0; j < 10; j++ {
			if i == j { // 10 skips
				continue
			}
			sum++
		}
		i = i*3 + 1
		if i&1 == 0 {
			is = append(is, &i)
			for i&2 == 0 {
				i = i >> 1
			}
		} else {
			i = i + i
		}
	}

	asum := 0
	for _, pi := range is {
		asum += *pi
	}
	return sum, asum
}
