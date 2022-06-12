// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
)

type I int

func (x *I) method() int {
	return int(*x)
}

func main() {
	sum := 0
	var is []func() int
	for i := I(0); int(i) < 10; i++ {
		for j := 0; j < 10; j++ {
			if int(i) == j { // 10 skips
				continue
			}
			sum++
		}
		if i&1 == 0 {
			is = append(is, i.method)
		}
	}

	bug := false
	if sum != 100-10 {
		fmt.Printf("wrong sum, expected %d, saw %d\n", 90, sum)
		bug = true
	}
	sum = 0
	for _, m := range is {
		sum += m()
	}
	if sum != 2+4+6+8 {
		fmt.Printf("wrong sum, expected %d, saw %d\n", 20, sum)
		bug = true
	}
	if !bug {
		fmt.Printf("PASS\n")
	} else {
		os.Exit(11)
	}
}
