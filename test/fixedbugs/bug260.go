// run

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file

// Test that structures pack densely, according to the alignment of the largest field.

package main

import (
	"fmt"
	"os"
	"strconv"
)

type T1 struct {
	x uint8
}
type T2 struct {
	x uint16
}
type T4 struct {
	x uint32
}

func main() {
	report := len(os.Args) > 1
	status := 0
	var b1 [10]T1
	a0, _ := strconv.ParseUint(fmt.Sprintf("%p", &b1[0])[2:], 16, 64)
	a1, _ := strconv.ParseUint(fmt.Sprintf("%p", &b1[1])[2:], 16, 64)
	if a1 != a0+1 {
		fmt.Println("FAIL")
		if report {
			fmt.Println("alignment should be 1, is", a1-a0)
		}
		status = 1
	}
	var b2 [10]T2
	a0, _ = strconv.ParseUint(fmt.Sprintf("%p", &b2[0])[2:], 16, 64)
	a1, _ = strconv.ParseUint(fmt.Sprintf("%p", &b2[1])[2:], 16, 64)
	if a1 != a0+2 {
		if status == 0 {
			fmt.Println("FAIL")
			status = 1
		}
		if report {
			fmt.Println("alignment should be 2, is", a1-a0)
		}
	}
	var b4 [10]T4
	a0, _ = strconv.ParseUint(fmt.Sprintf("%p", &b4[0])[2:], 16, 64)
	a1, _ = strconv.ParseUint(fmt.Sprintf("%p", &b4[1])[2:], 16, 64)
	if a1 != a0+4 {
		if status == 0 {
			fmt.Println("FAIL")
			status = 1
		}
		if report {
			fmt.Println("alignment should be 4, is", a1-a0)
		}
	}
	os.Exit(status)
}
