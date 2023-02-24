// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"unsafe"
)

var intGlo int

func main() {
	r := bar(&intGlo)
	fmt.Printf("r value is %d", r)
}

func bar(a *int) int {
	p := (*int)(unsafe.Add(unsafe.Pointer(a), 1*unsafe.Sizeof(int(1))))
	if *p == 10 { // BOOM
		fmt.Println("its value is 10")
	}
	return *p
}
