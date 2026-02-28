// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

type MyStruct struct {
	F0 [0]float64
	F1 byte
	F2 int16
	_  struct {
		F0 uint32
	}
}

func main() {
	p0 := MyStruct{F0: [0]float64{}, F1: byte(27), F2: int16(9887)}
	fmt.Println(p0)
}
