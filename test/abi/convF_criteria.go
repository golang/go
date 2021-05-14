// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

type myStruct struct {
	F0 [0]struct{}
	F1 float32
}

type myStruct2 struct {
	F0 [0]struct{}
	F1 float32
	F2 [0]struct{}
}

func main() {
	x := myStruct{F1: -1.25}
	fmt.Println(x)
	x2 := myStruct2{F1: -7.97}
	fmt.Println(x2)
}
