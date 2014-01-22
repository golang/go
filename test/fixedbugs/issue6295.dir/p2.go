// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"./p0"
	"./p1"
)

var (
	_ p0.T0 = p0.S0{}
	_ p0.T0 = p1.S1{}
	_ p0.T0 = p1.NewT0()
	_ p0.T0 = p1.NewT1() // same as p1.S1{}
)

func main() {}
