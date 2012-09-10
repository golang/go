// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"strconv"
)

// These functions are implemented in bug453.s
func bug453a() float64
func bug453b() float64

func main() {
	if v := bug453a(); v != -1 {
		panic("a: bad result, want -1, got " + strconv.FormatFloat(v, 'f', -1, 64))
	}
	if v := bug453b(); v != 1 {
		panic("b: bad result, want 1, got " + strconv.FormatFloat(v, 'f', -1, 64))
	}
}
