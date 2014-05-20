// run

// Check conversion of constant to float32/float64 near min/max boundaries.

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

var cvt = []struct {
	val    interface{}
	binary string
}{
	{float32(-340282356779733661637539395458142568447), "-16777215p+104"},
	{float32(-340282326356119256160033759537265639424), "-16777214p+104"},
	{float32(340282326356119256160033759537265639424), "16777214p+104"},
	{float32(340282356779733661637539395458142568447), "16777215p+104"},
	{float64(-1.797693134862315807937289714053e+308), "-9007199254740991p+971"},
	{float64(-1.797693134862315708145274237317e+308), "-9007199254740991p+971"},
	{float64(-1.797693134862315608353258760581e+308), "-9007199254740990p+971"},
	{float64(1.797693134862315608353258760581e+308), "9007199254740990p+971"},
	{float64(1.797693134862315708145274237317e+308), "9007199254740991p+971"},
	{float64(1.797693134862315807937289714053e+308), "9007199254740991p+971"},
}

func main() {
	bug := false
	for i, c := range cvt {
		s := fmt.Sprintf("%b", c.val)
		if s != c.binary {
			if !bug {
				bug = true
				fmt.Println("BUG")
			}
			fmt.Printf("#%d: have %s, want %s\n", i, s, c.binary)
		}
	}
}
