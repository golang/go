// errorcheck -0 -m -l
//go:build goexperiment.unified
// +build goexperiment.unified

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

var sink interface{}

func dotTypeEscape2() { // #13805, #15796
	{
		i := 0
		j := 0
		var ok bool
		var x interface{} = i // ERROR "i does not escape"
		var y interface{} = j // ERROR "j does not escape"

		sink = x.(int)         // ERROR "x.\(int\) escapes to heap"
		sink, *(&ok) = y.(int) // ERROR "autotmp_.* escapes to heap"
	}
}
