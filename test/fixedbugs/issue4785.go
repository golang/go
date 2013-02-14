// run

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 4785: used to fail to compile

package main

func t(x, y interface{}) interface{} {
	return x.(float64) > y.(float64)
}

func main() {
	v := t(1.0, 2.0)
	if v != false {
		panic("bad comparison")
	}
}
