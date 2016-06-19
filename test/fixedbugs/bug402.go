// run

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

var a = []int64{
	0.0005 * 1e9,
	0.001 * 1e9,
	0.005 * 1e9,
	0.01 * 1e9,
	0.05 * 1e9,
	0.1 * 1e9,
	0.5 * 1e9,
	1 * 1e9,
	5 * 1e9,
}

func main() {
	s := ""
	for _, v := range a {
		s += fmt.Sprint(v) + " "
	}
	if s != "500000 1000000 5000000 10000000 50000000 100000000 500000000 1000000000 5000000000 " {
		panic(s)
	}
}
