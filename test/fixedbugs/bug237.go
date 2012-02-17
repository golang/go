// run

// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

var indent uint = 10
func main() {
	const dots = ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . " +
		". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . "
	const n = uint(len(dots))
	i := 2 * indent
	var s string
	for ; i > n; i -= n {
		s += fmt.Sprint(dots)
	}
	s += dots[0:i]
	if s != ". . . . . . . . . . " {
		panic(s)
	}
}
