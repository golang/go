// run

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2821
package main

type matrix struct {
	e []int
}

func (a matrix) equal() bool {
	for _ = range a.e {
	}
	return true
}

func main() {
	var a matrix
	var i interface{}
	i = true && a.equal()
	_ = i
}
