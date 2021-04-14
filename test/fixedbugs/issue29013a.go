// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type TestSuite struct {
	Tests []int
}

var Suites = []TestSuite{
	Dicts,
}
var Dicts = TestSuite{
	Tests: []int{0},
}

func main() {
	if &Dicts.Tests[0] != &Suites[0].Tests[0] {
		panic("bad")
	}
}
