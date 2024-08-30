// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type TestSuite struct {
	Tests []Test
}
type Test struct {
	Want interface{}
}
type Int struct {
	i int
}

func NewInt(v int) Int {
	return Int{i: v}
}

var Suites = []TestSuite{
	Dicts,
}
var Dicts = TestSuite{
	Tests: []Test{
		{
			Want: map[Int]bool{NewInt(1): true},
		},
		{
			Want: map[Int]string{
				NewInt(3): "3",
			},
		},
	},
}

func main() {
	if Suites[0].Tests[0].Want.(map[Int]bool)[NewInt(3)] {
		panic("bad")
	}
}
