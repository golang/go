// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(mapdeleteTests, mapdelete)
}

var mapdeleteTests = []testCase{
	{
		Name: "mapdelete.0",
		In: `package main

func f() {
	m[x] = 0, false
	m[x] = g(), false
	m[x] = 1
	delete(m, x)
	m[x] = 0, b
}

func g(false bool) {
	m[x] = 0, false
}
`,
		Out: `package main

func f() {
	delete(m, x)
	m[x] = g(), false
	m[x] = 1
	delete(m, x)
	m[x] = 0, b
}

func g(false bool) {
	m[x] = 0, false
}
`,
	},
}
