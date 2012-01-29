// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(sorthelpersTests, sorthelpers)
}

var sorthelpersTests = []testCase{
	{
		Name: "sortslice.0",
		In: `package main

import (
	"sort"
)

func main() {
	var s []string
	sort.SortStrings(s)
	var i []ints
	sort.SortInts(i)
	var f []float64
	sort.SortFloat64s(f)
}
`,
		Out: `package main

import (
	"sort"
)

func main() {
	var s []string
	sort.Strings(s)
	var i []ints
	sort.Ints(i)
	var f []float64
	sort.Float64s(f)
}
`,
	},
}
