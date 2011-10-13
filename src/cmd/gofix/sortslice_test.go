// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(sortsliceTests, sortslice)
}

var sortsliceTests = []testCase{
	{
		Name: "sortslice.0",
		In: `package main

import (
	"sort"
)

var _ = sort.Float64Array
var _ = sort.IntArray
var _ = sort.StringArray
`,
		Out: `package main

import (
	"sort"
)

var _ = sort.Float64Slice
var _ = sort.IntSlice
var _ = sort.StringSlice
`,
	},
}
