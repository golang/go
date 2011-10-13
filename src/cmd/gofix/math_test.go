// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(mathTests, math)
}

var mathTests = []testCase{
	{
		Name: "math.0",
		In: `package main

import (
	"math"
)

func f() {
	math.Fabs(1)
	math.Fdim(1)
	math.Fmax(1)
	math.Fmin(1)
	math.Fmod(1)
	math.Abs(1)
	foo.Fabs(1)
}
`,
		Out: `package main

import (
	"math"
)

func f() {
	math.Abs(1)
	math.Dim(1)
	math.Max(1)
	math.Min(1)
	math.Mod(1)
	math.Abs(1)
	foo.Fabs(1)
}
`,
	},
}
