// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(stringssplitTests, stringssplit)
}

var stringssplitTests = []testCase{
	{
		Name: "stringssplit.0",
		In: `package main

import (
	"bytes"
	"strings"
)

func f() {
	bytes.Split(a, b, c)
	bytes.Split(a, b, -1)
	bytes.SplitAfter(a, b, c)
	bytes.SplitAfter(a, b, -1)
	strings.Split(a, b, c)
	strings.Split(a, b, -1)
	strings.SplitAfter(a, b, c)
	strings.SplitAfter(a, b, -1)
}
`,
		Out: `package main

import (
	"bytes"
	"strings"
)

func f() {
	bytes.SplitN(a, b, c)
	bytes.Split(a, b)
	bytes.SplitAfterN(a, b, c)
	bytes.SplitAfter(a, b)
	strings.SplitN(a, b, c)
	strings.Split(a, b)
	strings.SplitAfterN(a, b, c)
	strings.SplitAfter(a, b)
}
`,
	},
}
