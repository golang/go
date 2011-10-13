// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(filepathTests, filepathFunc)
}

var filepathTests = []testCase{
	{
		Name: "filepath.0",
		In: `package main

import (
	"path/filepath"
)

var _ = filepath.SeparatorString
var _ = filepath.ListSeparatorString
`,
		Out: `package main

import (
	"path/filepath"
)

var _ = string(filepath.Separator)
var _ = string(filepath.ListSeparator)
`,
	},
}
