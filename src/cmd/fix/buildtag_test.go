// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(buildtagTests, buildtag)
}

var buildtagTests = []testCase{
	{
		Name:    "buildtag.oldGo",
		Version: "go1.10",
		In: `//go:build yes
// +build yes

package main
`,
	},
	{
		Name:    "buildtag.new",
		Version: "go1.99",
		In: `//go:build yes
// +build yes

package main
`,
		Out: `//go:build yes

package main
`,
	},
}
