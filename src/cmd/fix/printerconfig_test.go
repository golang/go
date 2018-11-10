// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(printerconfigTests, printerconfig)
}

var printerconfigTests = []testCase{
	{
		Name: "printerconfig.0",
		In: `package main

import "go/printer"

func f() printer.Config {
	b := printer.Config{0, 8}
	c := &printer.Config{0}
	d := &printer.Config{Tabwidth: 8, Mode: 0}
	return printer.Config{0, 8}
}
`,
		Out: `package main

import "go/printer"

func f() printer.Config {
	b := printer.Config{Mode: 0, Tabwidth: 8}
	c := &printer.Config{Mode: 0}
	d := &printer.Config{Tabwidth: 8, Mode: 0}
	return printer.Config{Mode: 0, Tabwidth: 8}
}
`,
	},
}
