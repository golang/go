// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(templateTests, template)
}

var templateTests = []testCase{
	{
		Name: "template.0",
		In: `package main

import (
	"text/template"
)

func f() {
	template.ParseFile(a)
	var t template.Template
	x, y := template.ParseFile()
	template.New("x").Funcs(m).ParseFile(a) // chained method
	// Output should complain about these as functions or methods.
	var s *template.Set
	s.ParseSetFiles(a)
	template.ParseSetGlob(a)
	s.ParseTemplateFiles(a)
	template.ParseTemplateGlob(a)
	x := template.SetMust(a())
}
`,
		Out: `package main

import (
	"text/template"
)

func f() {
	template.ParseFiles(a)
	var t template.Template
	x, y := template.ParseFiles()
	template.New("x").Funcs(m).ParseFiles(a) // chained method
	// Output should complain about these as functions or methods.
	var s *template.Set
	s.ParseSetFiles(a)
	template.ParseSetGlob(a)
	s.ParseTemplateFiles(a)
	template.ParseTemplateGlob(a)
	x := template.SetMust(a())
}
`,
	},
}
