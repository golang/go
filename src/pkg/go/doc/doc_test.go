// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
	"text/template"
)

type sources map[string]string // filename -> file contents

type testCase struct {
	name       string
	importPath string
	mode       Mode
	srcs       sources
	doc        string
}

var tests = make(map[string]*testCase)

// To register a new test case, use the pattern:
//
//	var _ = register(&testCase{ ... })
//
// (The result value of register is always 0 and only present to enable the pattern.)
//
func register(test *testCase) int {
	if _, found := tests[test.name]; found {
		panic(fmt.Sprintf("registration failed: test case %q already exists", test.name))
	}
	tests[test.name] = test
	return 0
}

func runTest(t *testing.T, test *testCase) {
	// create AST
	fset := token.NewFileSet()
	var pkg ast.Package
	pkg.Files = make(map[string]*ast.File)
	for filename, src := range test.srcs {
		file, err := parser.ParseFile(fset, filename, src, parser.ParseComments)
		if err != nil {
			t.Errorf("test %s: %v", test.name, err)
			return
		}
		switch {
		case pkg.Name == "":
			pkg.Name = file.Name.Name
		case pkg.Name != file.Name.Name:
			t.Errorf("test %s: different package names in test files", test.name)
			return
		}
		pkg.Files[filename] = file
	}

	doc := New(&pkg, test.importPath, test.mode).String()
	if doc != test.doc {
		//TODO(gri) Enable this once the sorting issue of comments is fixed
		//t.Errorf("test %s\n\tgot : %s\n\twant: %s", test.name, doc, test.doc)
	}
}

func Test(t *testing.T) {
	for _, test := range tests {
		runTest(t, test)
	}
}

// ----------------------------------------------------------------------------
// Printing support

func (pkg *Package) String() string {
	var buf bytes.Buffer
	docText.Execute(&buf, pkg) // ignore error - test will fail w/ incorrect output
	return buf.String()
}

// TODO(gri) complete template
var docText = template.Must(template.New("docText").Parse(
	`
PACKAGE {{.Name}}
DOC {{printf "%q" .Doc}}
IMPORTPATH {{.ImportPath}}
FILENAMES {{.Filenames}}
`))

// ----------------------------------------------------------------------------
// Test cases

// Test that all package comments and bugs are collected,
// and that the importPath is correctly set.
//
var _ = register(&testCase{
	name:       "p",
	importPath: "p",
	srcs: sources{
		"p1.go": "// comment 1\npackage p\n//BUG(uid): bug1",
		"p0.go": "// comment 0\npackage p\n// BUG(uid): bug0",
	},
	doc: `
PACKAGE p
DOC "comment 0\n\ncomment 1\n"
IMPORTPATH p
FILENAMES [p0.go p1.go]
`,
})

// Test basic functionality.
//
var _ = register(&testCase{
	name:       "p1",
	importPath: "p",
	srcs: sources{
		"p.go": `
package p
import "a"
const pi = 3.14       // pi
type T struct{}       // T
var V T               // v
func F(x int) int {}  // F
`,
	},
	doc: `
PACKAGE p
DOC ""
IMPORTPATH p
FILENAMES [p.go]
`,
})
