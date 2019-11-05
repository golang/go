package source

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"testing"

	"golang.org/x/tools/internal/lsp/diff"
)

var fset = token.NewFileSet()

func parse(t *testing.T, name, in string) *ast.File {
	file, err := parser.ParseFile(fset, name, in, parser.ParseComments)
	if err != nil {
		t.Fatalf("%s parse: %v", name, err)
	}
	return file
}

func print(t *testing.T, name string, f *ast.File) string {
	var buf bytes.Buffer
	if err := format.Node(&buf, fset, f); err != nil {
		t.Fatalf("%s gofmt: %v", name, err)
	}
	return buf.String()
}

type test struct {
	name       string
	renamedPkg string
	pkg        string
	in         string
	want       []imp
	unchanged  bool // Expect added/deleted return value to be false.
}

type imp struct {
	name string
	path string
}

var addTests = []test{
	{
		name: "leave os alone",
		pkg:  "os",
		in: `package main

import (
	"os"
)
`,
		want: []imp{
			{
				name: "",
				path: "os",
			},
		},
		unchanged: true,
	},
	{
		name: "package statement only",
		pkg:  "os",
		in: `package main
`,
		want: []imp{
			{
				name: "",
				path: "os",
			},
		},
	},
	{
		name: "package statement no new line",
		pkg:  "os",
		in:   `package main`,
		want: []imp{
			{
				name: "",
				path: "os",
			},
		},
	},
	{
		// Issue 33721: add import statement after package declaration preceded by comments.
		name: "issue 33721 package statement comments before",
		pkg:  "os",
		in: `// Here is a comment before
package main
`,
		want: []imp{
			{
				name: "",
				path: "os",
			},
		},
	},
	{
		name: "package statement comments same line",
		pkg:  "os",
		in: `package main // Here is a comment after
`,
		want: []imp{
			{
				name: "",
				path: "os",
			},
		},
	},
	{
		name: "package statement comments before and after",
		pkg:  "os",
		in: `// Here is a comment before
package main // Here is a comment after`,
		want: []imp{
			{
				name: "",
				path: "os",
			},
		},
	},
	{
		name: "package statement multiline comments",
		pkg:  "os",
		in: `package main /* This is a multiline comment
and it extends
further down*/`,
		want: []imp{
			{
				name: "",
				path: "os",
			},
		},
	},
	{
		name: "import c",
		pkg:  "os",
		in: `package main

import "C"
`,
		want: []imp{
			{
				name: "",
				path: "os",
			},
			{
				name: "",
				path: "C",
			},
		},
	},
	{
		name: "existing imports",
		pkg:  "os",
		in: `package main

import "io"
`,
		want: []imp{
			{
				name: "",
				path: "os",
			},
			{
				name: "",
				path: "io",
			},
		},
	},
	{
		name: "existing imports with comment",
		pkg:  "os",
		in: `package main

import "io" // A comment
`,
		want: []imp{
			{
				name: "",
				path: "os",
			},
			{
				name: "",
				path: "io",
			},
		},
	},
	{
		name: "existing imports multiline comment",
		pkg:  "os",
		in: `package main

import "io" /* A comment
that
extends */
`,
		want: []imp{
			{
				name: "",
				path: "os",
			},
			{
				name: "",
				path: "io",
			},
		},
	},
	{
		name:       "renamed import",
		renamedPkg: "o",
		pkg:        "os",
		in: `package main
`,
		want: []imp{
			{
				name: "o",
				path: "os",
			},
		},
	},
}

func TestAddImport(t *testing.T) {
	for _, test := range addTests {
		file := parse(t, test.name, test.in)
		var before bytes.Buffer
		ast.Fprint(&before, fset, file, nil)
		edits, err := addNamedImport(fset, file, test.renamedPkg, test.pkg)
		if err != nil && !test.unchanged {
			t.Errorf("error adding import: %s", err)
			continue
		}

		// Apply the edits and parse the file.
		got := applyEdits(test.in, edits)
		gotFile := parse(t, test.name, got)

		compareImports(t, fmt.Sprintf("first run: %s:\n", test.name), gotFile.Imports, test.want)

		// AddNamedImport should be idempotent. Verify that by calling it again,
		// expecting no change to the AST, and the returned added value to always be false.
		edits, err = addNamedImport(fset, gotFile, test.renamedPkg, test.pkg)
		if err != nil && !test.unchanged {
			t.Errorf("error adding import: %s", err)
			continue
		}
		// Apply the edits and parse the file.
		got = applyEdits(got, edits)
		gotFile = parse(t, test.name, got)

		compareImports(t, test.name, gotFile.Imports, test.want)

	}
}

func TestDoubleAddNamedImport(t *testing.T) {
	name := "doublenamedimport"
	in := "package main\n"
	file := parse(t, name, in)
	// Add a named import
	edits, err := addNamedImport(fset, file, "o", "os")
	if err != nil {
		t.Errorf("error adding import: %s", err)
		return
	}
	got := applyEdits(in, edits)
	gotFile := parse(t, name, got)

	// Add a second named import
	edits, err = addNamedImport(fset, gotFile, "i", "io")
	if err != nil {
		t.Errorf("error adding import: %s", err)
		return
	}
	got = applyEdits(got, edits)
	gotFile = parse(t, name, got)

	want := []imp{
		{
			name: "o",
			path: "os",
		},
		{
			name: "i",
			path: "io",
		},
	}
	compareImports(t, "", gotFile.Imports, want)
}

func compareImports(t *testing.T, prefix string, got []*ast.ImportSpec, want []imp) {
	if len(got) != len(want) {
		t.Errorf("%s\ngot %d imports\nwant %d", prefix, len(got), len(want))
		return
	}

	for _, imp := range got {
		name := importName(imp)
		path := importPath(imp)
		found := false
		for _, want := range want {
			if want.name == name && want.path == path {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("%s\n\ngot unexpected import: name: %q,path: %q", prefix, name, path)
			continue
		}
	}
}

func applyEdits(contents string, edits []diff.TextEdit) string {
	res := contents

	// Apply the edits from the end of the file forward
	// to preserve the offsets
	for i := len(edits) - 1; i >= 0; i-- {
		edit := edits[i]
		start := edit.Span.Start().Offset()
		end := edit.Span.End().Offset()
		tmp := res[0:start] + edit.NewText
		res = tmp + res[end:]
	}
	return res
}
