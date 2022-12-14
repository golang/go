// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package checker_test

import (
	"fmt"
	"go/ast"
	"io/ioutil"
	"path/filepath"
	"testing"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/internal/checker"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/testenv"
)

func TestApplyFixes(t *testing.T) {
	testenv.NeedsGoPackages(t)

	files := map[string]string{
		"rename/test.go": `package rename

func Foo() {
	bar := 12
	_ = bar
}

// the end
`}
	want := `package rename

func Foo() {
	baz := 12
	_ = baz
}

// the end
`

	testdata, cleanup, err := analysistest.WriteFiles(files)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(testdata, "src/rename/test.go")
	checker.Fix = true
	checker.Run([]string{"file=" + path}, []*analysis.Analyzer{analyzer})

	contents, err := ioutil.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	got := string(contents)
	if got != want {
		t.Errorf("contents of rewritten file\ngot: %s\nwant: %s", got, want)
	}

	defer cleanup()
}

var analyzer = &analysis.Analyzer{
	Name:     "rename",
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

var other = &analysis.Analyzer{ // like analyzer but with a different Name.
	Name:     "other",
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (interface{}, error) {
	const (
		from      = "bar"
		to        = "baz"
		conflict  = "conflict"  // add conflicting edits to package conflict.
		duplicate = "duplicate" // add duplicate edits to package conflict.
		other     = "other"     // add conflicting edits to package other from different analyzers.
	)

	if pass.Analyzer.Name == other {
		if pass.Pkg.Name() != other {
			return nil, nil // only apply Analyzer other to packages named other
		}
	}

	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	nodeFilter := []ast.Node{(*ast.Ident)(nil)}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		ident := n.(*ast.Ident)
		if ident.Name == from {
			msg := fmt.Sprintf("renaming %q to %q", from, to)
			edits := []analysis.TextEdit{
				{Pos: ident.Pos(), End: ident.End(), NewText: []byte(to)},
			}
			switch pass.Pkg.Name() {
			case conflict:
				edits = append(edits, []analysis.TextEdit{
					{Pos: ident.Pos() - 1, End: ident.End(), NewText: []byte(to)},
					{Pos: ident.Pos(), End: ident.End() - 1, NewText: []byte(to)},
					{Pos: ident.Pos(), End: ident.End(), NewText: []byte("lorem ipsum")},
				}...)
			case duplicate:
				edits = append(edits, edits...)
			case other:
				if pass.Analyzer.Name == other {
					edits[0].Pos = edits[0].Pos + 1 // shift by one to mismatch analyzer and other
				}
			}
			pass.Report(analysis.Diagnostic{
				Pos:            ident.Pos(),
				End:            ident.End(),
				Message:        msg,
				SuggestedFixes: []analysis.SuggestedFix{{Message: msg, TextEdits: edits}}})
		}
	})

	return nil, nil
}

func TestRunDespiteErrors(t *testing.T) {
	testenv.NeedsGoPackages(t)

	files := map[string]string{
		"rderr/test.go": `package rderr

// Foo deliberately has a type error
func Foo(s string) int {
	return s + 1
}
`}

	testdata, cleanup, err := analysistest.WriteFiles(files)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(testdata, "src/rderr/test.go")

	// A no-op analyzer that should finish regardless of
	// parse or type errors in the code.
	noop := &analysis.Analyzer{
		Name:     "noop",
		Requires: []*analysis.Analyzer{inspect.Analyzer},
		Run: func(pass *analysis.Pass) (interface{}, error) {
			return nil, nil
		},
		RunDespiteErrors: true,
	}

	// A no-op analyzer that should finish regardless of
	// parse or type errors in the code.
	noopWithFact := &analysis.Analyzer{
		Name:     "noopfact",
		Requires: []*analysis.Analyzer{inspect.Analyzer},
		Run: func(pass *analysis.Pass) (interface{}, error) {
			return nil, nil
		},
		RunDespiteErrors: true,
		FactTypes:        []analysis.Fact{&EmptyFact{}},
	}

	for _, test := range []struct {
		name      string
		pattern   []string
		analyzers []*analysis.Analyzer
		code      int
	}{
		// parse/type errors
		{name: "skip-error", pattern: []string{"file=" + path}, analyzers: []*analysis.Analyzer{analyzer}, code: 1},
		// RunDespiteErrors allows a driver to run an Analyzer even after parse/type errors.
		//
		// The noop analyzer doesn't use facts, so the driver loads only the root
		// package from source. For the rest, it asks 'go list' for export data,
		// which fails because the compiler encounters the type error.  Since the
		// errors come from 'go list', the driver doesn't run the analyzer.
		{name: "despite-error", pattern: []string{"file=" + path}, analyzers: []*analysis.Analyzer{noop}, code: 1},
		// The noopfact analyzer does use facts, so the driver loads source for
		// all dependencies, does type checking itself, recognizes the error as a
		// type error, and runs the analyzer.
		{name: "despite-error-fact", pattern: []string{"file=" + path}, analyzers: []*analysis.Analyzer{noopWithFact}, code: 0},
		// combination of parse/type errors and no errors
		{name: "despite-error-and-no-error", pattern: []string{"file=" + path, "sort"}, analyzers: []*analysis.Analyzer{analyzer, noop}, code: 1},
		// non-existing package error
		{name: "no-package", pattern: []string{"xyz"}, analyzers: []*analysis.Analyzer{analyzer}, code: 1},
		{name: "no-package-despite-error", pattern: []string{"abc"}, analyzers: []*analysis.Analyzer{noop}, code: 1},
		{name: "no-multi-package-despite-error", pattern: []string{"xyz", "abc"}, analyzers: []*analysis.Analyzer{noop}, code: 1},
		// combination of type/parsing and different errors
		{name: "different-errors", pattern: []string{"file=" + path, "xyz"}, analyzers: []*analysis.Analyzer{analyzer, noop}, code: 1},
		// non existing dir error
		{name: "no-match-dir", pattern: []string{"file=non/existing/dir"}, analyzers: []*analysis.Analyzer{analyzer, noop}, code: 1},
		// no errors
		{name: "no-errors", pattern: []string{"sort"}, analyzers: []*analysis.Analyzer{analyzer, noop}, code: 0},
	} {
		if test.name == "despite-error" && testenv.Go1Point() < 20 {
			// The behavior in the comment on the despite-error test only occurs for Go 1.20+.
			continue
		}
		if got := checker.Run(test.pattern, test.analyzers); got != test.code {
			t.Errorf("got incorrect exit code %d for test %s; want %d", got, test.name, test.code)
		}
	}

	defer cleanup()
}

type EmptyFact struct{}

func (f *EmptyFact) AFact() {}
