// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// No testdata on Android.

//go:build !android
// +build !android

package cha_test

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/go/callgraph"
	"golang.org/x/tools/go/callgraph/cha"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
	"golang.org/x/tools/internal/typeparams"
)

var inputs = []string{
	"testdata/func.go",
	"testdata/iface.go",
	"testdata/recv.go",
	"testdata/issue23925.go",
}

func expectation(f *ast.File) (string, token.Pos) {
	for _, c := range f.Comments {
		text := strings.TrimSpace(c.Text())
		if t := strings.TrimPrefix(text, "WANT:\n"); t != text {
			return t, c.Pos()
		}
	}
	return "", token.NoPos
}

// TestCHA runs CHA on each file in inputs, prints the dynamic edges of
// the call graph, and compares it with the golden results embedded in
// the WANT comment at the end of the file.
func TestCHA(t *testing.T) {
	for _, filename := range inputs {
		prog, f, mainPkg, err := loadProgInfo(filename, ssa.InstantiateGenerics)
		if err != nil {
			t.Error(err)
			continue
		}

		want, pos := expectation(f)
		if pos == token.NoPos {
			t.Error(fmt.Errorf("No WANT: comment in %s", filename))
			continue
		}

		cg := cha.CallGraph(prog)

		if got := printGraph(cg, mainPkg.Pkg, "dynamic", "Dynamic calls"); got != want {
			t.Errorf("%s: got:\n%s\nwant:\n%s",
				prog.Fset.Position(pos), got, want)
		}
	}
}

// TestCHAGenerics is TestCHA tailored for testing generics,
func TestCHAGenerics(t *testing.T) {
	if !typeparams.Enabled {
		t.Skip("TestCHAGenerics requires type parameters")
	}

	filename := "testdata/generics.go"
	prog, f, mainPkg, err := loadProgInfo(filename, ssa.InstantiateGenerics)
	if err != nil {
		t.Fatal(err)
	}

	want, pos := expectation(f)
	if pos == token.NoPos {
		t.Fatal(fmt.Errorf("No WANT: comment in %s", filename))
	}

	cg := cha.CallGraph(prog)

	if got := printGraph(cg, mainPkg.Pkg, "", "All calls"); got != want {
		t.Errorf("%s: got:\n%s\nwant:\n%s",
			prog.Fset.Position(pos), got, want)
	}
}

func loadProgInfo(filename string, mode ssa.BuilderMode) (*ssa.Program, *ast.File, *ssa.Package, error) {
	content, err := os.ReadFile(filename)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("couldn't read file '%s': %s", filename, err)
	}

	conf := loader.Config{
		ParserMode: parser.ParseComments,
	}
	f, err := conf.ParseFile(filename, content)
	if err != nil {
		return nil, nil, nil, err
	}

	conf.CreateFromFiles("main", f)
	iprog, err := conf.Load()
	if err != nil {
		return nil, nil, nil, err
	}

	prog := ssautil.CreateProgram(iprog, mode)
	prog.Build()

	return prog, f, prog.Package(iprog.Created[0].Pkg), nil
}

// printGraph returns a string representation of cg involving only edges
// whose description contains edgeMatch. The string representation is
// prefixed with a desc line.
func printGraph(cg *callgraph.Graph, from *types.Package, edgeMatch string, desc string) string {
	var edges []string
	callgraph.GraphVisitEdges(cg, func(e *callgraph.Edge) error {
		if strings.Contains(e.Description(), edgeMatch) {
			edges = append(edges, fmt.Sprintf("%s --> %s",
				e.Caller.Func.RelString(from),
				e.Callee.Func.RelString(from)))
		}
		return nil
	})
	sort.Strings(edges)

	var buf bytes.Buffer
	buf.WriteString(desc + "\n")
	for _, edge := range edges {
		fmt.Fprintf(&buf, "  %s\n", edge)
	}
	return strings.TrimSpace(buf.String())
}
