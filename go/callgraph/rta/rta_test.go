// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// No testdata on Android.

//go:build !android
// +build !android

package rta_test

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
	"golang.org/x/tools/go/callgraph/rta"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
	"golang.org/x/tools/internal/typeparams"
)

var inputs = []string{
	"testdata/func.go",
	"testdata/rtype.go",
	"testdata/iface.go",
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

// TestRTA runs RTA on each file in inputs, prints the results, and
// compares it with the golden results embedded in the WANT comment at
// the end of the file.
//
// The results string consists of two parts: the set of dynamic call
// edges, "f --> g", one per line, and the set of reachable functions,
// one per line.  Each set is sorted.
func TestRTA(t *testing.T) {
	for _, filename := range inputs {
		prog, f, mainPkg, err := loadProgInfo(filename, ssa.BuilderMode(0))
		if err != nil {
			t.Error(err)
			continue
		}

		want, pos := expectation(f)
		if pos == token.NoPos {
			t.Errorf("No WANT: comment in %s", filename)
			continue
		}

		res := rta.Analyze([]*ssa.Function{
			mainPkg.Func("main"),
			mainPkg.Func("init"),
		}, true)

		if got := printResult(res, mainPkg.Pkg, "dynamic", "Dynamic calls"); got != want {
			t.Errorf("%s: got:\n%s\nwant:\n%s",
				prog.Fset.Position(pos), got, want)
		}
	}
}

// TestRTAGenerics is TestRTA specialized for testing generics.
func TestRTAGenerics(t *testing.T) {
	if !typeparams.Enabled {
		t.Skip("TestRTAGenerics requires type parameters")
	}

	filename := "testdata/generics.go"
	prog, f, mainPkg, err := loadProgInfo(filename, ssa.InstantiateGenerics)
	if err != nil {
		t.Fatal(err)
	}

	want, pos := expectation(f)
	if pos == token.NoPos {
		t.Fatalf("No WANT: comment in %s", filename)
	}

	res := rta.Analyze([]*ssa.Function{
		mainPkg.Func("main"),
		mainPkg.Func("init"),
	}, true)

	if got := printResult(res, mainPkg.Pkg, "", "All calls"); got != want {
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

// printResult returns a string representation of res, i.e., call graph,
// reachable functions, and reflect types. For call graph, only edges
// whose description contains edgeMatch are returned and their string
// representation is prefixed with a desc line.
func printResult(res *rta.Result, from *types.Package, edgeMatch, desc string) string {
	var buf bytes.Buffer

	writeSorted := func(ss []string) {
		sort.Strings(ss)
		for _, s := range ss {
			fmt.Fprintf(&buf, "  %s\n", s)
		}
	}

	buf.WriteString(desc + "\n")
	var edges []string
	callgraph.GraphVisitEdges(res.CallGraph, func(e *callgraph.Edge) error {
		if strings.Contains(e.Description(), edgeMatch) {
			edges = append(edges, fmt.Sprintf("%s --> %s",
				e.Caller.Func.RelString(from),
				e.Callee.Func.RelString(from)))
		}
		return nil
	})
	writeSorted(edges)

	buf.WriteString("Reachable functions\n")
	var reachable []string
	for f := range res.Reachable {
		reachable = append(reachable, f.RelString(from))
	}
	writeSorted(reachable)

	buf.WriteString("Reflect types\n")
	var rtypes []string
	res.RuntimeTypes.Iterate(func(key types.Type, value interface{}) {
		if value == false { // accessible to reflection
			rtypes = append(rtypes, types.TypeString(key, types.RelativeTo(from)))
		}
	})
	writeSorted(rtypes)

	return strings.TrimSpace(buf.String())
}
