// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// No testdata on Android.

// +build !android

package rta_test

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"io/ioutil"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/go/callgraph"
	"golang.org/x/tools/go/callgraph/rta"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
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
//
func TestRTA(t *testing.T) {
	for _, filename := range inputs {
		content, err := ioutil.ReadFile(filename)
		if err != nil {
			t.Errorf("couldn't read file '%s': %s", filename, err)
			continue
		}

		conf := loader.Config{
			ParserMode: parser.ParseComments,
		}
		f, err := conf.ParseFile(filename, content)
		if err != nil {
			t.Error(err)
			continue
		}

		want, pos := expectation(f)
		if pos == token.NoPos {
			t.Errorf("No WANT: comment in %s", filename)
			continue
		}

		conf.CreateFromFiles("main", f)
		iprog, err := conf.Load()
		if err != nil {
			t.Error(err)
			continue
		}

		prog := ssautil.CreateProgram(iprog, 0)
		mainPkg := prog.Package(iprog.Created[0].Pkg)
		prog.Build()

		res := rta.Analyze([]*ssa.Function{
			mainPkg.Func("main"),
			mainPkg.Func("init"),
		}, true)

		if got := printResult(res, mainPkg.Pkg); got != want {
			t.Errorf("%s: got:\n%s\nwant:\n%s",
				prog.Fset.Position(pos), got, want)
		}
	}
}

func printResult(res *rta.Result, from *types.Package) string {
	var buf bytes.Buffer

	writeSorted := func(ss []string) {
		sort.Strings(ss)
		for _, s := range ss {
			fmt.Fprintf(&buf, "  %s\n", s)
		}
	}

	buf.WriteString("Dynamic calls\n")
	var edges []string
	callgraph.GraphVisitEdges(res.CallGraph, func(e *callgraph.Edge) error {
		if strings.Contains(e.Description(), "dynamic") {
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
