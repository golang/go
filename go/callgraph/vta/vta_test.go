// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vta

import (
	"fmt"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/go/callgraph"
	"golang.org/x/tools/go/callgraph/cha"
	"golang.org/x/tools/go/ssa"

	"golang.org/x/tools/go/ssa/ssautil"
)

// callGraphStr stringifes `g` into a list of strings where
// each entry is of the form
//   f: cs1 -> f1, f2, ...; ...; csw -> fx, fy, ...
// f is a function, cs1, ..., csw are call sites in f, and
// f1, f2, ..., fx, fy, ... are the resolved callees.
func callGraphStr(g *callgraph.Graph) []string {
	var gs []string
	for f, n := range g.Nodes {
		c := make(map[string][]string)
		for _, edge := range n.Out {
			cs := edge.Site.String()
			c[cs] = append(c[cs], funcName(edge.Callee.Func))
		}

		var cs []string
		for site, fs := range c {
			sort.Strings(fs)
			entry := fmt.Sprintf("%v -> %v", site, strings.Join(fs, ", "))
			cs = append(cs, entry)
		}

		sort.Strings(cs)
		entry := fmt.Sprintf("%v: %v", funcName(f), strings.Join(cs, "; "))
		gs = append(gs, entry)
	}
	return gs
}

func TestVTACallGraph(t *testing.T) {
	for _, file := range []string{
		"testdata/callgraph_static.go",
		"testdata/callgraph_ho.go",
		"testdata/callgraph_interfaces.go",
		"testdata/callgraph_pointers.go",
		"testdata/callgraph_collections.go",
	} {
		t.Run(file, func(t *testing.T) {
			prog, want, err := testProg(file)
			if err != nil {
				t.Fatalf("couldn't load test file '%s': %s", file, err)
			}
			if len(want) == 0 {
				t.Fatalf("couldn't find want in `%s`", file)
			}

			g := CallGraph(ssautil.AllFunctions(prog), cha.CallGraph(prog))
			if got := callGraphStr(g); !subGraph(want, got) {
				t.Errorf("computed callgraph %v should contain %v", got, want)
			}
		})
	}
}

// TestVTAProgVsFuncSet exemplifies and tests different possibilities
// enabled by having an arbitrary function set as input to CallGraph
// instead of the whole program (i.e., ssautil.AllFunctions(prog)).
func TestVTAProgVsFuncSet(t *testing.T) {
	prog, want, err := testProg("testdata/callgraph_nested_ptr.go")
	if err != nil {
		t.Fatalf("couldn't load test `testdata/callgraph_nested_ptr.go`: %s", err)
	}
	if len(want) == 0 {
		t.Fatal("couldn't find want in `testdata/callgraph_nested_ptr.go`")
	}

	allFuncs := ssautil.AllFunctions(prog)
	g := CallGraph(allFuncs, cha.CallGraph(prog))
	// VTA over the whole program will produce a call graph that
	// includes Baz:(**i).Foo -> A.Foo, B.Foo.
	if got := callGraphStr(g); !subGraph(want, got) {
		t.Errorf("computed callgraph %v should contain %v", got, want)
	}

	// Prune the set of program functions to exclude Bar(). This should
	// yield a call graph that includes different set of callees for Baz
	// Baz:(**i).Foo -> A.Foo
	//
	// Note that the exclusion of Bar can happen, for instance, if Baz is
	// considered an entry point of some data flow analysis and Bar is
	// provably (e.g., using CHA forward reachability) unreachable from Baz.
	noBarFuncs := make(map[*ssa.Function]bool)
	for f, in := range allFuncs {
		noBarFuncs[f] = in && (funcName(f) != "Bar")
	}
	want = []string{"Baz: Do(i) -> Do; invoke t2.Foo() -> A.Foo"}
	g = CallGraph(noBarFuncs, cha.CallGraph(prog))
	if got := callGraphStr(g); !subGraph(want, got) {
		t.Errorf("pruned callgraph %v should contain %v", got, want)
	}
}
