// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pointer_test

import (
	"fmt"
	"sort"

	"golang.org/x/tools/go/callgraph"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/pointer"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
)

// This program demonstrates how to use the pointer analysis to
// obtain a conservative call-graph of a Go program.
// It also shows how to compute the points-to set of a variable,
// in this case, (C).f's ch parameter.
//
func Example() {
	const myprog = `
package main

import "fmt"

type I interface {
	f(map[string]int)
}

type C struct{}

func (C) f(m map[string]int) {
	fmt.Println("C.f()")
}

func main() {
	var i I = C{}
	x := map[string]int{"one":1}
	i.f(x) // dynamic method call
}
`
	var conf loader.Config

	// Parse the input file, a string.
	// (Command-line tools should use conf.FromArgs.)
	file, err := conf.ParseFile("myprog.go", myprog)
	if err != nil {
		fmt.Print(err) // parse error
		return
	}

	// Create single-file main package and import its dependencies.
	conf.CreateFromFiles("main", file)

	iprog, err := conf.Load()
	if err != nil {
		fmt.Print(err) // type error in some package
		return
	}

	// Create SSA-form program representation.
	prog := ssautil.CreateProgram(iprog, 0)
	mainPkg := prog.Package(iprog.Created[0].Pkg)

	// Build SSA code for bodies of all functions in the whole program.
	prog.Build()

	// Configure the pointer analysis to build a call-graph.
	config := &pointer.Config{
		Mains:          []*ssa.Package{mainPkg},
		BuildCallGraph: true,
	}

	// Query points-to set of (C).f's parameter m, a map.
	C := mainPkg.Type("C").Type()
	Cfm := prog.LookupMethod(C, mainPkg.Pkg, "f").Params[1]
	config.AddQuery(Cfm)

	// Run the pointer analysis.
	result, err := pointer.Analyze(config)
	if err != nil {
		panic(err) // internal error in pointer analysis
	}

	// Find edges originating from the main package.
	// By converting to strings, we de-duplicate nodes
	// representing the same function due to context sensitivity.
	var edges []string
	callgraph.GraphVisitEdges(result.CallGraph, func(edge *callgraph.Edge) error {
		caller := edge.Caller.Func
		if caller.Pkg == mainPkg {
			edges = append(edges, fmt.Sprint(caller, " --> ", edge.Callee.Func))
		}
		return nil
	})

	// Print the edges in sorted order.
	sort.Strings(edges)
	for _, edge := range edges {
		fmt.Println(edge)
	}
	fmt.Println()

	// Print the labels of (C).f(m)'s points-to set.
	fmt.Println("m may point to:")
	var labels []string
	for _, l := range result.Queries[Cfm].PointsTo().Labels() {
		label := fmt.Sprintf("  %s: %s", prog.Fset.Position(l.Pos()), l)
		labels = append(labels, label)
	}
	sort.Strings(labels)
	for _, label := range labels {
		fmt.Println(label)
	}

	// Output:
	// (main.C).f --> fmt.Println
	// main.init --> fmt.init
	// main.main --> (main.C).f
	//
	// m may point to:
	//   myprog.go:18:21: makemap
}
