// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pointer_test

import (
	"fmt"
	"go/build"
	"go/parser"
	"sort"

	"code.google.com/p/go.tools/call"
	"code.google.com/p/go.tools/importer"
	"code.google.com/p/go.tools/pointer"
	"code.google.com/p/go.tools/ssa"
)

// This program demonstrates how to use the pointer analysis to
// obtain a conservative call-graph of a Go program.
//
func Example() {
	const myprog = `
package main

import "fmt"

type I interface {
	f()
}

type C struct{}

func (C) f() {
	fmt.Println("C.f()")
}

func main() {
	var i I = C{}
	i.f() // dynamic method call
}
`
	// Construct an importer.
	// Imports will be loaded as if by 'go build'.
	imp := importer.New(&importer.Config{Build: &build.Default})

	// Parse the input file.
	file, err := parser.ParseFile(imp.Fset, "myprog.go", myprog, 0)
	if err != nil {
		fmt.Print(err) // parse error
		return
	}

	// Create single-file main package and import its dependencies.
	mainInfo := imp.CreatePackage("main", file)

	// Create SSA-form program representation.
	var mode ssa.BuilderMode
	prog := ssa.NewProgram(imp.Fset, mode)
	if err := prog.CreatePackages(imp); err != nil {
		fmt.Print(err) // type error in some package
		return
	}
	mainPkg := prog.Package(mainInfo.Pkg)

	// Build SSA code for bodies of all functions in the whole program.
	prog.BuildAll()

	// Run the pointer analysis and build the complete callgraph.
	config := &pointer.Config{
		Mains:          []*ssa.Package{mainPkg},
		BuildCallGraph: true,
	}
	result := pointer.Analyze(config)

	// Find edges originating from the main package.
	// By converting to strings, we de-duplicate nodes
	// representing the same function due to context sensitivity.
	var edges []string
	call.GraphVisitEdges(result.CallGraph, func(edge call.Edge) error {
		caller := edge.Caller.Func()
		if caller.Pkg == mainPkg {
			edges = append(edges, fmt.Sprint(caller, " --> ", edge.Callee.Func()))
		}
		return nil
	})

	// Print the edges in sorted order.
	sort.Strings(edges)
	for _, edge := range edges {
		fmt.Println(edge)
	}

	// Output:
	// (main.C).f --> fmt.Println
	// main.init --> fmt.init
	// main.main --> (main.C).f
}
