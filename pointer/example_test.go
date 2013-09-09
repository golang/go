// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pointer_test

import (
	"fmt"
	"go/build"
	"go/parser"
	"sort"

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
	file, err := parser.ParseFile(imp.Fset, "myprog.go", myprog, parser.DeclarationErrors)
	if err != nil {
		fmt.Print(err) // parse error
		return
	}

	// Create a "main" package containing one file.
	mainInfo := imp.LoadMainPackage(file)

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
	callgraph := make(pointer.CallGraph)
	config := &pointer.Config{
		Mains: []*ssa.Package{mainPkg},
		Call:  callgraph.AddEdge,
	}
	root := pointer.Analyze(config)

	// Visit callgraph in depth-first order.
	//
	// There may be multiple nodes for the
	// same function due to context sensitivity.
	var edges []string // call edges originating from the main package.
	seen := make(map[pointer.CallGraphNode]bool)
	var visit func(cgn pointer.CallGraphNode)
	visit = func(cgn pointer.CallGraphNode) {
		if seen[cgn] {
			return // already seen
		}
		seen[cgn] = true
		caller := cgn.Func()
		for callee := range callgraph[cgn] {
			if caller.Pkg == mainPkg {
				edges = append(edges, fmt.Sprint(caller, " --> ", callee.Func()))
			}
			visit(callee)
		}
	}
	visit(root)

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
