// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa_test

import (
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"log"
	"os"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
)

const hello = `
package main

import "fmt"

const message = "Hello, World!"

func main() {
	fmt.Println(message)
}
`

// This program demonstrates how to run the SSA builder on a single
// package of one or more already-parsed files.  Its dependencies are
// loaded from compiler export data.  This is what you'd typically use
// for a compiler; it does not depend on golang.org/x/tools/go/loader.
//
// It shows the printed representation of packages, functions, and
// instructions.  Within the function listing, the name of each
// BasicBlock such as ".0.entry" is printed left-aligned, followed by
// the block's Instructions.
//
// For each instruction that defines an SSA virtual register
// (i.e. implements Value), the type of that value is shown in the
// right column.
//
// Build and run the ssadump.go program if you want a standalone tool
// with similar functionality. It is located at
// golang.org/x/tools/cmd/ssadump.
//
func ExampleBuildPackage() {
	// Parse the source files.
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "hello.go", hello, parser.ParseComments)
	if err != nil {
		fmt.Print(err) // parse error
		return
	}
	files := []*ast.File{f}

	// Create the type-checker's package.
	pkg := types.NewPackage("hello", "")

	// Type-check the package, load dependencies.
	// Create and build the SSA program.
	hello, _, err := ssautil.BuildPackage(
		&types.Config{Importer: importer.Default()}, fset, pkg, files, ssa.SanityCheckFunctions)
	if err != nil {
		fmt.Print(err) // type error in some package
		return
	}

	// Print out the package.
	hello.WriteTo(os.Stdout)

	// Print out the package-level functions.
	hello.Func("init").WriteTo(os.Stdout)
	hello.Func("main").WriteTo(os.Stdout)

	// Output:
	//
	// package hello:
	//   func  init       func()
	//   var   init$guard bool
	//   func  main       func()
	//   const message    message = "Hello, World!":untyped string
	//
	// # Name: hello.init
	// # Package: hello
	// # Synthetic: package initializer
	// func init():
	// 0:                                                                entry P:0 S:2
	// 	t0 = *init$guard                                                   bool
	// 	if t0 goto 2 else 1
	// 1:                                                           init.start P:1 S:1
	// 	*init$guard = true:bool
	// 	t1 = fmt.init()                                                      ()
	// 	jump 2
	// 2:                                                            init.done P:2 S:0
	// 	return
	//
	// # Name: hello.main
	// # Package: hello
	// # Location: hello.go:8:6
	// func main():
	// 0:                                                                entry P:0 S:0
	// 	t0 = new [1]interface{} (varargs)                       *[1]interface{}
	// 	t1 = &t0[0:int]                                            *interface{}
	// 	t2 = make interface{} <- string ("Hello, World!":string)    interface{}
	// 	*t1 = t2
	// 	t3 = slice t0[:]                                          []interface{}
	// 	t4 = fmt.Println(t3...)                              (n int, err error)
	// 	return
}

// This example builds SSA code for a set of packages using the
// x/tools/go/packages API. This is what you would typically use for a
// analysis capable of operating on a single package.
func ExampleLoadPackages() {
	// Load, parse, and type-check the initial packages.
	cfg := &packages.Config{Mode: packages.LoadSyntax}
	initial, err := packages.Load(cfg, "fmt", "net/http")
	if err != nil {
		log.Fatal(err)
	}

	// Stop if any package had errors.
	// This step is optional; without it, the next step
	// will create SSA for only a subset of packages.
	if packages.PrintErrors(initial) > 0 {
		log.Fatalf("packages contain errors")
	}

	// Create SSA packages for all well-typed packages.
	prog, pkgs := ssautil.Packages(initial, ssa.PrintPackages)
	_ = prog

	// Build SSA code for the well-typed initial packages.
	for _, p := range pkgs {
		if p != nil {
			p.Build()
		}
	}
}

// This example builds SSA code for a set of packages plus all their dependencies,
// using the x/tools/go/packages API.
// This is what you'd typically use for a whole-program analysis.
func ExampleLoadWholeProgram() {
	// Load, parse, and type-check the whole program.
	cfg := packages.Config{Mode: packages.LoadAllSyntax}
	initial, err := packages.Load(&cfg, "fmt", "net/http")
	if err != nil {
		log.Fatal(err)
	}

	// Create SSA packages for all well-typed packages.
	prog, pkgs := ssautil.Packages(initial, ssa.PrintPackages)
	_ = pkgs

	// Build SSA code for the whole program.
	prog.Build()
}
