package ssa_test

import (
	"code.google.com/p/go.tools/importer"
	"code.google.com/p/go.tools/ssa"
	"fmt"
	"go/ast"
	"go/parser"
	"os"
)

// This program demonstrates how to run the SSA builder on a "Hello,
// World!" program and shows the printed representation of packages,
// functions and instructions.
//
// Within the function listing, the name of each BasicBlock such as
// ".0.entry" is printed left-aligned, followed by the block's
// Instructions.
//
// For each instruction that defines an SSA virtual register
// (i.e. implements Value), the type of that value is shown in the
// right column.
//
// Build and run the ssadump.go program in this package if you want a
// standalone tool with similar functionality.
//
func Example() {
	const hello = `
package main

import "fmt"

const message = "Hello, World!"

func main() {
	fmt.Println(message)
}
`
	// Construct an importer.  Imports will be loaded as if by 'go build'.
	imp := importer.New(&importer.Context{Loader: importer.MakeGoBuildLoader(nil)})

	// Parse the input file.
	file, err := parser.ParseFile(imp.Fset, "hello.go", hello, parser.DeclarationErrors)
	if err != nil {
		fmt.Printf(err.Error()) // parse error
		return
	}

	// Create a "main" package containing one file.
	info, err := imp.CreateSourcePackage("main", []*ast.File{file})
	if err != nil {
		fmt.Printf(err.Error()) // type error
		return
	}

	// Create SSA-form program representation.
	var mode ssa.BuilderMode
	prog := ssa.NewProgram(imp.Fset, mode)
	prog.CreatePackages(imp)
	mainPkg := prog.Package(info.Pkg)

	// Print out the package.
	mainPkg.DumpTo(os.Stdout)
	fmt.Println()

	// Build SSA code for bodies of functions in mainPkg.
	mainPkg.Build()

	// Print out the package-level functions.
	mainPkg.Init.DumpTo(os.Stdout)
	for _, mem := range mainPkg.Members {
		if fn, ok := mem.(*ssa.Function); ok {
			fn.DumpTo(os.Stdout)
		}
	}

	// Output:
	//
	// package main:
	//   var   init$guard *bool
	//   func  main       func()
	//   const message    message = "Hello, World!":untyped string
	//
	// # Name: main.init
	// # Synthetic
	// func init():
	// .0.entry:                                                               P:0 S:2
	// 	t0 = *init$guard                                                   bool
	// 	if t0 goto 2.init.done else 1.init.start
	// .1.init.start:                                                          P:1 S:1
	// 	*init$guard = true:bool
	// 	t1 = fmt.init()                                                      ()
	// 	jump 2.init.done
	// .2.init.done:                                                           P:2 S:0
	// 	ret
	//
	// # Name: main.main
	// # Declared at hello.go:8:6
	// func main():
	// .0.entry:                                                               P:0 S:0
	// 	a0 = new [1]interface{}                                 *[1]interface{}
	// 	t0 = &a0[0:untyped integer]                                *interface{}
	// 	t1 = make interface interface{} <- string ("Hello, World!":string) interface{}
	// 	*t0 = t1
	// 	t2 = slice a0[:]                                          []interface{}
	// 	t3 = fmt.Println(t2)                                 (n int, err error)
	// 	ret
}
