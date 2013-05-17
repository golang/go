package ssa_test

import (
	"code.google.com/p/go.tools/ssa"
	"fmt"
	"go/ast"
	"go/parser"
	"os"
)

// This example demonstrates the SSA builder.
func Example() {
	const hello = `
package main

import "fmt"

func main() {
	fmt.Println("Hello, World!")
}
`

	// Construct a builder.  Imports will be loaded as if by 'go build'.
	builder := ssa.NewBuilder(&ssa.Context{Loader: ssa.MakeGoBuildLoader(nil)})

	// Parse the input file.
	file, err := parser.ParseFile(builder.Prog.Files, "hello.go", hello, parser.DeclarationErrors)
	if err != nil {
		fmt.Printf("Parsing failed: %s\n", err.Error())
		return
	}

	// Create a "main" package containing one file.
	mainPkg, err := builder.CreatePackage("main", []*ast.File{file})
	if err != nil {
		fmt.Printf("Type-checking failed: %s\n", err.Error())
		return
	}

	// Build SSA code for bodies of functions in mainPkg.
	builder.BuildPackage(mainPkg)

	// Print out the package-level functions.
	for _, mem := range mainPkg.Members {
		if fn, ok := mem.(*ssa.Function); ok {
			fn.DumpTo(os.Stdout)
		}
	}

	// Output:
	// # Name: main.main
	// # Declared at hello.go:6:6
	// func main():
	// .0.entry:							       P:0 S:0
	// 	a0 = new [1]interface{}                                 *[1]interface{}
	// 	t0 = &a0[0:untyped integer]                                *interface{}
	// 	t1 = make interface interface{} <- string ("Hello, World!":string) interface{}
	// 	*t0 = t1
	// 	t2 = slice a0[:]                                          []interface{}
	// 	t3 = fmt.Println(t2)                                 (n int, err error)
	// 	ret
}
