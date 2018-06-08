// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.7
// +build gc

package gcexportdata_test

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"log"
	"os"
	"path/filepath"

	"golang.org/x/tools/go/gcexportdata"
)

// ExampleRead uses gcexportdata.Read to load type information for the
// "fmt" package from the fmt.a file produced by the gc compiler.
func ExampleRead() {
	// Find the export data file.
	filename, path := gcexportdata.Find("fmt", "")
	if filename == "" {
		log.Fatalf("can't find export data for fmt")
	}
	fmt.Printf("Package path:       %s\n", path)
	fmt.Printf("Export data:        %s\n", filepath.Base(filename))

	// Open and read the file.
	f, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	r, err := gcexportdata.NewReader(f)
	if err != nil {
		log.Fatalf("reading export data %s: %v", filename, err)
	}

	// Decode the export data.
	fset := token.NewFileSet()
	imports := make(map[string]*types.Package)
	pkg, err := gcexportdata.Read(r, fset, imports, path)
	if err != nil {
		log.Fatal(err)
	}

	// Print package information.
	fmt.Printf("Package members:    %s...\n", pkg.Scope().Names()[:5])
	println := pkg.Scope().Lookup("Println")
	posn := fset.Position(println.Pos())
	posn.Line = 123 // make example deterministic
	fmt.Printf("Println type:       %s\n", println.Type())
	fmt.Printf("Println location:   %s\n", slashify(posn))

	// Output:
	//
	// Package path:       fmt
	// Export data:        fmt.a
	// Package members:    [Errorf Formatter Fprint Fprintf Fprintln]...
	// Println type:       func(a ...interface{}) (n int, err error)
	// Println location:   $GOROOT/src/fmt/print.go:123:1
}

// ExampleNewImporter demonstrates usage of NewImporter to provide type
// information for dependencies when type-checking Go source code.
func ExampleNewImporter() {
	const src = `package myscanner

// choosing a package that is unlikely to change across releases
import "text/scanner"

const eof = scanner.EOF
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "myscanner.go", src, 0)
	if err != nil {
		log.Fatal(err)
	}

	packages := make(map[string]*types.Package)
	imp := gcexportdata.NewImporter(fset, packages)
	conf := types.Config{Importer: imp}
	pkg, err := conf.Check("myscanner", fset, []*ast.File{f}, nil)
	if err != nil {
		log.Fatal(err)
	}

	// object from imported package
	pi := packages["text/scanner"].Scope().Lookup("EOF")
	fmt.Printf("const %s.%s %s = %s // %s\n",
		pi.Pkg().Path(),
		pi.Name(),
		pi.Type(),
		pi.(*types.Const).Val(),
		slashify(fset.Position(pi.Pos())),
	)

	// object in source package
	twopi := pkg.Scope().Lookup("eof")
	fmt.Printf("const %s %s = %s // %s\n",
		twopi.Name(),
		twopi.Type(),
		twopi.(*types.Const).Val(),
		slashify(fset.Position(twopi.Pos())),
	)

	// Output:
	//
	// const text/scanner.EOF untyped int = -1 // $GOROOT/src/text/scanner/scanner.go:75:1
	// const eof untyped int = -1 // myscanner.go:6:7
}

func slashify(posn token.Position) token.Position {
	posn.Filename = filepath.ToSlash(posn.Filename) // for MS Windows portability
	return posn
}
