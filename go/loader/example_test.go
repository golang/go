// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loader_test

import (
	"fmt"
	"go/token"
	"log"
	"path/filepath"
	"runtime"
	"sort"

	"golang.org/x/tools/go/loader"
)

func printProg(prog *loader.Program) {
	// Created packages are the initial packages specified by a call
	// to CreateFromFilenames or CreateFromFiles.
	var names []string
	for _, info := range prog.Created {
		names = append(names, info.Pkg.Path())
	}
	fmt.Printf("created: %s\n", names)

	// Imported packages are the initial packages specified by a
	// call to Import or ImportWithTests.
	names = nil
	for _, info := range prog.Imported {
		names = append(names, info.Pkg.Path())
	}
	sort.Strings(names)
	fmt.Printf("imported: %s\n", names)

	// InitialPackages contains the union of created and imported.
	names = nil
	for _, info := range prog.InitialPackages() {
		names = append(names, info.Pkg.Path())
	}
	sort.Strings(names)
	fmt.Printf("initial: %s\n", names)

	// AllPackages contains all initial packages and their dependencies.
	names = nil
	for pkg := range prog.AllPackages {
		names = append(names, pkg.Path())
	}
	sort.Strings(names)
	fmt.Printf("all: %s\n", names)
}

func printFilenames(fset *token.FileSet, info *loader.PackageInfo) {
	var names []string
	for _, f := range info.Files {
		names = append(names, filepath.Base(fset.File(f.Pos()).Name()))
	}
	fmt.Printf("%s.Files: %s\n", info.Pkg.Path(), names)
}

// ExampleFromArgs loads a set of packages and all their dependencies
// from a typical command-line.  FromArgs parses a command line and
// makes calls to the other methods of Config shown in the examples that
// follow.
func ExampleFromArgs() {
	args := []string{"mytool", "unicode/utf8", "errors", "runtime", "--", "foo", "bar"}
	const wantTests = false

	var conf loader.Config
	rest, err := conf.FromArgs(args[1:], wantTests)
	prog, err := conf.Load()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("rest: %s\n", rest)
	printProg(prog)
	// Output:
	// rest: [foo bar]
	// created: []
	// imported: [errors runtime unicode/utf8]
	// initial: [errors runtime unicode/utf8]
	// all: [errors runtime unicode/utf8]
}

// ExampleCreateFromFilenames loads a single package (without tests) and
// all its dependencies from a list of filenames.
func ExampleCreateFromFilenames() {
	var conf loader.Config
	filename := filepath.Join(runtime.GOROOT(), "src/container/heap/heap.go")
	conf.CreateFromFilenames("container/heap", filename)
	prog, err := conf.Load()
	if err != nil {
		log.Fatal(err)
	}

	printProg(prog)
	// Output:
	// created: [container/heap]
	// imported: []
	// initial: [container/heap]
	// all: [container/heap sort]
}

// In the examples below, for stability, the chosen packages are
// relatively small, platform-independent, and low-level (and thus
// infrequently changing).
// The strconv package has internal and external tests.

const hello = `package main

import "fmt"

func main() {
	fmt.Println("Hello, world.")
}
`

// ExampleCreateFromFiles loads a package and all its dependencies from
// a list of already-parsed files.
func ExampleCreateFromFiles() {
	var conf loader.Config
	f, err := conf.ParseFile("hello.go", hello)
	if err != nil {
		log.Fatal(err)
	}
	conf.CreateFromFiles("hello", f)
	prog, err := conf.Load()
	if err != nil {
		log.Fatal(err)
	}

	printProg(prog)
	printFilenames(prog.Fset, prog.Package("strconv"))
	// Output:
	// created: [hello]
	// imported: []
	// initial: [hello]
	// all: [errors fmt hello io math os reflect runtime strconv sync sync/atomic syscall time unicode/utf8]
	// strconv.Files: [atob.go atof.go atoi.go decimal.go extfloat.go ftoa.go isprint.go itoa.go quote.go]
}

// ExampleImport loads three packages and all their dependencies.
func ExampleImport() {
	// ImportWithTest("strconv") causes strconv to include
	// internal_test.go, and creates an external test package,
	// strconv_test.
	// (Compare with output of ExampleCreateFromFiles.)

	var conf loader.Config
	conf.Import("unicode/utf8")
	conf.Import("errors")
	conf.ImportWithTests("strconv")
	prog, err := conf.Load()
	if err != nil {
		log.Fatal(err)
	}

	printProg(prog)
	printFilenames(prog.Fset, prog.Package("strconv"))
	printFilenames(prog.Fset, prog.Package("strconv_test"))
	// Output:
	// created: [strconv_test]
	// imported: [errors strconv unicode/utf8]
	// initial: [errors strconv strconv_test unicode/utf8]
	// all: [bufio bytes errors flag fmt io math math/rand os reflect runtime runtime/pprof sort strconv strconv_test strings sync sync/atomic syscall testing text/tabwriter time unicode unicode/utf8]
	// strconv.Files: [atob.go atof.go atoi.go decimal.go extfloat.go ftoa.go isprint.go itoa.go quote.go internal_test.go]
	// strconv_test.Files: [atob_test.go atof_test.go atoi_test.go decimal_test.go fp_test.go ftoa_test.go itoa_test.go quote_example_test.go quote_test.go strconv_test.go]
}
