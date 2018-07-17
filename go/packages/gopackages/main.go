// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The gopackages command is a diagnostic tool that demonstrates
// how to use golang.org/x/tools/go/packages to load, parse,
// type-check, and print one or more Go packages.
// Its precise output is unspecified and may change.
package main

import (
	"flag"
	"fmt"
	"go/types"
	"log"
	"os"
	"runtime"
	"runtime/pprof"
	"runtime/trace"
	"sort"
	"strings"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/types/typeutil"
)

// flags
var (
	depsFlag = flag.Bool("deps", false, "show dependencies too")
	testFlag = flag.Bool("test", false, "include any tests implied by the patterns")
	cgoFlag  = flag.Bool("cgo", true, "process cgo files")
	mode     = flag.String("mode", "metadata", "mode (one of metadata, typecheck, wholeprogram)")
	private  = flag.Bool("private", false, "show non-exported declarations too")

	cpuprofile = flag.String("cpuprofile", "", "write CPU profile to this file")
	memprofile = flag.String("memprofile", "", "write memory profile to this file")
	traceFlag  = flag.String("trace", "", "write trace log to this file")
)

func usage() {
	fmt.Fprintln(os.Stderr, `Usage: gopackages [-deps] [-cgo] [-mode=...] [-private] package...

The gopackages command loads, parses, type-checks,
and prints one or more Go packages.

Packages are specified using the notation of "go list",
or other underlying build system.

Flags:`)
	flag.PrintDefaults()
}

func main() {
	log.SetPrefix("gopackages: ")
	log.SetFlags(0)
	flag.Usage = usage
	flag.Parse()

	if len(flag.Args()) == 0 {
		usage()
		os.Exit(1)
	}

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatal(err)
		}
		// NB: profile won't be written in case of error.
		defer pprof.StopCPUProfile()
	}

	if *traceFlag != "" {
		f, err := os.Create(*traceFlag)
		if err != nil {
			log.Fatal(err)
		}
		if err := trace.Start(f); err != nil {
			log.Fatal(err)
		}
		// NB: trace log won't be written in case of error.
		defer func() {
			trace.Stop()
			log.Printf("To view the trace, run:\n$ go tool trace view %s", *traceFlag)
		}()
	}

	if *memprofile != "" {
		f, err := os.Create(*memprofile)
		if err != nil {
			log.Fatal(err)
		}
		// NB: memprofile won't be written in case of error.
		defer func() {
			runtime.GC() // get up-to-date statistics
			if err := pprof.WriteHeapProfile(f); err != nil {
				log.Fatalf("Writing memory profile: %v", err)
			}
			f.Close()
		}()
	}

	// -mode flag
	load := packages.TypeCheck
	switch strings.ToLower(*mode) {
	case "metadata":
		load = packages.Metadata
	case "typecheck":
		load = packages.TypeCheck
	case "wholeprogram":
		load = packages.WholeProgram
	default:
		log.Fatalf("invalid mode: %s", *mode)
	}

	// Load, parse, and type-check the packages named on the command line.
	opts := &packages.Options{
		Error:      func(error) {}, // we'll take responsibility for printing errors
		DisableCgo: !*cgoFlag,
		Tests:      *testFlag,
	}
	lpkgs, err := load(opts, flag.Args()...)
	if err != nil {
		log.Fatal(err)
	}

	// -deps: print dependencies too.
	if *depsFlag {
		// We can't use packages.All because
		// we need an ordered traversal.
		var all []*packages.Package // postorder
		seen := make(map[*packages.Package]bool)
		var visit func(*packages.Package)
		visit = func(lpkg *packages.Package) {
			if !seen[lpkg] {
				seen[lpkg] = true

				// visit imports
				var importPaths []string
				for path := range lpkg.Imports {
					importPaths = append(importPaths, path)
				}
				sort.Strings(importPaths) // for determinism
				for _, path := range importPaths {
					visit(lpkg.Imports[path])
				}

				all = append(all, lpkg)
			}
		}
		for _, lpkg := range lpkgs {
			visit(lpkg)
		}
		lpkgs = all
	}

	for _, lpkg := range lpkgs {
		print(lpkg)
	}
}

func print(lpkg *packages.Package) {
	// title
	var kind string
	// TODO(matloob): If IsTest is added back print "test command" or
	// "test package" for packages with IsTest == true.
	if lpkg.Name == "main" {
		kind += "command"
	} else {
		kind += "package"
	}
	fmt.Printf("Go %s %q:\n", kind, lpkg.ID) // unique ID
	fmt.Printf("\tpackage %s\n", lpkg.Name)
	fmt.Printf("\treflect.Type.PkgPath %q\n", lpkg.PkgPath)

	// characterize type info
	if lpkg.Type == nil {
		fmt.Printf("\thas no exported type info\n")
	} else if !lpkg.Type.Complete() {
		fmt.Printf("\thas incomplete exported type info\n")
	} else if len(lpkg.Files) == 0 {
		fmt.Printf("\thas complete exported type info\n")
	} else {
		fmt.Printf("\thas complete exported type info and typed ASTs\n")
	}
	if lpkg.Type != nil && lpkg.IllTyped && len(lpkg.Errors) == 0 {
		fmt.Printf("\thas an error among its dependencies\n")
	}

	// source files
	for _, src := range lpkg.Srcs {
		fmt.Printf("\tfile %s\n", src)
	}

	// imports
	var lines []string
	for importPath, imp := range lpkg.Imports {
		var line string
		if imp.ID == importPath {
			line = fmt.Sprintf("\timport %q", importPath)
		} else {
			line = fmt.Sprintf("\timport %q => %q", importPath, imp.ID)
		}
		lines = append(lines, line)
	}
	sort.Strings(lines)
	for _, line := range lines {
		fmt.Println(line)
	}

	// errors
	for _, err := range lpkg.Errors {
		fmt.Printf("\t%s\n", err)
	}

	// package members (TypeCheck or WholeProgram mode)
	if lpkg.Type != nil {
		qual := types.RelativeTo(lpkg.Type)
		scope := lpkg.Type.Scope()
		for _, name := range scope.Names() {
			obj := scope.Lookup(name)
			if !obj.Exported() && !*private {
				continue // skip unexported names
			}

			fmt.Printf("\t%s\n", types.ObjectString(obj, qual))
			if _, ok := obj.(*types.TypeName); ok {
				for _, meth := range typeutil.IntuitiveMethodSet(obj.Type(), nil) {
					if !meth.Obj().Exported() && !*private {
						continue // skip unexported names
					}
					fmt.Printf("\t%s\n", types.SelectionString(meth, qual))
				}
			}
		}
	}

	fmt.Println()
}
