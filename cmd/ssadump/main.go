// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// ssadump: a tool for displaying and interpreting the SSA form of Go programs.
package main

import (
	"flag"
	"fmt"
	"go/build"
	"log"
	"os"
	"runtime"
	"runtime/pprof"

	"code.google.com/p/go.tools/importer"
	"code.google.com/p/go.tools/ssa"
	"code.google.com/p/go.tools/ssa/interp"
)

var buildFlag = flag.String("build", "", `Options controlling the SSA builder.
The value is a sequence of zero or more of these letters:
C	perform sanity [C]hecking of the SSA form.
D	include [D]ebug info for every function.
P	log [P]ackage inventory.
F	log [F]unction SSA code.
S	log [S]ource locations as SSA builder progresses.
G	use binary object files from gc to provide imports (no code).
L	build distinct packages seria[L]ly instead of in parallel.
N	build [N]aive SSA form: don't replace local loads/stores with registers.
`)

var runFlag = flag.Bool("run", false, "Invokes the SSA interpreter on the program.")

var interpFlag = flag.String("interp", "", `Options controlling the SSA test interpreter.
The value is a sequence of zero or more more of these letters:
R	disable [R]ecover() from panic; show interpreter crash instead.
T	[T]race execution of the program.  Best for single-threaded programs!
`)

const usage = `SSA builder and interpreter.
Usage: ssadump [<flag> ...] [<file.go> ...] [<arg> ...]
       ssadump [<flag> ...] <import/path>   [<arg> ...]
Use -help flag to display options.

Examples:
% ssadump -run -interp=T hello.go     # interpret a program, with tracing
% ssadump -build=FPG hello.go         # quickly dump SSA form of a single package
`

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

func init() {
	// If $GOMAXPROCS isn't set, use the full capacity of the machine.
	// For small machines, use at least 4 threads.
	if os.Getenv("GOMAXPROCS") == "" {
		n := runtime.NumCPU()
		if n < 4 {
			n = 4
		}
		runtime.GOMAXPROCS(n)
	}
}

func main() {
	flag.Parse()
	args := flag.Args()

	impctx := importer.Config{Build: &build.Default}

	var debugMode bool
	var mode ssa.BuilderMode
	for _, c := range *buildFlag {
		switch c {
		case 'D':
			debugMode = true
		case 'P':
			mode |= ssa.LogPackages | ssa.BuildSerially
		case 'F':
			mode |= ssa.LogFunctions | ssa.BuildSerially
		case 'S':
			mode |= ssa.LogSource | ssa.BuildSerially
		case 'C':
			mode |= ssa.SanityCheckFunctions
		case 'N':
			mode |= ssa.NaiveForm
		case 'G':
			impctx.Build = nil
		case 'L':
			mode |= ssa.BuildSerially
		default:
			log.Fatalf("Unknown -build option: '%c'.", c)
		}
	}

	var interpMode interp.Mode
	for _, c := range *interpFlag {
		switch c {
		case 'T':
			interpMode |= interp.EnableTracing
		case 'R':
			interpMode |= interp.DisableRecover
		default:
			log.Fatalf("Unknown -interp option: '%c'.", c)
		}
	}

	if len(args) == 0 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(1)
	}

	// Profiling support.
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	// Load, parse and type-check the program.
	imp := importer.New(&impctx)
	infos, args, err := imp.LoadInitialPackages(args)
	if err != nil {
		log.Fatal(err)
	}

	// Create and build SSA-form program representation.
	prog := ssa.NewProgram(imp.Fset, mode)
	if err := prog.CreatePackages(imp); err != nil {
		log.Fatal(err)
	}
	if debugMode {
		for _, pkg := range prog.AllPackages() {
			pkg.SetDebugMode(true)
		}
	}
	prog.BuildAll()

	// Run the interpreter on the first package with a main function.
	if *runFlag {
		var main *ssa.Package
		for _, info := range infos {
			pkg := prog.Package(info.Pkg)
			if pkg.Func("main") != nil || pkg.CreateTestMainFunction() != nil {
				main = pkg
				break
			}
		}
		if main == nil {
			log.Fatal("No main function")
		}
		interp.Interpret(main, interpMode, main.Object.Path(), args)
	}
}
