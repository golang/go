// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// ssadump: a tool for displaying and interpreting the SSA form of Go programs.
package main // import "golang.org/x/tools/cmd/ssadump"

import (
	"flag"
	"fmt"
	"go/build"
	"go/types"
	"os"
	"runtime"
	"runtime/pprof"

	"golang.org/x/tools/go/buildutil"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/interp"
	"golang.org/x/tools/go/ssa/ssautil"
)

// flags
var (
	mode = ssa.BuilderMode(0)

	testFlag = flag.Bool("test", false, "Loads test code (*_test.go) for imported packages.")

	runFlag = flag.Bool("run", false, "Invokes the SSA interpreter on the program.")

	interpFlag = flag.String("interp", "", `Options controlling the SSA test interpreter.
The value is a sequence of zero or more more of these letters:
R	disable [R]ecover() from panic; show interpreter crash instead.
T	[T]race execution of the program.  Best for single-threaded programs!
`)

	cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")
)

func init() {
	flag.Var(&mode, "build", ssa.BuilderModeDoc)
	flag.Var((*buildutil.TagsFlag)(&build.Default.BuildTags), "tags", buildutil.TagsFlagDoc)
}

const usage = `SSA builder and interpreter.
Usage: ssadump [<flag> ...] <args> ...
Use -help flag to display options.

Examples:
% ssadump -build=F hello.go              # dump SSA form of a single package
% ssadump -run -interp=T hello.go        # interpret a program, with tracing
% ssadump -run -test unicode -- -test.v  # interpret the unicode package's tests, verbosely
` + loader.FromArgsUsage +
	`
When -run is specified, ssadump will run the program.
The entry point depends on the -test flag:
if clear, it runs the first package named main.
if set, it runs the tests of each package.
`

func main() {
	if err := doMain(); err != nil {
		fmt.Fprintf(os.Stderr, "ssadump: %s\n", err)
		os.Exit(1)
	}
}

func doMain() error {
	flag.Parse()
	args := flag.Args()

	conf := loader.Config{Build: &build.Default}

	// Choose types.Sizes from conf.Build.
	var wordSize int64 = 8
	switch conf.Build.GOARCH {
	case "386", "arm":
		wordSize = 4
	}
	conf.TypeChecker.Sizes = &types.StdSizes{
		MaxAlign: 8,
		WordSize: wordSize,
	}

	var interpMode interp.Mode
	for _, c := range *interpFlag {
		switch c {
		case 'T':
			interpMode |= interp.EnableTracing
		case 'R':
			interpMode |= interp.DisableRecover
		default:
			return fmt.Errorf("unknown -interp option: '%c'", c)
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
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	// Use the initial packages from the command line.
	args, err := conf.FromArgs(args, *testFlag)
	if err != nil {
		return err
	}

	// The interpreter needs the runtime package.
	if *runFlag {
		conf.Import("runtime")
	}

	// Load, parse and type-check the whole program.
	lprog, err := conf.Load()
	if err != nil {
		return err
	}

	// Create and build SSA-form program representation.
	prog := ssautil.CreateProgram(lprog, mode)

	// Build and display only the initial packages
	// (and synthetic wrappers), unless -run is specified.
	var initpkgs []*ssa.Package
	for _, info := range lprog.InitialPackages() {
		ssapkg := prog.Package(info.Pkg)
		ssapkg.Build()
		if info.Pkg.Path() != "runtime" {
			initpkgs = append(initpkgs, ssapkg)
		}
	}

	// Run the interpreter.
	if *runFlag {
		prog.Build()

		var mains []*ssa.Package
		if *testFlag {
			// If -test, run the tests.
			for _, pkg := range initpkgs {
				if main := prog.CreateTestMainPackage(pkg); main != nil {
					mains = append(mains, main)
				}
			}
			if mains == nil {
				return fmt.Errorf("no tests")
			}
		} else {
			// Otherwise, run the main packages.
			mains = ssautil.MainPackages(initpkgs)
			if len(mains) == 0 {
				return fmt.Errorf("no main package")
			}
		}

		if runtime.GOARCH != build.Default.GOARCH {
			return fmt.Errorf("cross-interpretation is not supported (target has GOARCH %s, interpreter has %s)",
				build.Default.GOARCH, runtime.GOARCH)
		}

		for _, main := range mains {
			if len(mains) > 1 {
				fmt.Fprintf(os.Stderr, "Running: %s\n", main.Pkg.Path())
			}
			interp.Interpret(main, interpMode, conf.TypeChecker.Sizes, main.Pkg.Path(), args)
		}
	}
	return nil
}
