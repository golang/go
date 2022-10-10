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
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/interp"
	"golang.org/x/tools/go/ssa/ssautil"
)

// flags
var (
	mode = ssa.BuilderMode(0)

	testFlag = flag.Bool("test", false, "include implicit test packages and executables")

	runFlag = flag.Bool("run", false, "interpret the SSA program")

	interpFlag = flag.String("interp", "", `Options controlling the SSA test interpreter.
The value is a sequence of zero or more more of these letters:
R	disable [R]ecover() from panic; show interpreter crash instead.
T	[T]race execution of the program.  Best for single-threaded programs!
`)

	cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

	args stringListValue
)

func init() {
	flag.Var(&mode, "build", ssa.BuilderModeDoc)
	flag.Var((*buildutil.TagsFlag)(&build.Default.BuildTags), "tags", buildutil.TagsFlagDoc)
	flag.Var(&args, "arg", "add argument to interpreted program")
}

const usage = `SSA builder and interpreter.
Usage: ssadump [-build=[DBCSNFLG]] [-test] [-run] [-interp=[TR]] [-arg=...] package...
Use -help flag to display options.

Examples:
% ssadump -build=F hello.go              # dump SSA form of a single package
% ssadump -build=F -test fmt             # dump SSA form of a package and its tests
% ssadump -run -interp=T hello.go        # interpret a program, with tracing

The -run flag causes ssadump to build the code in a runnable form and run the first
package named main.

Interpretation of the standard "testing" package is no longer supported.
`

func main() {
	if err := doMain(); err != nil {
		fmt.Fprintf(os.Stderr, "ssadump: %s\n", err)
		os.Exit(1)
	}
}

func doMain() error {
	flag.Parse()
	if len(flag.Args()) == 0 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(1)
	}

	cfg := &packages.Config{
		Mode:  packages.LoadSyntax,
		Tests: *testFlag,
	}

	// Choose types.Sizes from conf.Build.
	// TODO(adonovan): remove this when go/packages provides a better way.
	var wordSize int64 = 8
	switch build.Default.GOARCH {
	case "386", "arm":
		wordSize = 4
	}
	sizes := &types.StdSizes{
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

	// Load, parse and type-check the initial packages,
	// and, if -run, their dependencies.
	if *runFlag {
		cfg.Mode = packages.LoadAllSyntax
	}
	initial, err := packages.Load(cfg, flag.Args()...)
	if err != nil {
		return err
	}
	if len(initial) == 0 {
		return fmt.Errorf("no packages")
	}
	if packages.PrintErrors(initial) > 0 {
		return fmt.Errorf("packages contain errors")
	}

	// Turn on instantiating generics during build if the program will be run.
	if *runFlag {
		mode |= ssa.InstantiateGenerics
	}

	// Create SSA-form program representation.
	prog, pkgs := ssautil.AllPackages(initial, mode)

	for i, p := range pkgs {
		if p == nil {
			return fmt.Errorf("cannot build SSA for package %s", initial[i])
		}
	}

	if !*runFlag {
		// Build and display only the initial packages
		// (and synthetic wrappers).
		for _, p := range pkgs {
			p.Build()
		}

	} else {
		// Run the interpreter.
		// Build SSA for all packages.
		prog.Build()

		// Earlier versions of the interpreter needed the runtime
		// package; however, interp cannot handle unsafe constructs
		// used during runtime's package initialization at the moment.
		// The key construct blocking support is:
		//    *((*T)(unsafe.Pointer(p)))
		// Unfortunately, this means only trivial programs can be
		// interpreted by ssadump.
		if prog.ImportedPackage("runtime") != nil {
			return fmt.Errorf("-run: program depends on runtime package (interpreter can run only trivial programs)")
		}

		if runtime.GOARCH != build.Default.GOARCH {
			return fmt.Errorf("cross-interpretation is not supported (target has GOARCH %s, interpreter has %s)",
				build.Default.GOARCH, runtime.GOARCH)
		}

		// Run first main package.
		for _, main := range ssautil.MainPackages(pkgs) {
			fmt.Fprintf(os.Stderr, "Running: %s\n", main.Pkg.Path())
			os.Exit(interp.Interpret(main, interpMode, sizes, main.Pkg.Path(), args))
		}
		return fmt.Errorf("no main package")
	}
	return nil
}

// stringListValue is a flag.Value that accumulates strings.
// e.g. --flag=one --flag=two would produce []string{"one", "two"}.
type stringListValue []string

func newStringListValue(val []string, p *[]string) *stringListValue {
	*p = val
	return (*stringListValue)(p)
}

func (ss *stringListValue) Get() interface{} { return []string(*ss) }

func (ss *stringListValue) String() string { return fmt.Sprintf("%q", *ss) }

func (ss *stringListValue) Set(s string) error { *ss = append(*ss, s); return nil }
