// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/internal/cov"
	"cmd/internal/pkgpattern"
	"flag"
	"fmt"
	"os"
	"runtime"
	"runtime/pprof"
	"strings"
)

var verbflag = flag.Int("v", 0, "Verbose trace output level")
var hflag = flag.Bool("h", false, "Panic on fatal errors (for stack trace)")
var hwflag = flag.Bool("hw", false, "Panic on warnings (for stack trace)")
var indirsflag = flag.String("i", "", "Input dirs to examine (comma separated)")
var pkgpatflag = flag.String("pkg", "", "Restrict output to package(s) matching specified package pattern.")
var cpuprofileflag = flag.String("cpuprofile", "", "Write CPU profile to specified file")
var memprofileflag = flag.String("memprofile", "", "Write memory profile to specified file")
var memprofilerateflag = flag.Int("memprofilerate", 0, "Set memprofile sampling rate to value")

var matchpkg func(name string) bool

var atExitFuncs []func()

func atExit(f func()) {
	atExitFuncs = append(atExitFuncs, f)
}

func Exit(code int) {
	for i := len(atExitFuncs) - 1; i >= 0; i-- {
		f := atExitFuncs[i]
		atExitFuncs = atExitFuncs[:i]
		f()
	}
	os.Exit(code)
}

func dbgtrace(vlevel int, s string, a ...interface{}) {
	if *verbflag >= vlevel {
		fmt.Printf(s, a...)
		fmt.Printf("\n")
	}
}

func warn(s string, a ...interface{}) {
	fmt.Fprintf(os.Stderr, "warning: ")
	fmt.Fprintf(os.Stderr, s, a...)
	fmt.Fprintf(os.Stderr, "\n")
	if *hwflag {
		panic("unexpected warning")
	}
}

func fatal(s string, a ...interface{}) {
	fmt.Fprintf(os.Stderr, "error: ")
	fmt.Fprintf(os.Stderr, s, a...)
	fmt.Fprintf(os.Stderr, "\n")
	if *hflag {
		panic("fatal error")
	}
	Exit(1)
}

func usage(msg string) {
	if len(msg) > 0 {
		fmt.Fprintf(os.Stderr, "error: %s\n", msg)
	}
	fmt.Fprintf(os.Stderr, "usage: go tool covdata [command]\n")
	fmt.Fprintf(os.Stderr, `
Commands are:

textfmt     convert coverage data to textual format
percent     output total percentage of statements covered
pkglist     output list of package import paths
func        output coverage profile information for each function
merge       merge data files together
subtract    subtract one set of data files from another set
intersect   generate intersection of two sets of data files
debugdump   dump data in human-readable format for debugging purposes
`)
	fmt.Fprintf(os.Stderr, "\nFor help on a specific subcommand, try:\n")
	fmt.Fprintf(os.Stderr, "\ngo tool covdata <cmd> -help\n")
	Exit(2)
}

type covOperation interface {
	cov.CovDataVisitor
	Setup()
	Usage(string)
}

// Modes of operation.
const (
	funcMode      = "func"
	mergeMode     = "merge"
	intersectMode = "intersect"
	subtractMode  = "subtract"
	percentMode   = "percent"
	pkglistMode   = "pkglist"
	textfmtMode   = "textfmt"
	debugDumpMode = "debugdump"
)

func main() {
	// First argument should be mode/subcommand.
	if len(os.Args) < 2 {
		usage("missing command selector")
	}

	// Select mode
	var op covOperation
	cmd := os.Args[1]
	switch cmd {
	case mergeMode:
		op = makeMergeOp()
	case debugDumpMode:
		op = makeDumpOp(debugDumpMode)
	case textfmtMode:
		op = makeDumpOp(textfmtMode)
	case percentMode:
		op = makeDumpOp(percentMode)
	case funcMode:
		op = makeDumpOp(funcMode)
	case pkglistMode:
		op = makeDumpOp(pkglistMode)
	case subtractMode:
		op = makeSubtractIntersectOp(subtractMode)
	case intersectMode:
		op = makeSubtractIntersectOp(intersectMode)
	default:
		usage(fmt.Sprintf("unknown command selector %q", cmd))
	}

	// Edit out command selector, then parse flags.
	os.Args = append(os.Args[:1], os.Args[2:]...)
	flag.Usage = func() {
		op.Usage("")
	}
	flag.Parse()

	// Mode-independent flag setup
	dbgtrace(1, "starting mode-independent setup")
	if flag.NArg() != 0 {
		op.Usage("unknown extra arguments")
	}
	if *pkgpatflag != "" {
		pats := strings.Split(*pkgpatflag, ",")
		matchers := []func(name string) bool{}
		for _, p := range pats {
			if p == "" {
				continue
			}
			f := pkgpattern.MatchSimplePattern(p)
			matchers = append(matchers, f)
		}
		matchpkg = func(name string) bool {
			for _, f := range matchers {
				if f(name) {
					return true
				}
			}
			return false
		}
	}
	if *cpuprofileflag != "" {
		f, err := os.Create(*cpuprofileflag)
		if err != nil {
			fatal("%v", err)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			fatal("%v", err)
		}
		atExit(func() {
			pprof.StopCPUProfile()
			if err = f.Close(); err != nil {
				fatal("error closing cpu profile: %v", err)
			}
		})
	}
	if *memprofileflag != "" {
		if *memprofilerateflag != 0 {
			runtime.MemProfileRate = *memprofilerateflag
		}
		f, err := os.Create(*memprofileflag)
		if err != nil {
			fatal("%v", err)
		}
		atExit(func() {
			runtime.GC()
			const writeLegacyFormat = 1
			if err := pprof.Lookup("heap").WriteTo(f, writeLegacyFormat); err != nil {
				fatal("%v", err)
			}
			if err = f.Close(); err != nil {
				fatal("error closing memory profile: %v", err)
			}
		})
	} else {
		// Not doing memory profiling; disable it entirely.
		runtime.MemProfileRate = 0
	}

	// Mode-dependent setup.
	op.Setup()

	// ... off and running now.
	dbgtrace(1, "starting perform")

	indirs := strings.Split(*indirsflag, ",")
	vis := cov.CovDataVisitor(op)
	var flags cov.CovDataReaderFlags
	if *hflag {
		flags |= cov.PanicOnError
	}
	if *hwflag {
		flags |= cov.PanicOnWarning
	}
	reader := cov.MakeCovDataReader(vis, indirs, *verbflag, flags, matchpkg)
	st := 0
	if err := reader.Visit(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		st = 1
	}
	dbgtrace(1, "leaving main")
	Exit(st)
}
