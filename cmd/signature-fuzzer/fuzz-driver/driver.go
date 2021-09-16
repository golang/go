// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Stand-alone driver for emitting function-signature test code.  This
// program is mainly just a wrapper around the code that lives in the
// fuzz-generator package; it is useful for generating a specific bad
// code scenario for a given seed, or for doing development on the
// fuzzer, but for doing actual fuzz testing, better to use
// fuzz-runner.

package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	generator "golang.org/x/tools/cmd/signature-fuzzer/internal/fuzz-generator"
)

// Basic options
var numfcnflag = flag.Int("numfcns", 10, "Number of test func pairs to emit in each package")
var numpkgflag = flag.Int("numpkgs", 1, "Number of test packages to emit")
var seedflag = flag.Int64("seed", -1, "Random seed")
var tagflag = flag.String("tag", "gen", "Prefix name of go files/pkgs to generate")
var outdirflag = flag.String("outdir", "", "Output directory for generated files")
var pkgpathflag = flag.String("pkgpath", "gen", "Base package path for generated files")

// Options used for test case minimization.
var fcnmaskflag = flag.String("fcnmask", "", "Mask containing list of fcn numbers to emit")
var pkmaskflag = flag.String("pkgmask", "", "Mask containing list of pkg numbers to emit")

// Options used to control which features are used in the generated code.
var reflectflag = flag.Bool("reflect", true, "Include testing of reflect.Call.")
var deferflag = flag.Bool("defer", true, "Include testing of defer stmts.")
var recurflag = flag.Bool("recur", true, "Include testing of recursive calls.")
var takeaddrflag = flag.Bool("takeaddr", true, "Include functions that take the address of their parameters and results.")
var methodflag = flag.Bool("method", true, "Include testing of method calls.")
var inlimitflag = flag.Int("inmax", -1, "Max number of input params.")
var outlimitflag = flag.Int("outmax", -1, "Max number of input params.")
var pragmaflag = flag.String("pragma", "", "Tag generated test routines with pragma //go:<value>.")
var maxfailflag = flag.Int("maxfail", 10, "Maximum runtime failures before test self-terminates")
var stackforceflag = flag.Bool("forcestackgrowth", true, "Use hooks to force stack growth.")

// Debugging options
var verbflag = flag.Int("v", 0, "Verbose trace output level")

// Debugging/testing options. These tell the generator to emit "bad" code so as to
// test the logic for detecting errors and/or minimization (in the fuzz runner).
var emitbadflag = flag.Int("emitbad", 0, "[Testing only] force generator to emit 'bad' code.")
var selbadpkgflag = flag.Int("badpkgidx", 0, "[Testing only] select index of bad package (used with -emitbad)")
var selbadfcnflag = flag.Int("badfcnidx", 0, "[Testing only] select index of bad function (used with -emitbad)")

// Misc options
var goimpflag = flag.Bool("goimports", false, "Run 'goimports' on generated code.")
var randctlflag = flag.Int("randctl", generator.RandCtlChecks|generator.RandCtlPanic, "Wraprand control flag")

func verb(vlevel int, s string, a ...interface{}) {
	if *verbflag >= vlevel {
		fmt.Printf(s, a...)
		fmt.Printf("\n")
	}
}

func usage(msg string) {
	if len(msg) > 0 {
		fmt.Fprintf(os.Stderr, "error: %s\n", msg)
	}
	fmt.Fprintf(os.Stderr, "usage: fuzz-driver [flags]\n\n")
	flag.PrintDefaults()
	fmt.Fprintf(os.Stderr, "Example:\n\n")
	fmt.Fprintf(os.Stderr, "  fuzz-driver -numpkgs=23 -numfcns=19 -seed 10101 -outdir gendir\n\n")
	fmt.Fprintf(os.Stderr, "  \tgenerates a Go program with 437 test cases (23 packages, each \n")
	fmt.Fprintf(os.Stderr, "  \twith 19 functions, for a total of 437 funcs total) into a set of\n")
	fmt.Fprintf(os.Stderr, "  \tsub-directories in 'gendir', using random see 10101\n")

	os.Exit(2)
}

func setupTunables() {
	tunables := generator.DefaultTunables()
	if !*reflectflag {
		tunables.DisableReflectionCalls()
	}
	if !*deferflag {
		tunables.DisableDefer()
	}
	if !*recurflag {
		tunables.DisableRecursiveCalls()
	}
	if !*takeaddrflag {
		tunables.DisableTakeAddr()
	}
	if !*methodflag {
		tunables.DisableMethodCalls()
	}
	if *inlimitflag != -1 {
		tunables.LimitInputs(*inlimitflag)
	}
	if *outlimitflag != -1 {
		tunables.LimitOutputs(*outlimitflag)
	}
	generator.SetTunables(tunables)
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("fuzz-driver: ")
	flag.Parse()
	generator.Verbctl = *verbflag
	if *outdirflag == "" {
		usage("select an output directory with -o flag")
	}
	verb(1, "in main verblevel=%d", *verbflag)
	if *seedflag == -1 {
		// user has not selected a specific seed -- pick one.
		now := time.Now()
		*seedflag = now.UnixNano() % 123456789
		verb(0, "selected seed: %d", *seedflag)
	}
	rand.Seed(*seedflag)
	if flag.NArg() != 0 {
		usage("unknown extra arguments")
	}
	verb(1, "tag is %s", *tagflag)

	fcnmask, err := generator.ParseMaskString(*fcnmaskflag, "fcn")
	if err != nil {
		usage(fmt.Sprintf("mangled fcn mask arg: %v", err))
	}
	pkmask, err := generator.ParseMaskString(*pkmaskflag, "pkg")
	if err != nil {
		usage(fmt.Sprintf("mangled pkg mask arg: %v", err))
	}
	verb(2, "pkg mask is %v", pkmask)
	verb(2, "fn mask is %v", fcnmask)

	verb(1, "starting generation")
	setupTunables()
	config := generator.GenConfig{
		PkgPath:          *pkgpathflag,
		Tag:              *tagflag,
		OutDir:           *outdirflag,
		NumTestPackages:  *numpkgflag,
		NumTestFunctions: *numfcnflag,
		Seed:             *seedflag,
		Pragma:           *pragmaflag,
		FcnMask:          fcnmask,
		PkgMask:          pkmask,
		MaxFail:          *maxfailflag,
		ForceStackGrowth: *stackforceflag,
		RandCtl:          *randctlflag,
		RunGoImports:     *goimpflag,
		EmitBad:          *emitbadflag,
		BadPackageIdx:    *selbadpkgflag,
		BadFuncIdx:       *selbadfcnflag,
	}
	errs := generator.Generate(config)
	if errs != 0 {
		log.Fatal("errors during generation")
	}
	verb(1, "... files written to directory %s", *outdirflag)
	verb(1, "leaving main")
}
