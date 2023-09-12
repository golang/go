// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Program for performing test runs using "fuzz-driver".
// Main loop iteratively runs "fuzz-driver" to create a corpus,
// then builds and runs the code. If a failure in the run is
// detected, then a testcase minimization phase kicks in.

package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	generator "golang.org/x/tools/cmd/signature-fuzzer/internal/fuzz-generator"
)

const pkName = "fzTest"

// Basic options
var verbflag = flag.Int("v", 0, "Verbose trace output level")
var loopitflag = flag.Int("numit", 10, "Number of main loop iterations to run")
var seedflag = flag.Int64("seed", -1, "Random seed")
var execflag = flag.Bool("execdriver", false, "Exec fuzz-driver binary instead of invoking generator directly")
var numpkgsflag = flag.Int("numpkgs", 50, "Number of test packages")
var numfcnsflag = flag.Int("numfcns", 20, "Number of test functions per package.")

// Debugging/testing options. These tell the generator to emit "bad" code so as to
// test the logic for detecting errors and/or minimization.
var emitbadflag = flag.Int("emitbad", -1, "[Testing only] force generator to emit 'bad' code.")
var selbadpkgflag = flag.Int("badpkgidx", 0, "[Testing only] select index of bad package (used with -emitbad)")
var selbadfcnflag = flag.Int("badfcnidx", 0, "[Testing only] select index of bad function (used with -emitbad)")
var forcetmpcleanflag = flag.Bool("forcetmpclean", false, "[Testing only] force cleanup of temp dir")
var cleancacheflag = flag.Bool("cleancache", true, "[Testing only] don't clean the go cache")
var raceflag = flag.Bool("race", false, "[Testing only] build generated code with -race")

func verb(vlevel int, s string, a ...interface{}) {
	if *verbflag >= vlevel {
		fmt.Printf(s, a...)
		fmt.Printf("\n")
	}
}

func warn(s string, a ...interface{}) {
	fmt.Fprintf(os.Stderr, s, a...)
	fmt.Fprintf(os.Stderr, "\n")
}

func fatal(s string, a ...interface{}) {
	fmt.Fprintf(os.Stderr, s, a...)
	fmt.Fprintf(os.Stderr, "\n")
	os.Exit(1)
}

type config struct {
	generator.GenConfig
	tmpdir       string
	gendir       string
	buildOutFile string
	runOutFile   string
	gcflags      string
	nerrors      int
}

func usage(msg string) {
	if len(msg) > 0 {
		fmt.Fprintf(os.Stderr, "error: %s\n", msg)
	}
	fmt.Fprintf(os.Stderr, "usage: fuzz-runner [flags]\n\n")
	flag.PrintDefaults()
	fmt.Fprintf(os.Stderr, "Example:\n\n")
	fmt.Fprintf(os.Stderr, "  fuzz-runner -numit=500 -numpkgs=11 -numfcns=13 -seed=10101\n\n")
	fmt.Fprintf(os.Stderr, "  \tRuns 500 rounds of test case generation\n")
	fmt.Fprintf(os.Stderr, "  \tusing random see 10101, in each round emitting\n")
	fmt.Fprintf(os.Stderr, "  \t11 packages each with 13 function pairs.\n")

	os.Exit(2)
}

// docmd executes the specified command in the dir given and pipes the
// output to stderr. return status is 0 if command passed, 1
// otherwise.
func docmd(cmd []string, dir string) int {
	verb(2, "docmd: %s", strings.Join(cmd, " "))
	c := exec.Command(cmd[0], cmd[1:]...)
	if dir != "" {
		c.Dir = dir
	}
	b, err := c.CombinedOutput()
	st := 0
	if err != nil {
		warn("error executing cmd %s: %v",
			strings.Join(cmd, " "), err)
		st = 1
	}
	os.Stderr.Write(b)
	return st
}

// docmdout forks and execs command 'cmd' in dir 'dir', redirecting
// stderr and stdout from the execution to file 'outfile'.
func docmdout(cmd []string, dir string, outfile string) int {
	of, err := os.OpenFile(outfile, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		fatal("opening outputfile %s: %v", outfile, err)
	}
	c := exec.Command(cmd[0], cmd[1:]...)
	defer of.Close()
	if dir != "" {
		verb(2, "setting cmd.Dir to %s", dir)
		c.Dir = dir
	}
	verb(2, "docmdout: %s > %s", strings.Join(cmd, " "), outfile)
	c.Stdout = of
	c.Stderr = of
	err = c.Run()
	st := 0
	if err != nil {
		warn("error executing cmd %s: %v",
			strings.Join(cmd, " "), err)
		st = 1
	}
	return st
}

// gen is the main hook for kicking off code generation. For
// non-minimization runs, 'singlepk' and 'singlefn' will both be -1
// (indicating that we want all functions and packages to be
// generated).  If 'singlepk' is set to a non-negative value, then
// code generation will be restricted to the single package with that
// index (as a try at minimization), similarly with 'singlefn'
// restricting the codegen to a single specified function.
func (c *config) gen(singlepk int, singlefn int) {

	// clean the output dir
	verb(2, "cleaning outdir %s", c.gendir)
	if err := os.RemoveAll(c.gendir); err != nil {
		fatal("error cleaning gen dir %s: %v", c.gendir, err)
	}

	// emit code into the output dir. Here we either invoke the
	// generator directly, or invoke fuzz-driver if -execflag is
	// set.  If the code generation process itself fails, this is
	// typically a bug in the fuzzer itself, so it gets reported
	// as a fatal error.
	if *execflag {
		args := []string{"fuzz-driver",
			"-numpkgs", strconv.Itoa(c.NumTestPackages),
			"-numfcns", strconv.Itoa(c.NumTestFunctions),
			"-seed", strconv.Itoa(int(c.Seed)),
			"-outdir", c.OutDir,
			"-pkgpath", pkName,
			"-maxfail", strconv.Itoa(c.MaxFail)}
		if singlepk != -1 {
			args = append(args, "-pkgmask", strconv.Itoa(singlepk))
		}
		if singlefn != -1 {
			args = append(args, "-fcnmask", strconv.Itoa(singlefn))
		}
		if *emitbadflag != 0 {
			args = append(args, "-emitbad", strconv.Itoa(*emitbadflag),
				"-badpkgidx", strconv.Itoa(*selbadpkgflag),
				"-badfcnidx", strconv.Itoa(*selbadfcnflag))
		}
		verb(1, "invoking fuzz-driver with args: %v", args)
		st := docmd(args, "")
		if st != 0 {
			fatal("fatal error: generation failed, cmd was: %v", args)
		}
	} else {
		if singlepk != -1 {
			c.PkgMask = map[int]int{singlepk: 1}
		}
		if singlefn != -1 {
			c.FcnMask = map[int]int{singlefn: 1}
		}
		verb(1, "invoking generator.Generate with config: %v", c.GenConfig)
		errs := generator.Generate(c.GenConfig)
		if errs != 0 {
			log.Fatal("errors during generation")
		}
	}
}

// action performs a selected action/command in the generated code dir.
func (c *config) action(cmd []string, outfile string, emitout bool) int {
	st := docmdout(cmd, c.gendir, outfile)
	if emitout {
		content, err := os.ReadFile(outfile)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Fprintf(os.Stderr, "%s", content)
	}
	return st
}

func binaryName() string {
	if runtime.GOOS == "windows" {
		return pkName + ".exe"
	} else {
		return "./" + pkName
	}
}

// build builds a generated corpus of Go code. If 'emitout' is set, then dump out the
// results of the build after it completes (during minimization emitout is set to false,
// since there is no need to see repeated errors).
func (c *config) build(emitout bool) int {
	// Issue a build of the generated code.
	c.buildOutFile = filepath.Join(c.tmpdir, "build.err.txt")
	cmd := []string{"go", "build", "-o", binaryName()}
	if c.gcflags != "" {
		cmd = append(cmd, "-gcflags=all="+c.gcflags)
	}
	if *raceflag {
		cmd = append(cmd, "-race")
	}
	cmd = append(cmd, ".")
	verb(1, "build command is: %v", cmd)
	return c.action(cmd, c.buildOutFile, emitout)
}

// run invokes a binary built from a generated corpus of Go code. If
// 'emitout' is set, then dump out the results of the run after it
// completes.
func (c *config) run(emitout bool) int {
	// Issue a run of the generated code.
	c.runOutFile = filepath.Join(c.tmpdir, "run.err.txt")
	cmd := []string{filepath.Join(c.gendir, binaryName())}
	verb(1, "run command is: %v", cmd)
	return c.action(cmd, c.runOutFile, emitout)
}

type minimizeMode int

const (
	minimizeBuildFailure = iota
	minimizeRuntimeFailure
)

// minimize tries to minimize a failing scenario down to a single
// package and/or function if possible. This is done using an
// iterative search. Here 'minimizeMode' tells us whether we're
// looking for a compile-time error or a runtime error.
func (c *config) minimize(mode minimizeMode) int {

	verb(0, "... starting minimization for failed directory %s", c.gendir)

	foundPkg := -1
	foundFcn := -1

	// Locate bad package. Uses brute-force linear search, could do better...
	for pidx := 0; pidx < c.NumTestPackages; pidx++ {
		verb(1, "minimization: trying package %d", pidx)
		c.gen(pidx, -1)
		st := c.build(false)
		if mode == minimizeBuildFailure {
			if st != 0 {
				// Found.
				foundPkg = pidx
				c.nerrors++
				break
			}
		} else {
			if st != 0 {
				warn("run minimization: unexpected build failed while searching for bad pkg")
				return 1
			}
			st := c.run(false)
			if st != 0 {
				// Found.
				c.nerrors++
				verb(1, "run minimization found bad package: %d", pidx)
				foundPkg = pidx
				break
			}
		}
	}
	if foundPkg == -1 {
		verb(0, "** minimization failed, could not locate bad package")
		return 1
	}
	warn("package minimization succeeded: found bad pkg %d", foundPkg)

	// clean unused packages
	for pidx := 0; pidx < c.NumTestPackages; pidx++ {
		if pidx != foundPkg {
			chp := filepath.Join(c.gendir, fmt.Sprintf("%s%s%d", c.Tag, generator.CheckerName, pidx))
			if err := os.RemoveAll(chp); err != nil {
				fatal("failed to clean pkg subdir %s: %v", chp, err)
			}
			clp := filepath.Join(c.gendir, fmt.Sprintf("%s%s%d", c.Tag, generator.CallerName, pidx))
			if err := os.RemoveAll(clp); err != nil {
				fatal("failed to clean pkg subdir %s: %v", clp, err)
			}
		}
	}

	// Locate bad function. Again, brute force.
	for fidx := 0; fidx < c.NumTestFunctions; fidx++ {
		c.gen(foundPkg, fidx)
		st := c.build(false)
		if mode == minimizeBuildFailure {
			if st != 0 {
				// Found.
				verb(1, "build minimization found bad function: %d", fidx)
				foundFcn = fidx
				break
			}
		} else {
			if st != 0 {
				warn("run minimization: unexpected build failed while searching for bad fcn")
				return 1
			}
			st := c.run(false)
			if st != 0 {
				// Found.
				verb(1, "run minimization found bad function: %d", fidx)
				foundFcn = fidx
				break
			}
		}
		// not the function we want ... continue the hunt
	}
	if foundFcn == -1 {
		verb(0, "** function minimization failed, could not locate bad function")
		return 1
	}
	warn("function minimization succeeded: found bad fcn %d", foundFcn)

	return 0
}

// cleanTemp removes the temp dir we've been working with.
func (c *config) cleanTemp() {
	if !*forcetmpcleanflag {
		if c.nerrors != 0 {
			verb(1, "preserving temp dir %s", c.tmpdir)
			return
		}
	}
	verb(1, "cleaning temp dir %s", c.tmpdir)
	os.RemoveAll(c.tmpdir)
}

// perform is the top level driver routine for the program, containing the
// main loop. Each iteration of the loop performs a generate/build/run
// sequence, and then updates the seed afterwards if no failure is found.
// If a failure is detected, we try to minimize it and then return without
// attempting any additional tests.
func (c *config) perform() int {
	defer c.cleanTemp()

	// Main loop
	for iter := 0; iter < *loopitflag; iter++ {
		if iter != 0 && iter%50 == 0 {
			// Note: cleaning the Go cache periodically is
			// pretty much a requirement if you want to do
			// things like overnight runs of the fuzzer,
			// but it is also a very unfriendly thing do
			// to if we're executing as part of a unit
			// test run (in which case there may be other
			// tests running in parallel with this
			// one). Check the "cleancache" flag before
			// doing this.
			if *cleancacheflag {
				docmd([]string{"go", "clean", "-cache"}, "")
			}
		}
		verb(0, "... begin iteration %d with current seed %d", iter, c.Seed)
		c.gen(-1, -1)
		st := c.build(true)
		if st != 0 {
			c.minimize(minimizeBuildFailure)
			return 1
		}
		st = c.run(true)
		if st != 0 {
			c.minimize(minimizeRuntimeFailure)
			return 1
		}
		// update seed so that we get different code on the next iter.
		c.Seed += 101
	}
	return 0
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("fuzz-runner: ")
	flag.Parse()
	if flag.NArg() != 0 {
		usage("unknown extra arguments")
	}
	verb(1, "in main, verblevel=%d", *verbflag)

	tmpdir, err := os.MkdirTemp("", "fuzzrun")
	if err != nil {
		fatal("creation of tempdir failed: %v", err)
	}
	gendir := filepath.Join(tmpdir, "fuzzTest")

	// select starting seed
	if *seedflag == -1 {
		now := time.Now()
		*seedflag = now.UnixNano() % 123456789
	}

	// set up params for this run
	c := &config{
		GenConfig: generator.GenConfig{
			NumTestPackages:  *numpkgsflag, // 100
			NumTestFunctions: *numfcnsflag, // 20
			Seed:             *seedflag,
			OutDir:           gendir,
			Pragma:           "-maxfail=9999",
			PkgPath:          pkName,
			EmitBad:          *emitbadflag,
			BadPackageIdx:    *selbadpkgflag,
			BadFuncIdx:       *selbadfcnflag,
		},
		tmpdir: tmpdir,
		gendir: gendir,
	}

	// kick off the main loop.
	st := c.perform()

	// done
	verb(1, "leaving main, num errors=%d", c.nerrors)
	os.Exit(st)
}
