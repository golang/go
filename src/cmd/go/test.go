// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"errors"
	"fmt"
	"go/ast"
	"go/build"
	"go/doc"
	"go/parser"
	"go/token"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"text/template"
	"time"
	"unicode"
	"unicode/utf8"
)

// Break init loop.
func init() {
	cmdTest.Run = runTest
}

var cmdTest = &Command{
	CustomFlags: true,
	UsageLine:   "test [-c] [-i] [build and test flags] [packages] [flags for test binary]",
	Short:       "test packages",
	Long: `
'Go test' automates testing the packages named by the import paths.
It prints a summary of the test results in the format:

	ok   archive/tar   0.011s
	FAIL archive/zip   0.022s
	ok   compress/gzip 0.033s
	...

followed by detailed output for each failed package.

'Go test' recompiles each package along with any files with names matching
the file pattern "*_test.go".
Files whose names begin with "_" (including "_test.go") or "." are ignored.
These additional files can contain test functions, benchmark functions, and
example functions.  See 'go help testfunc' for more.
Each listed package causes the execution of a separate test binary.

Test files that declare a package with the suffix "_test" will be compiled as a
separate package, and then linked and run with the main test binary.

By default, go test needs no arguments.  It compiles and tests the package
with source in the current directory, including tests, and runs the tests.

The package is built in a temporary directory so it does not interfere with the
non-test installation.

In addition to the build flags, the flags handled by 'go test' itself are:

	-c
		Compile the test binary to pkg.test but do not run it
		(where pkg is the last element of the package's import path).
		The file name can be changed with the -o flag.

	-exec xprog
	    Run the test binary using xprog. The behavior is the same as
	    in 'go run'. See 'go help run' for details.

	-i
	    Install packages that are dependencies of the test.
	    Do not run the test.

	-o file
		Compile the test binary to the named file.
		The test still runs (unless -c or -i is specified).


The test binary also accepts flags that control execution of the test; these
flags are also accessible by 'go test'.  See 'go help testflag' for details.

If the test binary needs any other flags, they should be presented after the
package names. The go tool treats as a flag the first argument that begins with
a minus sign that it does not recognize itself; that argument and all subsequent
arguments are passed as arguments to the test binary.

For more about build flags, see 'go help build'.
For more about specifying packages, see 'go help packages'.

See also: go build, go vet.
`,
}

var helpTestflag = &Command{
	UsageLine: "testflag",
	Short:     "description of testing flags",
	Long: `
The 'go test' command takes both flags that apply to 'go test' itself
and flags that apply to the resulting test binary.

Several of the flags control profiling and write an execution profile
suitable for "go tool pprof"; run "go tool pprof help" for more
information.  The --alloc_space, --alloc_objects, and --show_bytes
options of pprof control how the information is presented.

The following flags are recognized by the 'go test' command and
control the execution of any test:

	-bench regexp
	    Run benchmarks matching the regular expression.
	    By default, no benchmarks run. To run all benchmarks,
	    use '-bench .' or '-bench=.'.

	-benchmem
	    Print memory allocation statistics for benchmarks.

	-benchtime t
	    Run enough iterations of each benchmark to take t, specified
	    as a time.Duration (for example, -benchtime 1h30s).
	    The default is 1 second (1s).

	-blockprofile block.out
	    Write a goroutine blocking profile to the specified file
	    when all tests are complete.
	    Writes test binary as -c would.

	-blockprofilerate n
	    Control the detail provided in goroutine blocking profiles by
	    calling runtime.SetBlockProfileRate with n.
	    See 'godoc runtime SetBlockProfileRate'.
	    The profiler aims to sample, on average, one blocking event every
	    n nanoseconds the program spends blocked.  By default,
	    if -test.blockprofile is set without this flag, all blocking events
	    are recorded, equivalent to -test.blockprofilerate=1.

	-cover
	    Enable coverage analysis.

	-covermode set,count,atomic
	    Set the mode for coverage analysis for the package[s]
	    being tested. The default is "set" unless -race is enabled,
	    in which case it is "atomic".
	    The values:
		set: bool: does this statement run?
		count: int: how many times does this statement run?
		atomic: int: count, but correct in multithreaded tests;
			significantly more expensive.
	    Sets -cover.

	-coverpkg pkg1,pkg2,pkg3
	    Apply coverage analysis in each test to the given list of packages.
	    The default is for each test to analyze only the package being tested.
	    Packages are specified as import paths.
	    Sets -cover.

	-coverprofile cover.out
	    Write a coverage profile to the file after all tests have passed.
	    Sets -cover.

	-cpu 1,2,4
	    Specify a list of GOMAXPROCS values for which the tests or
	    benchmarks should be executed.  The default is the current value
	    of GOMAXPROCS.

	-cpuprofile cpu.out
	    Write a CPU profile to the specified file before exiting.
	    Writes test binary as -c would.

	-memprofile mem.out
	    Write a memory profile to the file after all tests have passed.
	    Writes test binary as -c would.

	-memprofilerate n
	    Enable more precise (and expensive) memory profiles by setting
	    runtime.MemProfileRate.  See 'godoc runtime MemProfileRate'.
	    To profile all memory allocations, use -test.memprofilerate=1
	    and pass --alloc_space flag to the pprof tool.

	-outputdir directory
	    Place output files from profiling in the specified directory,
	    by default the directory in which "go test" is running.

	-parallel n
	    Allow parallel execution of test functions that call t.Parallel.
	    The value of this flag is the maximum number of tests to run
	    simultaneously; by default, it is set to the value of GOMAXPROCS.

	-run regexp
	    Run only those tests and examples matching the regular
	    expression.

	-short
	    Tell long-running tests to shorten their run time.
	    It is off by default but set during all.bash so that installing
	    the Go tree can run a sanity check but not spend time running
	    exhaustive tests.

	-timeout t
	    If a test runs longer than t, panic.

	-trace trace.out
	    Write an execution trace to the specified file before exiting.
	    Writes test binary as -c would.

	-v
	    Verbose output: log all tests as they are run. Also print all
	    text from Log and Logf calls even if the test succeeds.

The test binary, called pkg.test where pkg is the name of the
directory containing the package sources, can be invoked directly
after building it with 'go test -c'. When invoking the test binary
directly, each of the standard flag names must be prefixed with 'test.',
as in -test.run=TestMyFunc or -test.v.

When running 'go test', flags not listed above are passed through
unaltered. For instance, the command

	go test -x -v -cpuprofile=prof.out -dir=testdata -update

will compile the test binary and then run it as

	pkg.test -test.v -test.cpuprofile=prof.out -dir=testdata -update

The test flags that generate profiles (other than for coverage) also
leave the test binary in pkg.test for use when analyzing the profiles.

Flags not recognized by 'go test' must be placed after any specified packages.
`,
}

var helpTestfunc = &Command{
	UsageLine: "testfunc",
	Short:     "description of testing functions",
	Long: `
The 'go test' command expects to find test, benchmark, and example functions
in the "*_test.go" files corresponding to the package under test.

A test function is one named TestXXX (where XXX is any alphanumeric string
not starting with a lower case letter) and should have the signature,

	func TestXXX(t *testing.T) { ... }

A benchmark function is one named BenchmarkXXX and should have the signature,

	func BenchmarkXXX(b *testing.B) { ... }

An example function is similar to a test function but, instead of using
*testing.T to report success or failure, prints output to os.Stdout.
That output is compared against the function's "Output:" comment, which
must be the last comment in the function body (see example below). An
example with no such comment, or with no text after "Output:" is compiled
but not executed.

Godoc displays the body of ExampleXXX to demonstrate the use
of the function, constant, or variable XXX.  An example of a method M with
receiver type T or *T is named ExampleT_M.  There may be multiple examples
for a given function, constant, or variable, distinguished by a trailing _xxx,
where xxx is a suffix not beginning with an upper case letter.

Here is an example of an example:

	func ExamplePrintln() {
		Println("The output of\nthis example.")
		// Output: The output of
		// this example.
	}

The entire test file is presented as the example when it contains a single
example function, at least one other function, type, variable, or constant
declaration, and no test or benchmark functions.

See the documentation of the testing package for more information.
`,
}

var (
	testC            bool       // -c flag
	testCover        bool       // -cover flag
	testCoverMode    string     // -covermode flag
	testCoverPaths   []string   // -coverpkg flag
	testCoverPkgs    []*Package // -coverpkg flag
	testO            string     // -o flag
	testProfile      bool       // some profiling flag
	testNeedBinary   bool       // profile needs to keep binary around
	testV            bool       // -v flag
	testTimeout      string     // -timeout flag
	testArgs         []string
	testBench        bool
	testStreamOutput bool // show output as it is generated
	testShowPass     bool // show passing output

	testKillTimeout = 10 * time.Minute
)

var testMainDeps = map[string]bool{
	// Dependencies for testmain.
	"testing": true,
	"regexp":  true,
	"os":      true,
}

func runTest(cmd *Command, args []string) {
	var pkgArgs []string
	pkgArgs, testArgs = testFlags(args)

	findExecCmd() // initialize cached result

	raceInit()
	pkgs := packagesForBuild(pkgArgs)
	if len(pkgs) == 0 {
		fatalf("no packages to test")
	}

	if testC && len(pkgs) != 1 {
		fatalf("cannot use -c flag with multiple packages")
	}
	if testO != "" && len(pkgs) != 1 {
		fatalf("cannot use -o flag with multiple packages")
	}
	if testProfile && len(pkgs) != 1 {
		fatalf("cannot use test profile flag with multiple packages")
	}

	// If a test timeout was given and is parseable, set our kill timeout
	// to that timeout plus one minute.  This is a backup alarm in case
	// the test wedges with a goroutine spinning and its background
	// timer does not get a chance to fire.
	if dt, err := time.ParseDuration(testTimeout); err == nil && dt > 0 {
		testKillTimeout = dt + 1*time.Minute
	}

	// show passing test output (after buffering) with -v flag.
	// must buffer because tests are running in parallel, and
	// otherwise the output will get mixed.
	testShowPass = testV

	// stream test output (no buffering) when no package has
	// been given on the command line (implicit current directory)
	// or when benchmarking.
	// Also stream if we're showing output anyway with a
	// single package under test.  In that case, streaming the
	// output produces the same result as not streaming,
	// just more immediately.
	testStreamOutput = len(pkgArgs) == 0 || testBench ||
		(len(pkgs) <= 1 && testShowPass)

	var b builder
	b.init()

	if buildI {
		buildV = testV

		deps := make(map[string]bool)
		for dep := range testMainDeps {
			deps[dep] = true
		}

		for _, p := range pkgs {
			// Dependencies for each test.
			for _, path := range p.Imports {
				deps[path] = true
			}
			for _, path := range p.TestImports {
				deps[path] = true
			}
			for _, path := range p.XTestImports {
				deps[path] = true
			}
		}

		// translate C to runtime/cgo
		if deps["C"] {
			delete(deps, "C")
			deps["runtime/cgo"] = true
			if buildContext.GOOS == runtime.GOOS && buildContext.GOARCH == runtime.GOARCH {
				deps["cmd/cgo"] = true
			}
		}
		// Ignore pseudo-packages.
		delete(deps, "unsafe")

		all := []string{}
		for path := range deps {
			if !build.IsLocalImport(path) {
				all = append(all, path)
			}
		}
		sort.Strings(all)

		a := &action{}
		for _, p := range packagesForBuild(all) {
			a.deps = append(a.deps, b.action(modeInstall, modeInstall, p))
		}
		b.do(a)
		if !testC || a.failed {
			return
		}
		b.init()
	}

	var builds, runs, prints []*action

	if testCoverPaths != nil {
		// Load packages that were asked about for coverage.
		// packagesForBuild exits if the packages cannot be loaded.
		testCoverPkgs = packagesForBuild(testCoverPaths)

		// Warn about -coverpkg arguments that are not actually used.
		used := make(map[string]bool)
		for _, p := range pkgs {
			used[p.ImportPath] = true
			for _, dep := range p.Deps {
				used[dep] = true
			}
		}
		for _, p := range testCoverPkgs {
			if !used[p.ImportPath] {
				log.Printf("warning: no packages being tested depend on %s", p.ImportPath)
			}
		}

		// Mark all the coverage packages for rebuilding with coverage.
		for _, p := range testCoverPkgs {
			p.Stale = true // rebuild
			p.fake = true  // do not warn about rebuild
			p.coverMode = testCoverMode
			var coverFiles []string
			coverFiles = append(coverFiles, p.GoFiles...)
			coverFiles = append(coverFiles, p.CgoFiles...)
			coverFiles = append(coverFiles, p.TestGoFiles...)
			p.coverVars = declareCoverVars(p.ImportPath, coverFiles...)
		}
	}

	// Prepare build + run + print actions for all packages being tested.
	for _, p := range pkgs {
		buildTest, runTest, printTest, err := b.test(p)
		if err != nil {
			str := err.Error()
			if strings.HasPrefix(str, "\n") {
				str = str[1:]
			}
			failed := fmt.Sprintf("FAIL\t%s [setup failed]\n", p.ImportPath)

			if p.ImportPath != "" {
				errorf("# %s\n%s\n%s", p.ImportPath, str, failed)
			} else {
				errorf("%s\n%s", str, failed)
			}
			continue
		}
		builds = append(builds, buildTest)
		runs = append(runs, runTest)
		prints = append(prints, printTest)
	}

	// Ultimately the goal is to print the output.
	root := &action{deps: prints}

	// Force the printing of results to happen in order,
	// one at a time.
	for i, a := range prints {
		if i > 0 {
			a.deps = append(a.deps, prints[i-1])
		}
	}

	// Force benchmarks to run in serial.
	if !testC && testBench {
		// The first run must wait for all builds.
		// Later runs must wait for the previous run's print.
		for i, run := range runs {
			if i == 0 {
				run.deps = append(run.deps, builds...)
			} else {
				run.deps = append(run.deps, prints[i-1])
			}
		}
	}

	// If we are building any out-of-date packages other
	// than those under test, warn.
	okBuild := map[*Package]bool{}
	for _, p := range pkgs {
		okBuild[p] = true
	}
	warned := false
	for _, a := range actionList(root) {
		if a.p == nil || okBuild[a.p] {
			continue
		}
		okBuild[a.p] = true // warn at most once

		// Don't warn about packages being rebuilt because of
		// things like coverage analysis.
		for _, p1 := range a.p.imports {
			if p1.fake {
				a.p.fake = true
			}
		}

		if a.f != nil && !okBuild[a.p] && !a.p.fake && !a.p.local {
			if !warned {
				fmt.Fprintf(os.Stderr, "warning: building out-of-date packages:\n")
				warned = true
			}
			fmt.Fprintf(os.Stderr, "\t%s\n", a.p.ImportPath)
		}
	}
	if warned {
		args := strings.Join(pkgArgs, " ")
		if args != "" {
			args = " " + args
		}
		extraOpts := ""
		if buildRace {
			extraOpts = "-race "
		}
		fmt.Fprintf(os.Stderr, "installing these packages with 'go test %s-i%s' will speed future tests.\n\n", extraOpts, args)
	}

	b.do(root)
}

func contains(x []string, s string) bool {
	for _, t := range x {
		if t == s {
			return true
		}
	}
	return false
}

var windowsBadWords = []string{
	"install",
	"patch",
	"setup",
	"update",
}

func (b *builder) test(p *Package) (buildAction, runAction, printAction *action, err error) {
	if len(p.TestGoFiles)+len(p.XTestGoFiles) == 0 {
		build := b.action(modeBuild, modeBuild, p)
		run := &action{p: p, deps: []*action{build}}
		print := &action{f: (*builder).notest, p: p, deps: []*action{run}}
		return build, run, print, nil
	}

	// Build Package structs describing:
	//	ptest - package + test files
	//	pxtest - package of external test files
	//	pmain - pkg.test binary
	var ptest, pxtest, pmain *Package

	var imports, ximports []*Package
	var stk importStack
	stk.push(p.ImportPath + " (test)")
	for _, path := range p.TestImports {
		p1 := loadImport(path, p.Dir, &stk, p.build.TestImportPos[path])
		if p1.Error != nil {
			return nil, nil, nil, p1.Error
		}
		if contains(p1.Deps, p.ImportPath) {
			// Same error that loadPackage returns (via reusePackage) in pkg.go.
			// Can't change that code, because that code is only for loading the
			// non-test copy of a package.
			err := &PackageError{
				ImportStack:   testImportStack(stk[0], p1, p.ImportPath),
				Err:           "import cycle not allowed in test",
				isImportCycle: true,
			}
			return nil, nil, nil, err
		}
		imports = append(imports, p1)
	}
	stk.pop()
	stk.push(p.ImportPath + "_test")
	pxtestNeedsPtest := false
	for _, path := range p.XTestImports {
		if path == p.ImportPath {
			pxtestNeedsPtest = true
			continue
		}
		p1 := loadImport(path, p.Dir, &stk, p.build.XTestImportPos[path])
		if p1.Error != nil {
			return nil, nil, nil, p1.Error
		}
		ximports = append(ximports, p1)
	}
	stk.pop()

	// Use last element of import path, not package name.
	// They differ when package name is "main".
	// But if the import path is "command-line-arguments",
	// like it is during 'go run', use the package name.
	var elem string
	if p.ImportPath == "command-line-arguments" {
		elem = p.Name
	} else {
		_, elem = path.Split(p.ImportPath)
	}
	testBinary := elem + ".test"

	// The ptest package needs to be importable under the
	// same import path that p has, but we cannot put it in
	// the usual place in the temporary tree, because then
	// other tests will see it as the real package.
	// Instead we make a _test directory under the import path
	// and then repeat the import path there.  We tell the
	// compiler and linker to look in that _test directory first.
	//
	// That is, if the package under test is unicode/utf8,
	// then the normal place to write the package archive is
	// $WORK/unicode/utf8.a, but we write the test package archive to
	// $WORK/unicode/utf8/_test/unicode/utf8.a.
	// We write the external test package archive to
	// $WORK/unicode/utf8/_test/unicode/utf8_test.a.
	testDir := filepath.Join(b.work, filepath.FromSlash(p.ImportPath+"/_test"))
	ptestObj := buildToolchain.pkgpath(testDir, p)

	// Create the directory for the .a files.
	ptestDir, _ := filepath.Split(ptestObj)
	if err := b.mkdir(ptestDir); err != nil {
		return nil, nil, nil, err
	}

	// Should we apply coverage analysis locally,
	// only for this package and only for this test?
	// Yes, if -cover is on but -coverpkg has not specified
	// a list of packages for global coverage.
	localCover := testCover && testCoverPaths == nil

	// Test package.
	if len(p.TestGoFiles) > 0 || localCover || p.Name == "main" {
		ptest = new(Package)
		*ptest = *p
		ptest.GoFiles = nil
		ptest.GoFiles = append(ptest.GoFiles, p.GoFiles...)
		ptest.GoFiles = append(ptest.GoFiles, p.TestGoFiles...)
		ptest.target = ""
		ptest.Imports = stringList(p.Imports, p.TestImports)
		ptest.imports = append(append([]*Package{}, p.imports...), imports...)
		ptest.pkgdir = testDir
		ptest.fake = true
		ptest.forceLibrary = true
		ptest.Stale = true
		ptest.build = new(build.Package)
		*ptest.build = *p.build
		m := map[string][]token.Position{}
		for k, v := range p.build.ImportPos {
			m[k] = append(m[k], v...)
		}
		for k, v := range p.build.TestImportPos {
			m[k] = append(m[k], v...)
		}
		ptest.build.ImportPos = m

		if localCover {
			ptest.coverMode = testCoverMode
			var coverFiles []string
			coverFiles = append(coverFiles, ptest.GoFiles...)
			coverFiles = append(coverFiles, ptest.CgoFiles...)
			ptest.coverVars = declareCoverVars(ptest.ImportPath, coverFiles...)
		}
	} else {
		ptest = p
	}

	// External test package.
	if len(p.XTestGoFiles) > 0 {
		pxtest = &Package{
			Name:        p.Name + "_test",
			ImportPath:  p.ImportPath + "_test",
			localPrefix: p.localPrefix,
			Root:        p.Root,
			Dir:         p.Dir,
			GoFiles:     p.XTestGoFiles,
			Imports:     p.XTestImports,
			build: &build.Package{
				ImportPos: p.build.XTestImportPos,
			},
			imports: ximports,
			pkgdir:  testDir,
			fake:    true,
			Stale:   true,
		}
		if pxtestNeedsPtest {
			pxtest.imports = append(pxtest.imports, ptest)
		}
	}

	// Action for building pkg.test.
	pmain = &Package{
		Name:       "main",
		Dir:        testDir,
		GoFiles:    []string{"_testmain.go"},
		ImportPath: "testmain",
		Root:       p.Root,
		build:      &build.Package{Name: "main"},
		pkgdir:     testDir,
		fake:       true,
		Stale:      true,
		omitDWARF:  !testC && !testNeedBinary,
	}

	// The generated main also imports testing, regexp, and os.
	stk.push("testmain")
	for dep := range testMainDeps {
		if dep == ptest.ImportPath {
			pmain.imports = append(pmain.imports, ptest)
		} else {
			p1 := loadImport(dep, "", &stk, nil)
			if p1.Error != nil {
				return nil, nil, nil, p1.Error
			}
			pmain.imports = append(pmain.imports, p1)
		}
	}

	if testCoverPkgs != nil {
		// Add imports, but avoid duplicates.
		seen := map[*Package]bool{p: true, ptest: true}
		for _, p1 := range pmain.imports {
			seen[p1] = true
		}
		for _, p1 := range testCoverPkgs {
			if !seen[p1] {
				seen[p1] = true
				pmain.imports = append(pmain.imports, p1)
			}
		}
	}

	// Do initial scan for metadata needed for writing _testmain.go
	// Use that metadata to update the list of imports for package main.
	// The list of imports is used by recompileForTest and by the loop
	// afterward that gathers t.Cover information.
	t, err := loadTestFuncs(ptest)
	if err != nil {
		return nil, nil, nil, err
	}
	if len(ptest.GoFiles) > 0 {
		pmain.imports = append(pmain.imports, ptest)
		t.ImportTest = true
	}
	if pxtest != nil {
		pmain.imports = append(pmain.imports, pxtest)
		t.ImportXtest = true
	}

	if ptest != p && localCover {
		// We have made modifications to the package p being tested
		// and are rebuilding p (as ptest), writing it to the testDir tree.
		// Arrange to rebuild, writing to that same tree, all packages q
		// such that the test depends on q, and q depends on p.
		// This makes sure that q sees the modifications to p.
		// Strictly speaking, the rebuild is only necessary if the
		// modifications to p change its export metadata, but
		// determining that is a bit tricky, so we rebuild always.
		//
		// This will cause extra compilation, so for now we only do it
		// when testCover is set. The conditions are more general, though,
		// and we may find that we need to do it always in the future.
		recompileForTest(pmain, p, ptest, testDir)
	}

	for _, cp := range pmain.imports {
		if len(cp.coverVars) > 0 {
			t.Cover = append(t.Cover, coverInfo{cp, cp.coverVars})
		}
	}

	// writeTestmain writes _testmain.go. This must happen after recompileForTest,
	// because recompileForTest modifies XXX.
	if err := writeTestmain(filepath.Join(testDir, "_testmain.go"), t); err != nil {
		return nil, nil, nil, err
	}

	computeStale(pmain)

	if ptest != p {
		a := b.action(modeBuild, modeBuild, ptest)
		a.objdir = testDir + string(filepath.Separator) + "_obj_test" + string(filepath.Separator)
		a.objpkg = ptestObj
		a.target = ptestObj
		a.link = false
	}

	if pxtest != nil {
		a := b.action(modeBuild, modeBuild, pxtest)
		a.objdir = testDir + string(filepath.Separator) + "_obj_xtest" + string(filepath.Separator)
		a.objpkg = buildToolchain.pkgpath(testDir, pxtest)
		a.target = a.objpkg
	}

	a := b.action(modeBuild, modeBuild, pmain)
	a.objdir = testDir + string(filepath.Separator)
	a.objpkg = filepath.Join(testDir, "main.a")
	a.target = filepath.Join(testDir, testBinary) + exeSuffix
	if goos == "windows" {
		// There are many reserved words on Windows that,
		// if used in the name of an executable, cause Windows
		// to try to ask for extra permissions.
		// The word list includes setup, install, update, and patch,
		// but it does not appear to be defined anywhere.
		// We have run into this trying to run the
		// go.codereview/patch tests.
		// For package names containing those words, use test.test.exe
		// instead of pkgname.test.exe.
		// Note that this file name is only used in the Go command's
		// temporary directory. If the -c or other flags are
		// given, the code below will still use pkgname.test.exe.
		// There are two user-visible effects of this change.
		// First, you can actually run 'go test' in directories that
		// have names that Windows thinks are installer-like,
		// without getting a dialog box asking for more permissions.
		// Second, in the Windows process listing during go test,
		// the test shows up as test.test.exe, not pkgname.test.exe.
		// That second one is a drawback, but it seems a small
		// price to pay for the test running at all.
		// If maintaining the list of bad words is too onerous,
		// we could just do this always on Windows.
		for _, bad := range windowsBadWords {
			if strings.Contains(testBinary, bad) {
				a.target = filepath.Join(testDir, "test.test") + exeSuffix
				break
			}
		}
	}
	buildAction = a

	if testC || testNeedBinary {
		// -c or profiling flag: create action to copy binary to ./test.out.
		target := filepath.Join(cwd, testBinary+exeSuffix)
		if testO != "" {
			target = testO
			if !filepath.IsAbs(target) {
				target = filepath.Join(cwd, target)
			}
		}
		buildAction = &action{
			f:      (*builder).install,
			deps:   []*action{buildAction},
			p:      pmain,
			target: target,
		}
		runAction = buildAction // make sure runAction != nil even if not running test
	}
	if testC {
		printAction = &action{p: p, deps: []*action{runAction}} // nop
	} else {
		// run test
		runAction = &action{
			f:          (*builder).runTest,
			deps:       []*action{buildAction},
			p:          p,
			ignoreFail: true,
		}
		cleanAction := &action{
			f:    (*builder).cleanTest,
			deps: []*action{runAction},
			p:    p,
		}
		printAction = &action{
			f:    (*builder).printTest,
			deps: []*action{cleanAction},
			p:    p,
		}
	}

	return buildAction, runAction, printAction, nil
}

func testImportStack(top string, p *Package, target string) []string {
	stk := []string{top, p.ImportPath}
Search:
	for p.ImportPath != target {
		for _, p1 := range p.imports {
			if p1.ImportPath == target || contains(p1.Deps, target) {
				stk = append(stk, p1.ImportPath)
				p = p1
				continue Search
			}
		}
		// Can't happen, but in case it does...
		stk = append(stk, "<lost path to cycle>")
		break
	}
	return stk
}

func recompileForTest(pmain, preal, ptest *Package, testDir string) {
	// The "test copy" of preal is ptest.
	// For each package that depends on preal, make a "test copy"
	// that depends on ptest. And so on, up the dependency tree.
	testCopy := map[*Package]*Package{preal: ptest}
	for _, p := range packageList([]*Package{pmain}) {
		// Copy on write.
		didSplit := false
		split := func() {
			if didSplit {
				return
			}
			didSplit = true
			if p.pkgdir != testDir {
				p1 := new(Package)
				testCopy[p] = p1
				*p1 = *p
				p1.imports = make([]*Package, len(p.imports))
				copy(p1.imports, p.imports)
				p = p1
				p.pkgdir = testDir
				p.target = ""
				p.fake = true
				p.Stale = true
			}
		}

		// Update p.deps and p.imports to use at test copies.
		for i, dep := range p.deps {
			if p1 := testCopy[dep]; p1 != nil && p1 != dep {
				split()
				p.deps[i] = p1
			}
		}
		for i, imp := range p.imports {
			if p1 := testCopy[imp]; p1 != nil && p1 != imp {
				split()
				p.imports[i] = p1
			}
		}
	}
}

var coverIndex = 0

// isTestFile reports whether the source file is a set of tests and should therefore
// be excluded from coverage analysis.
func isTestFile(file string) bool {
	// We don't cover tests, only the code they test.
	return strings.HasSuffix(file, "_test.go")
}

// declareCoverVars attaches the required cover variables names
// to the files, to be used when annotating the files.
func declareCoverVars(importPath string, files ...string) map[string]*CoverVar {
	coverVars := make(map[string]*CoverVar)
	for _, file := range files {
		if isTestFile(file) {
			continue
		}
		coverVars[file] = &CoverVar{
			File: filepath.Join(importPath, file),
			Var:  fmt.Sprintf("GoCover_%d", coverIndex),
		}
		coverIndex++
	}
	return coverVars
}

// runTest is the action for running a test binary.
func (b *builder) runTest(a *action) error {
	args := stringList(findExecCmd(), a.deps[0].target, testArgs)
	a.testOutput = new(bytes.Buffer)

	if buildN || buildX {
		b.showcmd("", "%s", strings.Join(args, " "))
		if buildN {
			return nil
		}
	}

	if a.failed {
		// We were unable to build the binary.
		a.failed = false
		fmt.Fprintf(a.testOutput, "FAIL\t%s [build failed]\n", a.p.ImportPath)
		setExitStatus(1)
		return nil
	}

	cmd := exec.Command(args[0], args[1:]...)
	cmd.Dir = a.p.Dir
	cmd.Env = envForDir(cmd.Dir)
	var buf bytes.Buffer
	if testStreamOutput {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	} else {
		cmd.Stdout = &buf
		cmd.Stderr = &buf
	}

	// If there are any local SWIG dependencies, we want to load
	// the shared library from the build directory.
	if a.p.usesSwig() {
		env := cmd.Env
		found := false
		prefix := "LD_LIBRARY_PATH="
		for i, v := range env {
			if strings.HasPrefix(v, prefix) {
				env[i] = v + ":."
				found = true
				break
			}
		}
		if !found {
			env = append(env, "LD_LIBRARY_PATH=.")
		}
		cmd.Env = env
	}

	t0 := time.Now()
	err := cmd.Start()

	// This is a last-ditch deadline to detect and
	// stop wedged test binaries, to keep the builders
	// running.
	if err == nil {
		tick := time.NewTimer(testKillTimeout)
		startSigHandlers()
		done := make(chan error)
		go func() {
			done <- cmd.Wait()
		}()
	Outer:
		select {
		case err = <-done:
			// ok
		case <-tick.C:
			if signalTrace != nil {
				// Send a quit signal in the hope that the program will print
				// a stack trace and exit. Give it five seconds before resorting
				// to Kill.
				cmd.Process.Signal(signalTrace)
				select {
				case err = <-done:
					fmt.Fprintf(&buf, "*** Test killed with %v: ran too long (%v).\n", signalTrace, testKillTimeout)
					break Outer
				case <-time.After(5 * time.Second):
				}
			}
			cmd.Process.Kill()
			err = <-done
			fmt.Fprintf(&buf, "*** Test killed: ran too long (%v).\n", testKillTimeout)
		}
		tick.Stop()
	}
	out := buf.Bytes()
	t := fmt.Sprintf("%.3fs", time.Since(t0).Seconds())
	if err == nil {
		if testShowPass {
			a.testOutput.Write(out)
		}
		fmt.Fprintf(a.testOutput, "ok  \t%s\t%s%s\n", a.p.ImportPath, t, coveragePercentage(out))
		return nil
	}

	setExitStatus(1)
	if len(out) > 0 {
		a.testOutput.Write(out)
		// assume printing the test binary's exit status is superfluous
	} else {
		fmt.Fprintf(a.testOutput, "%s\n", err)
	}
	fmt.Fprintf(a.testOutput, "FAIL\t%s\t%s\n", a.p.ImportPath, t)

	return nil
}

// coveragePercentage returns the coverage results (if enabled) for the
// test. It uncovers the data by scanning the output from the test run.
func coveragePercentage(out []byte) string {
	if !testCover {
		return ""
	}
	// The string looks like
	//	test coverage for encoding/binary: 79.9% of statements
	// Extract the piece from the percentage to the end of the line.
	re := regexp.MustCompile(`coverage: (.*)\n`)
	matches := re.FindSubmatch(out)
	if matches == nil {
		// Probably running "go test -cover" not "go test -cover fmt".
		// The coverage output will appear in the output directly.
		return ""
	}
	return fmt.Sprintf("\tcoverage: %s", matches[1])
}

// cleanTest is the action for cleaning up after a test.
func (b *builder) cleanTest(a *action) error {
	if buildWork {
		return nil
	}
	run := a.deps[0]
	testDir := filepath.Join(b.work, filepath.FromSlash(run.p.ImportPath+"/_test"))
	os.RemoveAll(testDir)
	return nil
}

// printTest is the action for printing a test result.
func (b *builder) printTest(a *action) error {
	clean := a.deps[0]
	run := clean.deps[0]
	os.Stdout.Write(run.testOutput.Bytes())
	run.testOutput = nil
	return nil
}

// notest is the action for testing a package with no test files.
func (b *builder) notest(a *action) error {
	fmt.Printf("?   \t%s\t[no test files]\n", a.p.ImportPath)
	return nil
}

// isTestMain tells whether fn is a TestMain(m *testing.M) function.
func isTestMain(fn *ast.FuncDecl) bool {
	if fn.Name.String() != "TestMain" ||
		fn.Type.Results != nil && len(fn.Type.Results.List) > 0 ||
		fn.Type.Params == nil ||
		len(fn.Type.Params.List) != 1 ||
		len(fn.Type.Params.List[0].Names) > 1 {
		return false
	}
	ptr, ok := fn.Type.Params.List[0].Type.(*ast.StarExpr)
	if !ok {
		return false
	}
	// We can't easily check that the type is *testing.M
	// because we don't know how testing has been imported,
	// but at least check that it's *M or *something.M.
	if name, ok := ptr.X.(*ast.Ident); ok && name.Name == "M" {
		return true
	}
	if sel, ok := ptr.X.(*ast.SelectorExpr); ok && sel.Sel.Name == "M" {
		return true
	}
	return false
}

// isTest tells whether name looks like a test (or benchmark, according to prefix).
// It is a Test (say) if there is a character after Test that is not a lower-case letter.
// We don't want TesticularCancer.
func isTest(name, prefix string) bool {
	if !strings.HasPrefix(name, prefix) {
		return false
	}
	if len(name) == len(prefix) { // "Test" is ok
		return true
	}
	rune, _ := utf8.DecodeRuneInString(name[len(prefix):])
	return !unicode.IsLower(rune)
}

type coverInfo struct {
	Package *Package
	Vars    map[string]*CoverVar
}

// loadTestFuncs returns the testFuncs describing the tests that will be run.
func loadTestFuncs(ptest *Package) (*testFuncs, error) {
	t := &testFuncs{
		Package: ptest,
	}
	for _, file := range ptest.TestGoFiles {
		if err := t.load(filepath.Join(ptest.Dir, file), "_test", &t.ImportTest, &t.NeedTest); err != nil {
			return nil, err
		}
	}
	for _, file := range ptest.XTestGoFiles {
		if err := t.load(filepath.Join(ptest.Dir, file), "_xtest", &t.ImportXtest, &t.NeedXtest); err != nil {
			return nil, err
		}
	}
	return t, nil
}

// writeTestmain writes the _testmain.go file for t to the file named out.
func writeTestmain(out string, t *testFuncs) error {
	f, err := os.Create(out)
	if err != nil {
		return err
	}
	defer f.Close()

	if err := testmainTmpl.Execute(f, t); err != nil {
		return err
	}

	return nil
}

type testFuncs struct {
	Tests       []testFunc
	Benchmarks  []testFunc
	Examples    []testFunc
	TestMain    *testFunc
	Package     *Package
	ImportTest  bool
	NeedTest    bool
	ImportXtest bool
	NeedXtest   bool
	Cover       []coverInfo
}

func (t *testFuncs) CoverMode() string {
	return testCoverMode
}

func (t *testFuncs) CoverEnabled() bool {
	return testCover
}

// Covered returns a string describing which packages are being tested for coverage.
// If the covered package is the same as the tested package, it returns the empty string.
// Otherwise it is a comma-separated human-readable list of packages beginning with
// " in", ready for use in the coverage message.
func (t *testFuncs) Covered() string {
	if testCoverPaths == nil {
		return ""
	}
	return " in " + strings.Join(testCoverPaths, ", ")
}

// Tested returns the name of the package being tested.
func (t *testFuncs) Tested() string {
	return t.Package.Name
}

type testFunc struct {
	Package string // imported package name (_test or _xtest)
	Name    string // function name
	Output  string // output, for examples
}

var testFileSet = token.NewFileSet()

func (t *testFuncs) load(filename, pkg string, doImport, seen *bool) error {
	f, err := parser.ParseFile(testFileSet, filename, nil, parser.ParseComments)
	if err != nil {
		return expandScanner(err)
	}
	for _, d := range f.Decls {
		n, ok := d.(*ast.FuncDecl)
		if !ok {
			continue
		}
		if n.Recv != nil {
			continue
		}
		name := n.Name.String()
		switch {
		case isTestMain(n):
			if t.TestMain != nil {
				return errors.New("multiple definitions of TestMain")
			}
			t.TestMain = &testFunc{pkg, name, ""}
			*doImport, *seen = true, true
		case isTest(name, "Test"):
			t.Tests = append(t.Tests, testFunc{pkg, name, ""})
			*doImport, *seen = true, true
		case isTest(name, "Benchmark"):
			t.Benchmarks = append(t.Benchmarks, testFunc{pkg, name, ""})
			*doImport, *seen = true, true
		}
	}
	ex := doc.Examples(f)
	sort.Sort(byOrder(ex))
	for _, e := range ex {
		*doImport = true // import test file whether executed or not
		if e.Output == "" && !e.EmptyOutput {
			// Don't run examples with no output.
			continue
		}
		t.Examples = append(t.Examples, testFunc{pkg, "Example" + e.Name, e.Output})
		*seen = true
	}
	return nil
}

type byOrder []*doc.Example

func (x byOrder) Len() int           { return len(x) }
func (x byOrder) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x byOrder) Less(i, j int) bool { return x[i].Order < x[j].Order }

var testmainTmpl = template.Must(template.New("main").Parse(`
package main

import (
{{if not .TestMain}}
	"os"
{{end}}
	"regexp"
	"testing"

{{if .ImportTest}}
	{{if .NeedTest}}_test{{else}}_{{end}} {{.Package.ImportPath | printf "%q"}}
{{end}}
{{if .ImportXtest}}
	{{if .NeedXtest}}_xtest{{else}}_{{end}} {{.Package.ImportPath | printf "%s_test" | printf "%q"}}
{{end}}
{{range $i, $p := .Cover}}
	_cover{{$i}} {{$p.Package.ImportPath | printf "%q"}}
{{end}}
)

var tests = []testing.InternalTest{
{{range .Tests}}
	{"{{.Name}}", {{.Package}}.{{.Name}}},
{{end}}
}

var benchmarks = []testing.InternalBenchmark{
{{range .Benchmarks}}
	{"{{.Name}}", {{.Package}}.{{.Name}}},
{{end}}
}

var examples = []testing.InternalExample{
{{range .Examples}}
	{"{{.Name}}", {{.Package}}.{{.Name}}, {{.Output | printf "%q"}}},
{{end}}
}

var matchPat string
var matchRe *regexp.Regexp

func matchString(pat, str string) (result bool, err error) {
	if matchRe == nil || matchPat != pat {
		matchPat = pat
		matchRe, err = regexp.Compile(matchPat)
		if err != nil {
			return
		}
	}
	return matchRe.MatchString(str), nil
}

{{if .CoverEnabled}}

// Only updated by init functions, so no need for atomicity.
var (
	coverCounters = make(map[string][]uint32)
	coverBlocks = make(map[string][]testing.CoverBlock)
)

func init() {
	{{range $i, $p := .Cover}}
	{{range $file, $cover := $p.Vars}}
	coverRegisterFile({{printf "%q" $cover.File}}, _cover{{$i}}.{{$cover.Var}}.Count[:], _cover{{$i}}.{{$cover.Var}}.Pos[:], _cover{{$i}}.{{$cover.Var}}.NumStmt[:])
	{{end}}
	{{end}}
}

func coverRegisterFile(fileName string, counter []uint32, pos []uint32, numStmts []uint16) {
	if 3*len(counter) != len(pos) || len(counter) != len(numStmts) {
		panic("coverage: mismatched sizes")
	}
	if coverCounters[fileName] != nil {
		// Already registered.
		return
	}
	coverCounters[fileName] = counter
	block := make([]testing.CoverBlock, len(counter))
	for i := range counter {
		block[i] = testing.CoverBlock{
			Line0: pos[3*i+0],
			Col0: uint16(pos[3*i+2]),
			Line1: pos[3*i+1],
			Col1: uint16(pos[3*i+2]>>16),
			Stmts: numStmts[i],
		}
	}
	coverBlocks[fileName] = block
}
{{end}}

func main() {
{{if .CoverEnabled}}
	testing.RegisterCover(testing.Cover{
		Mode: {{printf "%q" .CoverMode}},
		Counters: coverCounters,
		Blocks: coverBlocks,
		CoveredPackages: {{printf "%q" .Covered}},
	})
{{end}}
	m := testing.MainStart(matchString, tests, benchmarks, examples)
{{with .TestMain}}
	{{.Package}}.{{.Name}}(m)
{{else}}
	os.Exit(m.Run())
{{end}}
}

`))
