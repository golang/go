// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"internal/coverage"
	"internal/platform"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"cmd/go/internal/base"
	"cmd/go/internal/cache"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/lockedfile"
	"cmd/go/internal/modload"
	"cmd/go/internal/search"
	"cmd/go/internal/str"
	"cmd/go/internal/trace"
	"cmd/go/internal/work"
	"cmd/internal/test2json"

	"golang.org/x/mod/module"
)

// Break init loop.
func init() {
	CmdTest.Run = runTest
}

const testUsage = "go test [build/test flags] [packages] [build/test flags & test binary flags]"

var CmdTest = &base.Command{
	CustomFlags: true,
	UsageLine:   testUsage,
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
These additional files can contain test functions, benchmark functions, fuzz
tests and example functions. See 'go help testfunc' for more.
Each listed package causes the execution of a separate test binary.
Files whose names begin with "_" (including "_test.go") or "." are ignored.

Test files that declare a package with the suffix "_test" will be compiled as a
separate package, and then linked and run with the main test binary.

The go tool will ignore a directory named "testdata", making it available
to hold ancillary data needed by the tests.

As part of building a test binary, go test runs go vet on the package
and its test source files to identify significant problems. If go vet
finds any problems, go test reports those and does not run the test
binary. Only a high-confidence subset of the default go vet checks are
used. That subset is: atomic, bool, buildtags, directive, errorsas,
ifaceassert, nilfunc, printf, and stringintconv. You can see
the documentation for these and other vet tests via "go doc cmd/vet".
To disable the running of go vet, use the -vet=off flag. To run all
checks, use the -vet=all flag.

All test output and summary lines are printed to the go command's
standard output, even if the test printed them to its own standard
error. (The go command's standard error is reserved for printing
errors building the tests.)

The go command places $GOROOT/bin at the beginning of $PATH
in the test's environment, so that tests that execute
'go' commands use the same 'go' as the parent 'go test' command.

Go test runs in two different modes:

The first, called local directory mode, occurs when go test is
invoked with no package arguments (for example, 'go test' or 'go
test -v'). In this mode, go test compiles the package sources and
tests found in the current directory and then runs the resulting
test binary. In this mode, caching (discussed below) is disabled.
After the package test finishes, go test prints a summary line
showing the test status ('ok' or 'FAIL'), package name, and elapsed
time.

The second, called package list mode, occurs when go test is invoked
with explicit package arguments (for example 'go test math', 'go
test ./...', and even 'go test .'). In this mode, go test compiles
and tests each of the packages listed on the command line. If a
package test passes, go test prints only the final 'ok' summary
line. If a package test fails, go test prints the full test output.
If invoked with the -bench or -v flag, go test prints the full
output even for passing package tests, in order to display the
requested benchmark results or verbose logging. After the package
tests for all of the listed packages finish, and their output is
printed, go test prints a final 'FAIL' status if any package test
has failed.

In package list mode only, go test caches successful package test
results to avoid unnecessary repeated running of tests. When the
result of a test can be recovered from the cache, go test will
redisplay the previous output instead of running the test binary
again. When this happens, go test prints '(cached)' in place of the
elapsed time in the summary line.

The rule for a match in the cache is that the run involves the same
test binary and the flags on the command line come entirely from a
restricted set of 'cacheable' test flags, defined as -benchtime, -cpu,
-list, -parallel, -run, -short, -timeout, -failfast, -fullpath and -v.
If a run of go test has any test or non-test flags outside this set,
the result is not cached. To disable test caching, use any test flag
or argument other than the cacheable flags. The idiomatic way to disable
test caching explicitly is to use -count=1. Tests that open files within
the package's source root (usually $GOPATH) or that consult environment
variables only match future runs in which the files and environment
variables are unchanged. A cached test result is treated as executing
in no time at all, so a successful package test result will be cached and
reused regardless of -timeout setting.

In addition to the build flags, the flags handled by 'go test' itself are:

	-args
	    Pass the remainder of the command line (everything after -args)
	    to the test binary, uninterpreted and unchanged.
	    Because this flag consumes the remainder of the command line,
	    the package list (if present) must appear before this flag.

	-c
	    Compile the test binary to pkg.test in the current directory but do not run it
	    (where pkg is the last element of the package's import path).
	    The file name or target directory can be changed with the -o flag.

	-exec xprog
	    Run the test binary using xprog. The behavior is the same as
	    in 'go run'. See 'go help run' for details.

	-json
	    Convert test output to JSON suitable for automated processing.
	    See 'go doc test2json' for the encoding details.

	-o file
	    Compile the test binary to the named file.
	    The test still runs (unless -c or -i is specified).
	    If file ends in a slash or names an existing directory,
	    the test is written to pkg.test in that directory.

The test binary also accepts flags that control execution of the test; these
flags are also accessible by 'go test'. See 'go help testflag' for details.

For more about build flags, see 'go help build'.
For more about specifying packages, see 'go help packages'.

See also: go build, go vet.
`,
}

var HelpTestflag = &base.Command{
	UsageLine: "testflag",
	Short:     "testing flags",
	Long: `
The 'go test' command takes both flags that apply to 'go test' itself
and flags that apply to the resulting test binary.

Several of the flags control profiling and write an execution profile
suitable for "go tool pprof"; run "go tool pprof -h" for more
information. The --alloc_space, --alloc_objects, and --show_bytes
options of pprof control how the information is presented.

The following flags are recognized by the 'go test' command and
control the execution of any test:

	-bench regexp
	    Run only those benchmarks matching a regular expression.
	    By default, no benchmarks are run.
	    To run all benchmarks, use '-bench .' or '-bench=.'.
	    The regular expression is split by unbracketed slash (/)
	    characters into a sequence of regular expressions, and each
	    part of a benchmark's identifier must match the corresponding
	    element in the sequence, if any. Possible parents of matches
	    are run with b.N=1 to identify sub-benchmarks. For example,
	    given -bench=X/Y, top-level benchmarks matching X are run
	    with b.N=1 to find any sub-benchmarks matching Y, which are
	    then run in full.

	-benchtime t
	    Run enough iterations of each benchmark to take t, specified
	    as a time.Duration (for example, -benchtime 1h30s).
	    The default is 1 second (1s).
	    The special syntax Nx means to run the benchmark N times
	    (for example, -benchtime 100x).

	-count n
	    Run each test, benchmark, and fuzz seed n times (default 1).
	    If -cpu is set, run n times for each GOMAXPROCS value.
	    Examples are always run once. -count does not apply to
	    fuzz tests matched by -fuzz.

	-cover
	    Enable coverage analysis.
	    Note that because coverage works by annotating the source
	    code before compilation, compilation and test failures with
	    coverage enabled may report line numbers that don't correspond
	    to the original sources.

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

	-coverpkg pattern1,pattern2,pattern3
	    Apply coverage analysis in each test to packages matching the patterns.
	    The default is for each test to analyze only the package being tested.
	    See 'go help packages' for a description of package patterns.
	    Sets -cover.

	-cpu 1,2,4
	    Specify a list of GOMAXPROCS values for which the tests, benchmarks or
	    fuzz tests should be executed. The default is the current value
	    of GOMAXPROCS. -cpu does not apply to fuzz tests matched by -fuzz.

	-failfast
	    Do not start new tests after the first test failure.

	-fullpath
	    Show full file names in the error messages.

	-fuzz regexp
	    Run the fuzz test matching the regular expression. When specified,
	    the command line argument must match exactly one package within the
	    main module, and regexp must match exactly one fuzz test within
	    that package. Fuzzing will occur after tests, benchmarks, seed corpora
	    of other fuzz tests, and examples have completed. See the Fuzzing
	    section of the testing package documentation for details.

	-fuzztime t
	    Run enough iterations of the fuzz target during fuzzing to take t,
	    specified as a time.Duration (for example, -fuzztime 1h30s).
		The default is to run forever.
	    The special syntax Nx means to run the fuzz target N times
	    (for example, -fuzztime 1000x).

	-fuzzminimizetime t
	    Run enough iterations of the fuzz target during each minimization
	    attempt to take t, as specified as a time.Duration (for example,
	    -fuzzminimizetime 30s).
		The default is 60s.
	    The special syntax Nx means to run the fuzz target N times
	    (for example, -fuzzminimizetime 100x).

	-json
	    Log verbose output and test results in JSON. This presents the
	    same information as the -v flag in a machine-readable format.

	-list regexp
	    List tests, benchmarks, fuzz tests, or examples matching the regular
	    expression. No tests, benchmarks, fuzz tests, or examples will be run.
	    This will only list top-level tests. No subtest or subbenchmarks will be
	    shown.

	-parallel n
	    Allow parallel execution of test functions that call t.Parallel, and
	    fuzz targets that call t.Parallel when running the seed corpus.
	    The value of this flag is the maximum number of tests to run
	    simultaneously.
	    While fuzzing, the value of this flag is the maximum number of
	    subprocesses that may call the fuzz function simultaneously, regardless of
	    whether T.Parallel is called.
	    By default, -parallel is set to the value of GOMAXPROCS.
	    Setting -parallel to values higher than GOMAXPROCS may cause degraded
	    performance due to CPU contention, especially when fuzzing.
	    Note that -parallel only applies within a single test binary.
	    The 'go test' command may run tests for different packages
	    in parallel as well, according to the setting of the -p flag
	    (see 'go help build').

	-run regexp
	    Run only those tests, examples, and fuzz tests matching the regular
	    expression. For tests, the regular expression is split by unbracketed
	    slash (/) characters into a sequence of regular expressions, and each
	    part of a test's identifier must match the corresponding element in
	    the sequence, if any. Note that possible parents of matches are
	    run too, so that -run=X/Y matches and runs and reports the result
	    of all tests matching X, even those without sub-tests matching Y,
	    because it must run them to look for those sub-tests.
	    See also -skip.

	-short
	    Tell long-running tests to shorten their run time.
	    It is off by default but set during all.bash so that installing
	    the Go tree can run a sanity check but not spend time running
	    exhaustive tests.

	-shuffle off,on,N
	    Randomize the execution order of tests and benchmarks.
	    It is off by default. If -shuffle is set to on, then it will seed
	    the randomizer using the system clock. If -shuffle is set to an
	    integer N, then N will be used as the seed value. In both cases,
	    the seed will be reported for reproducibility.

	-skip regexp
	    Run only those tests, examples, fuzz tests, and benchmarks that
	    do not match the regular expression. Like for -run and -bench,
	    for tests and benchmarks, the regular expression is split by unbracketed
	    slash (/) characters into a sequence of regular expressions, and each
	    part of a test's identifier must match the corresponding element in
	    the sequence, if any.

	-timeout d
	    If a test binary runs longer than duration d, panic.
	    If d is 0, the timeout is disabled.
	    The default is 10 minutes (10m).

	-v
	    Verbose output: log all tests as they are run. Also print all
	    text from Log and Logf calls even if the test succeeds.

	-vet list
	    Configure the invocation of "go vet" during "go test"
	    to use the comma-separated list of vet checks.
	    If list is empty, "go test" runs "go vet" with a curated list of
	    checks believed to be always worth addressing.
	    If list is "off", "go test" does not run "go vet" at all.

The following flags are also recognized by 'go test' and can be used to
profile the tests during execution:

	-benchmem
	    Print memory allocation statistics for benchmarks.
	    Allocations made in C or using C.malloc are not counted.

	-blockprofile block.out
	    Write a goroutine blocking profile to the specified file
	    when all tests are complete.
	    Writes test binary as -c would.

	-blockprofilerate n
	    Control the detail provided in goroutine blocking profiles by
	    calling runtime.SetBlockProfileRate with n.
	    See 'go doc runtime.SetBlockProfileRate'.
	    The profiler aims to sample, on average, one blocking event every
	    n nanoseconds the program spends blocked. By default,
	    if -test.blockprofile is set without this flag, all blocking events
	    are recorded, equivalent to -test.blockprofilerate=1.

	-coverprofile cover.out
	    Write a coverage profile to the file after all tests have passed.
	    Sets -cover.

	-cpuprofile cpu.out
	    Write a CPU profile to the specified file before exiting.
	    Writes test binary as -c would.

	-memprofile mem.out
	    Write an allocation profile to the file after all tests have passed.
	    Writes test binary as -c would.

	-memprofilerate n
	    Enable more precise (and expensive) memory allocation profiles by
	    setting runtime.MemProfileRate. See 'go doc runtime.MemProfileRate'.
	    To profile all memory allocations, use -test.memprofilerate=1.

	-mutexprofile mutex.out
	    Write a mutex contention profile to the specified file
	    when all tests are complete.
	    Writes test binary as -c would.

	-mutexprofilefraction n
	    Sample 1 in n stack traces of goroutines holding a
	    contended mutex.

	-outputdir directory
	    Place output files from profiling in the specified directory,
	    by default the directory in which "go test" is running.

	-trace trace.out
	    Write an execution trace to the specified file before exiting.

Each of these flags is also recognized with an optional 'test.' prefix,
as in -test.v. When invoking the generated test binary (the result of
'go test -c') directly, however, the prefix is mandatory.

The 'go test' command rewrites or removes recognized flags,
as appropriate, both before and after the optional package list,
before invoking the test binary.

For instance, the command

	go test -v -myflag testdata -cpuprofile=prof.out -x

will compile the test binary and then run it as

	pkg.test -test.v -myflag testdata -test.cpuprofile=prof.out

(The -x flag is removed because it applies only to the go command's
execution, not to the test itself.)

The test flags that generate profiles (other than for coverage) also
leave the test binary in pkg.test for use when analyzing the profiles.

When 'go test' runs a test binary, it does so from within the
corresponding package's source code directory. Depending on the test,
it may be necessary to do the same when invoking a generated test
binary directly. Because that directory may be located within the
module cache, which may be read-only and is verified by checksums, the
test must not write to it or any other directory within the module
unless explicitly requested by the user (such as with the -fuzz flag,
which writes failures to testdata/fuzz).

The command-line package list, if present, must appear before any
flag not known to the go test command. Continuing the example above,
the package list would have to appear before -myflag, but could appear
on either side of -v.

When 'go test' runs in package list mode, 'go test' caches successful
package test results to avoid unnecessary repeated running of tests. To
disable test caching, use any test flag or argument other than the
cacheable flags. The idiomatic way to disable test caching explicitly
is to use -count=1.

To keep an argument for a test binary from being interpreted as a
known flag or a package name, use -args (see 'go help test') which
passes the remainder of the command line through to the test binary
uninterpreted and unaltered.

For instance, the command

	go test -v -args -x -v

will compile the test binary and then run it as

	pkg.test -test.v -x -v

Similarly,

	go test -args math

will compile the test binary and then run it as

	pkg.test math

In the first example, the -x and the second -v are passed through to the
test binary unchanged and with no effect on the go command itself.
In the second example, the argument math is passed through to the test
binary, instead of being interpreted as the package list.
`,
}

var HelpTestfunc = &base.Command{
	UsageLine: "testfunc",
	Short:     "testing functions",
	Long: `
The 'go test' command expects to find test, benchmark, and example functions
in the "*_test.go" files corresponding to the package under test.

A test function is one named TestXxx (where Xxx does not start with a
lower case letter) and should have the signature,

	func TestXxx(t *testing.T) { ... }

A benchmark function is one named BenchmarkXxx and should have the signature,

	func BenchmarkXxx(b *testing.B) { ... }

A fuzz test is one named FuzzXxx and should have the signature,

	func FuzzXxx(f *testing.F) { ... }

An example function is similar to a test function but, instead of using
*testing.T to report success or failure, prints output to os.Stdout.
If the last comment in the function starts with "Output:" then the output
is compared exactly against the comment (see examples below). If the last
comment begins with "Unordered output:" then the output is compared to the
comment, however the order of the lines is ignored. An example with no such
comment is compiled but not executed. An example with no text after
"Output:" is compiled, executed, and expected to produce no output.

Godoc displays the body of ExampleXxx to demonstrate the use
of the function, constant, or variable Xxx. An example of a method M with
receiver type T or *T is named ExampleT_M. There may be multiple examples
for a given function, constant, or variable, distinguished by a trailing _xxx,
where xxx is a suffix not beginning with an upper case letter.

Here is an example of an example:

	func ExamplePrintln() {
		Println("The output of\nthis example.")
		// Output: The output of
		// this example.
	}

Here is another example where the ordering of the output is ignored:

	func ExamplePerm() {
		for _, value := range Perm(4) {
			fmt.Println(value)
		}

		// Unordered output: 4
		// 2
		// 1
		// 3
		// 0
	}

The entire test file is presented as the example when it contains a single
example function, at least one other function, type, variable, or constant
declaration, and no tests, benchmarks, or fuzz tests.

See the documentation of the testing package for more information.
`,
}

var (
	testBench        string                            // -bench flag
	testC            bool                              // -c flag
	testCoverPkgs    []*load.Package                   // -coverpkg flag
	testCoverProfile string                            // -coverprofile flag
	testFailFast     bool                              // -failfast flag
	testFuzz         string                            // -fuzz flag
	testJSON         bool                              // -json flag
	testList         string                            // -list flag
	testO            string                            // -o flag
	testOutputDir    outputdirFlag                     // -outputdir flag
	testShuffle      shuffleFlag                       // -shuffle flag
	testTimeout      time.Duration                     // -timeout flag
	testV            testVFlag                         // -v flag
	testVet          = vetFlag{flags: defaultVetFlags} // -vet flag
)

type testVFlag struct {
	on   bool // -v is set in some form
	json bool // -v=test2json is set, to make output better for test2json
}

func (*testVFlag) IsBoolFlag() bool { return true }

func (f *testVFlag) Set(arg string) error {
	if v, err := strconv.ParseBool(arg); err == nil {
		f.on = v
		f.json = false
		return nil
	}
	if arg == "test2json" {
		f.on = true
		f.json = arg == "test2json"
		return nil
	}
	return fmt.Errorf("invalid flag -test.v=%s", arg)
}

func (f *testVFlag) String() string {
	if f.json {
		return "test2json"
	}
	if f.on {
		return "true"
	}
	return "false"
}

var (
	testArgs []string
	pkgArgs  []string
	pkgs     []*load.Package

	testHelp bool // -help option passed to test via -args

	testKillTimeout    = 100 * 365 * 24 * time.Hour // backup alarm; defaults to about a century if no timeout is set
	testWaitDelay      time.Duration                // how long to wait for output to close after a test binary exits; zero means unlimited
	testCacheExpire    time.Time                    // ignore cached test results before this time
	testShouldFailFast atomic.Bool                  // signals pending tests to fail fast

	testBlockProfile, testCPUProfile, testMemProfile, testMutexProfile, testTrace string // profiling flag that limits test to one package

	testODir = false
)

// testProfile returns the name of an arbitrary single-package profiling flag
// that is set, if any.
func testProfile() string {
	switch {
	case testBlockProfile != "":
		return "-blockprofile"
	case testCPUProfile != "":
		return "-cpuprofile"
	case testMemProfile != "":
		return "-memprofile"
	case testMutexProfile != "":
		return "-mutexprofile"
	case testTrace != "":
		return "-trace"
	default:
		return ""
	}
}

// testNeedBinary reports whether the test needs to keep the binary around.
func testNeedBinary() bool {
	switch {
	case testBlockProfile != "":
		return true
	case testCPUProfile != "":
		return true
	case testMemProfile != "":
		return true
	case testMutexProfile != "":
		return true
	case testO != "":
		return true
	default:
		return false
	}
}

// testShowPass reports whether the output for a passing test should be shown.
func testShowPass() bool {
	return testV.on || testList != "" || testHelp
}

var defaultVetFlags = []string{
	// TODO(rsc): Decide which tests are enabled by default.
	// See golang.org/issue/18085.
	// "-asmdecl",
	// "-assign",
	"-atomic",
	"-bool",
	"-buildtags",
	// "-cgocall",
	// "-composites",
	// "-copylocks",
	"-directive",
	"-errorsas",
	// "-httpresponse",
	"-ifaceassert",
	// "-lostcancel",
	// "-methods",
	"-nilfunc",
	"-printf",
	// "-rangeloops",
	// "-shift",
	"-slog",
	"-stringintconv",
	// "-structtags",
	// "-tests",
	// "-unreachable",
	// "-unsafeptr",
	// "-unusedresult",
}

func runTest(ctx context.Context, cmd *base.Command, args []string) {
	pkgArgs, testArgs = testFlags(args)
	modload.InitWorkfile() // The test command does custom flag processing; initialize workspaces after that.

	if cfg.DebugTrace != "" {
		var close func() error
		var err error
		ctx, close, err = trace.Start(ctx, cfg.DebugTrace)
		if err != nil {
			base.Fatalf("failed to start trace: %v", err)
		}
		defer func() {
			if err := close(); err != nil {
				base.Fatalf("failed to stop trace: %v", err)
			}
		}()
	}

	ctx, span := trace.StartSpan(ctx, fmt.Sprint("Running ", cmd.Name(), " command"))
	defer span.Done()

	work.FindExecCmd() // initialize cached result

	work.BuildInit()
	work.VetFlags = testVet.flags
	work.VetExplicit = testVet.explicit

	pkgOpts := load.PackageOpts{ModResolveTests: true}
	pkgs = load.PackagesAndErrors(ctx, pkgOpts, pkgArgs)
	load.CheckPackageErrors(pkgs)
	if len(pkgs) == 0 {
		base.Fatalf("no packages to test")
	}

	if testFuzz != "" {
		if !platform.FuzzSupported(cfg.Goos, cfg.Goarch) {
			base.Fatalf("-fuzz flag is not supported on %s/%s", cfg.Goos, cfg.Goarch)
		}
		if len(pkgs) != 1 {
			base.Fatalf("cannot use -fuzz flag with multiple packages")
		}
		if testCoverProfile != "" {
			base.Fatalf("cannot use -coverprofile flag with -fuzz flag")
		}
		if profileFlag := testProfile(); profileFlag != "" {
			base.Fatalf("cannot use %s flag with -fuzz flag", profileFlag)
		}

		// Reject the '-fuzz' flag if the package is outside the main module.
		// Otherwise, if fuzzing identifies a failure it could corrupt checksums in
		// the module cache (or permanently alter the behavior of std tests for all
		// users) by writing the failing input to the package's testdata directory.
		// (See https://golang.org/issue/48495 and test_fuzz_modcache.txt.)
		mainMods := modload.MainModules
		if m := pkgs[0].Module; m != nil && m.Path != "" {
			if !mainMods.Contains(m.Path) {
				base.Fatalf("cannot use -fuzz flag on package outside the main module")
			}
		} else if pkgs[0].Standard && modload.Enabled() {
			// Because packages in 'std' and 'cmd' are part of the standard library,
			// they are only treated as part of a module in 'go mod' subcommands and
			// 'go get'. However, we still don't want to accidentally corrupt their
			// testdata during fuzzing, nor do we want to fail with surprising errors
			// if GOROOT isn't writable (as is often the case for Go toolchains
			// installed through package managers).
			//
			// If the user is requesting to fuzz a standard-library package, ensure
			// that they are in the same module as that package (just like when
			// fuzzing any other package).
			if strings.HasPrefix(pkgs[0].ImportPath, "cmd/") {
				if !mainMods.Contains("cmd") || !mainMods.InGorootSrc(module.Version{Path: "cmd"}) {
					base.Fatalf("cannot use -fuzz flag on package outside the main module")
				}
			} else {
				if !mainMods.Contains("std") || !mainMods.InGorootSrc(module.Version{Path: "std"}) {
					base.Fatalf("cannot use -fuzz flag on package outside the main module")
				}
			}
		}
	}
	if testProfile() != "" && len(pkgs) != 1 {
		base.Fatalf("cannot use %s flag with multiple packages", testProfile())
	}

	if testO != "" {
		if strings.HasSuffix(testO, "/") || strings.HasSuffix(testO, string(os.PathSeparator)) {
			testODir = true
		} else if fi, err := os.Stat(testO); err == nil && fi.IsDir() {
			testODir = true
		}
	}

	if len(pkgs) > 1 && (testC || testO != "") && !base.IsNull(testO) {
		if testO != "" && !testODir {
			base.Fatalf("with multiple packages, -o must refer to a directory or %s", os.DevNull)
		}

		pkgsForBinary := map[string][]*load.Package{}

		for _, p := range pkgs {
			testBinary := testBinaryName(p)
			pkgsForBinary[testBinary] = append(pkgsForBinary[testBinary], p)
		}

		for testBinary, pkgs := range pkgsForBinary {
			if len(pkgs) > 1 {
				var buf strings.Builder
				for _, pkg := range pkgs {
					buf.WriteString(pkg.ImportPath)
					buf.WriteString("\n")
				}

				base.Errorf("cannot write test binary %s for multiple packages:\n%s", testBinary, buf.String())
			}
		}

		base.ExitIfErrors()
	}

	initCoverProfile()
	defer closeCoverProfile()

	// If a test timeout is finite, set our kill timeout
	// to that timeout plus one minute. This is a backup alarm in case
	// the test wedges with a goroutine spinning and its background
	// timer does not get a chance to fire.
	// Don't set this if fuzzing, since it should be able to run
	// indefinitely.
	if testTimeout > 0 && testFuzz == "" {
		// The WaitDelay for the test process depends on both the OS I/O and
		// scheduling overhead and the amount of I/O generated by the test just
		// before it exits. We set the minimum at 5 seconds to account for the OS
		// overhead, and scale it up from there proportional to the overall test
		// timeout on the assumption that the time to write and read a goroutine
		// dump from a timed-out test process scales roughly with the overall
		// running time of the test.
		//
		// This is probably too generous when the timeout is very long, but it seems
		// better to hard-code a scale factor than to hard-code a constant delay.
		if wd := testTimeout / 10; wd < 5*time.Second {
			testWaitDelay = 5 * time.Second
		} else {
			testWaitDelay = wd
		}

		// We expect the test binary to terminate itself (and dump stacks) after
		// exactly testTimeout. We give it up to one WaitDelay or one minute,
		// whichever is longer, to finish dumping stacks before we send it an
		// external signal: if the process has a lot of goroutines, dumping stacks
		// after the timeout can take a while.
		//
		// After the signal is delivered, the test process may have up to one
		// additional WaitDelay to finish writing its output streams.
		if testWaitDelay < 1*time.Minute {
			testKillTimeout = testTimeout + 1*time.Minute
		} else {
			testKillTimeout = testTimeout + testWaitDelay
		}
	}

	// Read testcache expiration time, if present.
	// (We implement go clean -testcache by writing an expiration date
	// instead of searching out and deleting test result cache entries.)
	if dir, _ := cache.DefaultDir(); dir != "off" {
		if data, _ := lockedfile.Read(filepath.Join(dir, "testexpire.txt")); len(data) > 0 && data[len(data)-1] == '\n' {
			if t, err := strconv.ParseInt(string(data[:len(data)-1]), 10, 64); err == nil {
				testCacheExpire = time.Unix(0, t)
			}
		}
	}

	b := work.NewBuilder("")
	defer func() {
		if err := b.Close(); err != nil {
			base.Fatal(err)
		}
	}()

	var builds, runs, prints []*work.Action
	var writeCoverMetaAct *work.Action

	if cfg.BuildCoverPkg != nil {
		match := make([]func(*load.Package) bool, len(cfg.BuildCoverPkg))
		for i := range cfg.BuildCoverPkg {
			match[i] = load.MatchPackage(cfg.BuildCoverPkg[i], base.Cwd())
		}

		// Select for coverage all dependencies matching the -coverpkg
		// patterns.
		plist := load.TestPackageList(ctx, pkgOpts, pkgs)
		testCoverPkgs = load.SelectCoverPackages(plist, match, "test")
		if cfg.Experiment.CoverageRedesign && len(testCoverPkgs) > 0 {
			// create a new singleton action that will collect up the
			// meta-data files from all of the packages mentioned in
			// "-coverpkg" and write them to a summary file. This new
			// action will depend on all the build actions for the
			// test packages, and all the run actions for these
			// packages will depend on it. Motivating example:
			// supposed we have a top level directory with three
			// package subdirs, "a", "b", and "c", and
			// from the top level, a user runs "go test -coverpkg=./... ./...".
			// This will result in (roughly) the following action graph:
			//
			//	build("a")       build("b")         build("c")
			//	    |               |                   |
			//	link("a.test")   link("b.test")     link("c.test")
			//	    |               |                   |
			//	run("a.test")    run("b.test")      run("c.test")
			//	    |               |                   |
			//	  print          print              print
			//
			// When -coverpkg=<pattern> is in effect, we want to
			// express the coverage percentage for each package as a
			// fraction of *all* the statements that match the
			// pattern, hence if "c" doesn't import "a", we need to
			// pass as meta-data file for "a" (emitted during the
			// package "a" build) to the package "c" run action, so
			// that it can be incorporated with "c"'s regular
			// metadata. To do this, we add edges from each compile
			// action to a "writeCoverMeta" action, then from the
			// writeCoverMeta action to each run action. Updated
			// graph:
			//
			//	build("a")       build("b")         build("c")
			//	    |   \       /   |               /   |
			//	    |    v     v    |              /    |
			//	    |   writemeta <-|-------------+     |
			//	    |         |||   |                   |
			//	    |         ||\   |                   |
			//	link("a.test")/\ \  link("b.test")      link("c.test")
			//	    |        /  \ +-|--------------+    |
			//	    |       /    \  |               \   |
			//	    |      v      v |                v  |
			//	run("a.test")    run("b.test")      run("c.test")
			//	    |               |                   |
			//	  print          print              print
			//
			writeCoverMetaAct = &work.Action{
				Mode:   "write coverage meta-data file",
				Actor:  work.ActorFunc(work.WriteCoverMetaFilesFile),
				Objdir: b.NewObjdir(),
			}
			for _, p := range testCoverPkgs {
				p.Internal.Cover.GenMeta = true
			}
		}
	}

	// Inform the compiler that it should instrument the binary at
	// build-time when fuzzing is enabled.
	if testFuzz != "" {
		// Don't instrument packages which may affect coverage guidance but are
		// unlikely to be useful. Most of these are used by the testing or
		// internal/fuzz packages concurrently with fuzzing.
		var skipInstrumentation = map[string]bool{
			"context":       true,
			"internal/fuzz": true,
			"reflect":       true,
			"runtime":       true,
			"sync":          true,
			"sync/atomic":   true,
			"syscall":       true,
			"testing":       true,
			"time":          true,
		}
		for _, p := range load.TestPackageList(ctx, pkgOpts, pkgs) {
			if !skipInstrumentation[p.ImportPath] {
				p.Internal.FuzzInstrument = true
			}
		}
	}

	// Collect all the packages imported by the packages being tested.
	allImports := make(map[*load.Package]bool)
	for _, p := range pkgs {
		if p.Error != nil && p.Error.IsImportCycle {
			continue
		}
		for _, p1 := range p.Internal.Imports {
			allImports[p1] = true
		}
	}

	if cfg.BuildCover {
		for _, p := range pkgs {
			// sync/atomic import is inserted by the cover tool if
			// we're using atomic mode (and not compiling
			// sync/atomic package itself). See #18486 and #57445.
			// Note that this needs to be done prior to any of the
			// builderTest invocations below, due to the fact that
			// a given package in the 'pkgs' list may import
			// package Q which appears later in the list (if this
			// happens we'll wind up building the Q compile action
			// before updating its deps to include sync/atomic).
			if cfg.BuildCoverMode == "atomic" && p.ImportPath != "sync/atomic" {
				load.EnsureImport(p, "sync/atomic")
			}
			// Tag the package for static meta-data generation if no
			// test files (this works only with the new coverage
			// design). Do this here (as opposed to in builderTest) so
			// as to handle the case where we're testing multiple
			// packages and one of the earlier packages imports a
			// later package. Note that if -coverpkg is in effect
			// p.Internal.Cover.GenMeta will wind up being set for
			// all matching packages.
			if len(p.TestGoFiles)+len(p.XTestGoFiles) == 0 &&
				cfg.BuildCoverPkg == nil &&
				cfg.Experiment.CoverageRedesign {
				p.Internal.Cover.GenMeta = true
			}
		}
	}

	// Prepare build + run + print actions for all packages being tested.
	for _, p := range pkgs {
		buildTest, runTest, printTest, err := builderTest(b, ctx, pkgOpts, p, allImports[p], writeCoverMetaAct)
		if err != nil {
			str := err.Error()
			str = strings.TrimPrefix(str, "\n")
			if p.ImportPath != "" {
				base.Errorf("# %s\n%s", p.ImportPath, str)
			} else {
				base.Errorf("%s", str)
			}
			fmt.Printf("FAIL\t%s [setup failed]\n", p.ImportPath)
			continue
		}
		builds = append(builds, buildTest)
		runs = append(runs, runTest)
		prints = append(prints, printTest)
	}

	// Order runs for coordinating start JSON prints.
	ch := make(chan struct{})
	close(ch)
	for _, a := range runs {
		if r, ok := a.Actor.(*runTestActor); ok {
			r.prev = ch
			ch = make(chan struct{})
			r.next = ch
		}
	}

	// Ultimately the goal is to print the output.
	root := &work.Action{Mode: "go test", Actor: work.ActorFunc(printExitStatus), Deps: prints}

	// Force the printing of results to happen in order,
	// one at a time.
	for i, a := range prints {
		if i > 0 {
			a.Deps = append(a.Deps, prints[i-1])
		}
	}

	// Force benchmarks to run in serial.
	if !testC && (testBench != "") {
		// The first run must wait for all builds.
		// Later runs must wait for the previous run's print.
		for i, run := range runs {
			if i == 0 {
				run.Deps = append(run.Deps, builds...)
			} else {
				run.Deps = append(run.Deps, prints[i-1])
			}
		}
	}

	b.Do(ctx, root)
}

var windowsBadWords = []string{
	"install",
	"patch",
	"setup",
	"update",
}

func builderTest(b *work.Builder, ctx context.Context, pkgOpts load.PackageOpts, p *load.Package, imported bool, writeCoverMetaAct *work.Action) (buildAction, runAction, printAction *work.Action, err error) {
	if len(p.TestGoFiles)+len(p.XTestGoFiles) == 0 {
		if cfg.BuildCover && cfg.Experiment.CoverageRedesign {
			if p.Internal.Cover.GenMeta {
				p.Internal.Cover.Mode = cfg.BuildCoverMode
			}
		}
		build := b.CompileAction(work.ModeBuild, work.ModeBuild, p)
		run := &work.Action{
			Mode:       "test run",
			Actor:      new(runTestActor),
			Deps:       []*work.Action{build},
			Objdir:     b.NewObjdir(),
			Package:    p,
			IgnoreFail: true, // run (prepare output) even if build failed
		}
		if writeCoverMetaAct != nil && build.Actor != nil {
			// There is no real "run" for this package (since there
			// are no tests), but if coverage is turned on, we can
			// collect coverage data for the code in the package by
			// asking cmd/cover for a static meta-data file as part of
			// the package build. This static meta-data file is then
			// consumed by a pseudo-action (writeCoverMetaAct) that
			// adds it to a summary file, then this summary file is
			// consumed by the various "run test" actions. Below we
			// add a dependence edge between the build action and the
			// "write meta files" pseudo-action, and then another dep
			// from writeCoverMetaAct to the run action. See the
			// comment in runTest() at the definition of
			// writeCoverMetaAct for more details.
			run.Deps = append(run.Deps, writeCoverMetaAct)
			writeCoverMetaAct.Deps = append(writeCoverMetaAct.Deps, build)
		}
		addTestVet(b, p, run, nil)
		print := &work.Action{
			Mode:       "test print",
			Actor:      work.ActorFunc(builderPrintTest),
			Deps:       []*work.Action{run},
			Package:    p,
			IgnoreFail: true, // print even if test failed
		}
		return build, run, print, nil
	}

	// Build Package structs describing:
	//	pmain - pkg.test binary
	//	ptest - package + test files
	//	pxtest - package of external test files
	var cover *load.TestCover
	if cfg.BuildCover {
		cover = &load.TestCover{
			Mode:  cfg.BuildCoverMode,
			Local: cfg.BuildCoverPkg == nil,
			Pkgs:  testCoverPkgs,
			Paths: cfg.BuildCoverPkg,
		}
	}
	pmain, ptest, pxtest, err := load.TestPackagesFor(ctx, pkgOpts, p, cover)
	if err != nil {
		return nil, nil, nil, err
	}

	// If imported is true then this package is imported by some
	// package being tested. Make building the test version of the
	// package depend on building the non-test version, so that we
	// only report build errors once. Issue #44624.
	if imported && ptest != p {
		buildTest := b.CompileAction(work.ModeBuild, work.ModeBuild, ptest)
		buildP := b.CompileAction(work.ModeBuild, work.ModeBuild, p)
		buildTest.Deps = append(buildTest.Deps, buildP)
	}

	testBinary := testBinaryName(p)

	testDir := b.NewObjdir()
	if err := b.BackgroundShell().Mkdir(testDir); err != nil {
		return nil, nil, nil, err
	}

	pmain.Dir = testDir
	pmain.Internal.OmitDebug = !testC && !testNeedBinary()
	if pmain.ImportPath == "runtime.test" {
		// The runtime package needs a symbolized binary for its tests.
		// See runtime/unsafepoint_test.go.
		pmain.Internal.OmitDebug = false
	}

	if !cfg.BuildN {
		// writeTestmain writes _testmain.go,
		// using the test description gathered in t.
		if err := os.WriteFile(testDir+"_testmain.go", *pmain.Internal.TestmainGo, 0666); err != nil {
			return nil, nil, nil, err
		}
	}

	// Set compile objdir to testDir we've already created,
	// so that the default file path stripping applies to _testmain.go.
	b.CompileAction(work.ModeBuild, work.ModeBuild, pmain).Objdir = testDir

	a := b.LinkAction(work.ModeBuild, work.ModeBuild, pmain)
	a.Target = testDir + testBinary + cfg.ExeSuffix
	if cfg.Goos == "windows" {
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
				a.Target = testDir + "test.test" + cfg.ExeSuffix
				break
			}
		}
	}
	buildAction = a
	var installAction, cleanAction *work.Action
	if testC || testNeedBinary() {
		// -c or profiling flag: create action to copy binary to ./test.out.
		target := filepath.Join(base.Cwd(), testBinary+cfg.ExeSuffix)
		isNull := false

		if testO != "" {
			target = testO

			if testODir {
				if filepath.IsAbs(target) {
					target = filepath.Join(target, testBinary+cfg.ExeSuffix)
				} else {
					target = filepath.Join(base.Cwd(), target, testBinary+cfg.ExeSuffix)
				}
			} else {
				if base.IsNull(target) {
					isNull = true
				} else if !filepath.IsAbs(target) {
					target = filepath.Join(base.Cwd(), target)
				}
			}
		}

		if isNull {
			runAction = buildAction
		} else {
			pmain.Target = target
			installAction = &work.Action{
				Mode:    "test build",
				Actor:   work.ActorFunc(work.BuildInstallFunc),
				Deps:    []*work.Action{buildAction},
				Package: pmain,
				Target:  target,
			}
			runAction = installAction // make sure runAction != nil even if not running test
		}
	}

	var vetRunAction *work.Action
	if testC {
		printAction = &work.Action{Mode: "test print (nop)", Package: p, Deps: []*work.Action{runAction}} // nop
		vetRunAction = printAction
	} else {
		// run test
		rta := &runTestActor{
			writeCoverMetaAct: writeCoverMetaAct,
		}
		runAction = &work.Action{
			Mode:       "test run",
			Actor:      rta,
			Deps:       []*work.Action{buildAction},
			Package:    p,
			IgnoreFail: true, // run (prepare output) even if build failed
			TryCache:   rta.c.tryCache,
		}
		if writeCoverMetaAct != nil {
			// If writeCoverMetaAct != nil, this indicates that our
			// "go test -coverpkg" run actions will need to read the
			// meta-files summary file written by writeCoverMetaAct,
			// so add a dependence edge from writeCoverMetaAct to the
			// run action.
			runAction.Deps = append(runAction.Deps, writeCoverMetaAct)
			if !p.IsTestOnly() {
				// Package p is not test only, meaning that the build
				// action for p may generate a static meta-data file.
				// Add a dependence edge from p to writeCoverMetaAct,
				// which needs to know the name of that meta-data
				// file.
				compileAction := b.CompileAction(work.ModeBuild, work.ModeBuild, p)
				writeCoverMetaAct.Deps = append(writeCoverMetaAct.Deps, compileAction)
			}
		}
		runAction.Objdir = testDir
		vetRunAction = runAction
		cleanAction = &work.Action{
			Mode:       "test clean",
			Actor:      work.ActorFunc(builderCleanTest),
			Deps:       []*work.Action{runAction},
			Package:    p,
			IgnoreFail: true, // clean even if test failed
			Objdir:     testDir,
		}
		printAction = &work.Action{
			Mode:       "test print",
			Actor:      work.ActorFunc(builderPrintTest),
			Deps:       []*work.Action{cleanAction},
			Package:    p,
			IgnoreFail: true, // print even if test failed
		}
	}

	if len(ptest.GoFiles)+len(ptest.CgoFiles) > 0 {
		addTestVet(b, ptest, vetRunAction, installAction)
	}
	if pxtest != nil {
		addTestVet(b, pxtest, vetRunAction, installAction)
	}

	if installAction != nil {
		if runAction != installAction {
			installAction.Deps = append(installAction.Deps, runAction)
		}
		if cleanAction != nil {
			cleanAction.Deps = append(cleanAction.Deps, installAction)
		}
	}

	return buildAction, runAction, printAction, nil
}

func addTestVet(b *work.Builder, p *load.Package, runAction, installAction *work.Action) {
	if testVet.off {
		return
	}

	vet := b.VetAction(work.ModeBuild, work.ModeBuild, p)
	runAction.Deps = append(runAction.Deps, vet)
	// Install will clean the build directory.
	// Make sure vet runs first.
	// The install ordering in b.VetAction does not apply here
	// because we are using a custom installAction (created above).
	if installAction != nil {
		installAction.Deps = append(installAction.Deps, vet)
	}
}

var noTestsToRun = []byte("\ntesting: warning: no tests to run\n")
var noFuzzTestsToFuzz = []byte("\ntesting: warning: no fuzz tests to fuzz\n")
var tooManyFuzzTestsToFuzz = []byte("\ntesting: warning: -fuzz matches more than one fuzz test, won't fuzz\n")

// runTestActor is the actor for running a test.
type runTestActor struct {
	c runCache

	// writeCoverMetaAct points to the pseudo-action for collecting
	// coverage meta-data files for selected -cover test runs. See the
	// comment in runTest at the definition of writeCoverMetaAct for
	// more details.
	writeCoverMetaAct *work.Action

	// sequencing of json start messages, to preserve test order
	prev <-chan struct{} // wait to start until prev is closed
	next chan<- struct{} // close next once the next test can start.
}

// runCache is the cache for running a single test.
type runCache struct {
	disableCache bool // cache should be disabled for this run

	buf *bytes.Buffer
	id1 cache.ActionID
	id2 cache.ActionID
}

// stdoutMu and lockedStdout provide a locked standard output
// that guarantees never to interlace writes from multiple
// goroutines, so that we can have multiple JSON streams writing
// to a lockedStdout simultaneously and know that events will
// still be intelligible.
var stdoutMu sync.Mutex

type lockedStdout struct{}

func (lockedStdout) Write(b []byte) (int, error) {
	stdoutMu.Lock()
	defer stdoutMu.Unlock()
	return os.Stdout.Write(b)
}

func (r *runTestActor) Act(b *work.Builder, ctx context.Context, a *work.Action) error {
	sh := b.Shell(a)

	// Wait for previous test to get started and print its first json line.
	select {
	case <-r.prev:
		// If should fail fast then release next test and exit.
		if testShouldFailFast.Load() {
			close(r.next)
			return nil
		}
	case <-base.Interrupted:
		// We can't wait for the previous test action to complete: we don't start
		// new actions after an interrupt, so if that action wasn't already running
		// it might never happen. Instead, just don't log anything for this action.
		base.SetExitStatus(1)
		return nil
	}

	var stdout io.Writer = os.Stdout
	var err error
	if testJSON {
		json := test2json.NewConverter(lockedStdout{}, a.Package.ImportPath, test2json.Timestamp)
		defer func() {
			json.Exited(err)
			json.Close()
		}()
		stdout = json
	}

	// Release next test to start (test2json.NewConverter writes the start event).
	close(r.next)

	if a.Failed {
		// We were unable to build the binary.
		a.Failed = false
		fmt.Fprintf(stdout, "FAIL\t%s [build failed]\n", a.Package.ImportPath)
		// Tell the JSON converter that this was a failure, not a passing run.
		err = errors.New("build failed")
		base.SetExitStatus(1)
		return nil
	}

	coverProfTempFile := func(a *work.Action) string {
		if a.Objdir == "" {
			panic("internal error: objdir not set in coverProfTempFile")
		}
		return a.Objdir + "_cover_.out"
	}

	if p := a.Package; len(p.TestGoFiles)+len(p.XTestGoFiles) == 0 {
		reportNoTestFiles := true
		if cfg.BuildCover && cfg.Experiment.CoverageRedesign && p.Internal.Cover.GenMeta {
			if err := sh.Mkdir(a.Objdir); err != nil {
				return err
			}
			mf, err := work.BuildActionCoverMetaFile(a)
			if err != nil {
				return err
			} else if mf != "" {
				reportNoTestFiles = false
				// Write out "percent statements covered".
				if err := work.WriteCoveragePercent(b, a, mf, stdout); err != nil {
					return err
				}
				// If -coverprofile is in effect, then generate a
				// coverage profile fragment for this package and
				// merge it with the final -coverprofile output file.
				if coverMerge.f != nil {
					cp := coverProfTempFile(a)
					if err := work.WriteCoverageProfile(b, a, mf, cp, stdout); err != nil {
						return err
					}
					mergeCoverProfile(stdout, cp)
				}
			}
		}
		if reportNoTestFiles {
			fmt.Fprintf(stdout, "?   \t%s\t[no test files]\n", p.ImportPath)
		}
		return nil
	}

	var buf bytes.Buffer
	if len(pkgArgs) == 0 || testBench != "" || testFuzz != "" {
		// Stream test output (no buffering) when no package has
		// been given on the command line (implicit current directory)
		// or when benchmarking or fuzzing.
		// No change to stdout.
	} else {
		// If we're only running a single package under test or if parallelism is
		// set to 1, and if we're displaying all output (testShowPass), we can
		// hurry the output along, echoing it as soon as it comes in.
		// We still have to copy to &buf for caching the result. This special
		// case was introduced in Go 1.5 and is intentionally undocumented:
		// the exact details of output buffering are up to the go command and
		// subject to change. It would be nice to remove this special case
		// entirely, but it is surely very helpful to see progress being made
		// when tests are run on slow single-CPU ARM systems.
		//
		// If we're showing JSON output, then display output as soon as
		// possible even when multiple tests are being run: the JSON output
		// events are attributed to specific package tests, so interlacing them
		// is OK.
		if testShowPass() && (len(pkgs) == 1 || cfg.BuildP == 1) || testJSON {
			// Write both to stdout and buf, for possible saving
			// to cache, and for looking for the "no tests to run" message.
			stdout = io.MultiWriter(stdout, &buf)
		} else {
			stdout = &buf
		}
	}

	if r.c.buf == nil {
		// We did not find a cached result using the link step action ID,
		// so we ran the link step. Try again now with the link output
		// content ID. The attempt using the action ID makes sure that
		// if the link inputs don't change, we reuse the cached test
		// result without even rerunning the linker. The attempt using
		// the link output (test binary) content ID makes sure that if
		// we have different link inputs but the same final binary,
		// we still reuse the cached test result.
		// c.saveOutput will store the result under both IDs.
		r.c.tryCacheWithID(b, a, a.Deps[0].BuildContentID())
	}
	if r.c.buf != nil {
		if stdout != &buf {
			stdout.Write(r.c.buf.Bytes())
			r.c.buf.Reset()
		}
		a.TestOutput = r.c.buf
		return nil
	}

	execCmd := work.FindExecCmd()
	testlogArg := []string{}
	if !r.c.disableCache && len(execCmd) == 0 {
		testlogArg = []string{"-test.testlogfile=" + a.Objdir + "testlog.txt"}
	}
	panicArg := "-test.paniconexit0"
	fuzzArg := []string{}
	if testFuzz != "" {
		fuzzCacheDir := filepath.Join(cache.Default().FuzzDir(), a.Package.ImportPath)
		fuzzArg = []string{"-test.fuzzcachedir=" + fuzzCacheDir}
	}
	coverdirArg := []string{}
	addToEnv := ""
	if cfg.BuildCover {
		gcd := filepath.Join(a.Objdir, "gocoverdir")
		if err := sh.Mkdir(gcd); err != nil {
			// If we can't create a temp dir, terminate immediately
			// with an error as opposed to returning an error to the
			// caller; failed MkDir most likely indicates that we're
			// out of disk space or there is some other systemic error
			// that will make forward progress unlikely.
			base.Fatalf("failed to create temporary dir: %v", err)
		}
		coverdirArg = append(coverdirArg, "-test.gocoverdir="+gcd)
		if r.writeCoverMetaAct != nil {
			// Copy the meta-files file over into the test's coverdir
			// directory so that the coverage runtime support will be
			// able to find it.
			src := r.writeCoverMetaAct.Objdir + coverage.MetaFilesFileName
			dst := filepath.Join(gcd, coverage.MetaFilesFileName)
			if err := sh.CopyFile(dst, src, 0666, false); err != nil {
				return err
			}
		}
		// Even though we are passing the -test.gocoverdir option to
		// the test binary, also set GOCOVERDIR as well. This is
		// intended to help with tests that run "go build" to build
		// fresh copies of tools to test as part of the testing.
		addToEnv = "GOCOVERDIR=" + gcd
	}
	args := str.StringList(execCmd, a.Deps[0].BuiltTarget(), testlogArg, panicArg, fuzzArg, coverdirArg, testArgs)

	if testCoverProfile != "" {
		// Write coverage to temporary profile, for merging later.
		for i, arg := range args {
			if strings.HasPrefix(arg, "-test.coverprofile=") {
				args[i] = "-test.coverprofile=" + coverProfTempFile(a)
			}
		}
	}

	if cfg.BuildN || cfg.BuildX {
		sh.ShowCmd("", "%s", strings.Join(args, " "))
		if cfg.BuildN {
			return nil
		}
	}

	// Normally, the test will terminate itself when the timeout expires,
	// but add a last-ditch deadline to detect and stop wedged binaries.
	ctx, cancel := context.WithTimeout(ctx, testKillTimeout)
	defer cancel()

	// Now we're ready to actually run the command.
	//
	// If the -o flag is set, or if at some point we change cmd/go to start
	// copying test executables into the build cache, we may run into spurious
	// ETXTBSY errors on Unix platforms (see https://go.dev/issue/22315).
	//
	// Since we know what causes those, and we know that they should resolve
	// quickly (the ETXTBSY error will resolve as soon as the subprocess
	// holding the descriptor open reaches its 'exec' call), we retry them
	// in a loop.

	var (
		cmd            *exec.Cmd
		t0             time.Time
		cancelKilled   = false
		cancelSignaled = false
	)
	for {
		cmd = exec.CommandContext(ctx, args[0], args[1:]...)
		cmd.Dir = a.Package.Dir

		env := slices.Clip(cfg.OrigEnv)
		env = base.AppendPATH(env)
		env = base.AppendPWD(env, cmd.Dir)
		cmd.Env = env
		if addToEnv != "" {
			cmd.Env = append(cmd.Env, addToEnv)
		}

		cmd.Stdout = stdout
		cmd.Stderr = stdout

		cmd.Cancel = func {
			if base.SignalTrace == nil {
				err := cmd.Process.Kill()
				if err == nil {
					cancelKilled = true
				}
				return err
			}

			// Send a quit signal in the hope that the program will print
			// a stack trace and exit.
			err := cmd.Process.Signal(base.SignalTrace)
			if err == nil {
				cancelSignaled = true
			}
			return err
		}
		cmd.WaitDelay = testWaitDelay

		base.StartSigHandlers()
		t0 = time.Now()
		err = cmd.Run()

		if !isETXTBSY(err) {
			// We didn't hit the race in #22315, so there is no reason to retry the
			// command.
			break
		}
	}

	out := buf.Bytes()
	a.TestOutput = &buf
	t := fmt.Sprintf("%.3fs", time.Since(t0).Seconds())

	mergeCoverProfile(cmd.Stdout, a.Objdir+"_cover_.out")

	if err == nil {
		norun := ""
		if !testShowPass() && !testJSON {
			buf.Reset()
		}
		if bytes.HasPrefix(out, noTestsToRun[1:]) || bytes.Contains(out, noTestsToRun) {
			norun = " [no tests to run]"
		}
		if bytes.HasPrefix(out, noFuzzTestsToFuzz[1:]) || bytes.Contains(out, noFuzzTestsToFuzz) {
			norun = " [no fuzz tests to fuzz]"
		}
		if bytes.HasPrefix(out, tooManyFuzzTestsToFuzz[1:]) || bytes.Contains(out, tooManyFuzzTestsToFuzz) {
			norun = "[-fuzz matches more than one fuzz test, won't fuzz]"
		}
		if len(out) > 0 && !bytes.HasSuffix(out, []byte("\n")) {
			// Ensure that the output ends with a newline before the "ok"
			// line we're about to print (https://golang.org/issue/49317).
			cmd.Stdout.Write([]byte("\n"))
		}
		fmt.Fprintf(cmd.Stdout, "ok  \t%s\t%s%s%s\n", a.Package.ImportPath, t, coveragePercentage(out), norun)
		r.c.saveOutput(a)
	} else {
		if testFailFast {
			testShouldFailFast.Store(true)
		}

		base.SetExitStatus(1)
		if cancelSignaled {
			fmt.Fprintf(cmd.Stdout, "*** Test killed with %v: ran too long (%v).\n", base.SignalTrace, testKillTimeout)
		} else if cancelKilled {
			fmt.Fprintf(cmd.Stdout, "*** Test killed: ran too long (%v).\n", testKillTimeout)
		} else if errors.Is(err, exec.ErrWaitDelay) {
			fmt.Fprintf(cmd.Stdout, "*** Test I/O incomplete %v after exiting.\n", cmd.WaitDelay)
		}
		var ee *exec.ExitError
		if len(out) == 0 || !errors.As(err, &ee) || !ee.Exited() {
			// If there was no test output, print the exit status so that the reason
			// for failure is clear.
			fmt.Fprintf(cmd.Stdout, "%s\n", err)
		} else if !bytes.HasSuffix(out, []byte("\n")) {
			// Otherwise, ensure that the output ends with a newline before the FAIL
			// line we're about to print (https://golang.org/issue/49317).
			cmd.Stdout.Write([]byte("\n"))
		}

		// NOTE(golang.org/issue/37555): test2json reports that a test passes
		// unless "FAIL" is printed at the beginning of a line. The test may not
		// actually print that if it panics, exits, or terminates abnormally,
		// so we print it here. We can't always check whether it was printed
		// because some tests need stdout to be a terminal (golang.org/issue/34791),
		// not a pipe.
		// TODO(golang.org/issue/29062): tests that exit with status 0 without
		// printing a final result should fail.
		prefix := ""
		if testJSON || testV.json {
			prefix = "\x16"
		}
		fmt.Fprintf(cmd.Stdout, "%sFAIL\t%s\t%s\n", prefix, a.Package.ImportPath, t)
	}

	if cmd.Stdout != &buf {
		buf.Reset() // cmd.Stdout was going to os.Stdout already
	}
	return nil
}

// tryCache is called just before the link attempt,
// to see if the test result is cached and therefore the link is unneeded.
// It reports whether the result can be satisfied from cache.
func (c *runCache) tryCache(b *work.Builder, a *work.Action) bool {
	return c.tryCacheWithID(b, a, a.Deps[0].BuildActionID())
}

func (c *runCache) tryCacheWithID(b *work.Builder, a *work.Action, id string) bool {
	if len(pkgArgs) == 0 {
		// Caching does not apply to "go test",
		// only to "go test foo" (including "go test .").
		if cache.DebugTest {
			fmt.Fprintf(os.Stderr, "testcache: caching disabled in local directory mode\n")
		}
		c.disableCache = true
		return false
	}

	if a.Package.Root == "" {
		// Caching does not apply to tests outside of any module, GOPATH, or GOROOT.
		if cache.DebugTest {
			fmt.Fprintf(os.Stderr, "testcache: caching disabled for package outside of module root, GOPATH, or GOROOT: %s\n", a.Package.ImportPath)
		}
		c.disableCache = true
		return false
	}

	var cacheArgs []string
	for _, arg := range testArgs {
		i := strings.Index(arg, "=")
		if i < 0 || !strings.HasPrefix(arg, "-test.") {
			if cache.DebugTest {
				fmt.Fprintf(os.Stderr, "testcache: caching disabled for test argument: %s\n", arg)
			}
			c.disableCache = true
			return false
		}
		switch arg[:i] {
		case "-test.benchtime",
			"-test.cpu",
			"-test.list",
			"-test.parallel",
			"-test.run",
			"-test.short",
			"-test.timeout",
			"-test.failfast",
			"-test.v",
			"-test.fullpath":
			// These are cacheable.
			// Note that this list is documented above,
			// so if you add to this list, update the docs too.
			cacheArgs = append(cacheArgs, arg)

		default:
			// nothing else is cacheable
			if cache.DebugTest {
				fmt.Fprintf(os.Stderr, "testcache: caching disabled for test argument: %s\n", arg)
			}
			c.disableCache = true
			return false
		}
	}

	// The test cache result fetch is a two-level lookup.
	//
	// First, we use the content hash of the test binary
	// and its command-line arguments to find the
	// list of environment variables and files consulted
	// the last time the test was run with those arguments.
	// (To avoid unnecessary links, we store this entry
	// under two hashes: id1 uses the linker inputs as a
	// proxy for the test binary, and id2 uses the actual
	// test binary. If the linker inputs are unchanged,
	// this way we avoid the link step, even though we
	// do not cache link outputs.)
	//
	// Second, we compute a hash of the values of the
	// environment variables and the content of the files
	// listed in the log from the previous run.
	// Then we look up test output using a combination of
	// the hash from the first part (testID) and the hash of the
	// test inputs (testInputsID).
	//
	// In order to store a new test result, we must redo the
	// testInputsID computation using the log from the run
	// we want to cache, and then we store that new log and
	// the new outputs.

	h := cache.NewHash("testResult")
	fmt.Fprintf(h, "test binary %s args %q execcmd %q", id, cacheArgs, work.ExecCmd)
	testID := h.Sum()
	if c.id1 == (cache.ActionID{}) {
		c.id1 = testID
	} else {
		c.id2 = testID
	}
	if cache.DebugTest {
		fmt.Fprintf(os.Stderr, "testcache: %s: test ID %x => %x\n", a.Package.ImportPath, id, testID)
	}

	// Load list of referenced environment variables and files
	// from last run of testID, and compute hash of that content.
	data, entry, err := cache.GetBytes(cache.Default(), testID)
	if !bytes.HasPrefix(data, testlogMagic) || data[len(data)-1] != '\n' {
		if cache.DebugTest {
			if err != nil {
				fmt.Fprintf(os.Stderr, "testcache: %s: input list not found: %v\n", a.Package.ImportPath, err)
			} else {
				fmt.Fprintf(os.Stderr, "testcache: %s: input list malformed\n", a.Package.ImportPath)
			}
		}
		return false
	}
	testInputsID, err := computeTestInputsID(a, data)
	if err != nil {
		return false
	}
	if cache.DebugTest {
		fmt.Fprintf(os.Stderr, "testcache: %s: test ID %x => input ID %x => %x\n", a.Package.ImportPath, testID, testInputsID, testAndInputKey(testID, testInputsID))
	}

	// Parse cached result in preparation for changing run time to "(cached)".
	// If we can't parse the cached result, don't use it.
	data, entry, err = cache.GetBytes(cache.Default(), testAndInputKey(testID, testInputsID))
	if len(data) == 0 || data[len(data)-1] != '\n' {
		if cache.DebugTest {
			if err != nil {
				fmt.Fprintf(os.Stderr, "testcache: %s: test output not found: %v\n", a.Package.ImportPath, err)
			} else {
				fmt.Fprintf(os.Stderr, "testcache: %s: test output malformed\n", a.Package.ImportPath)
			}
		}
		return false
	}
	if entry.Time.Before(testCacheExpire) {
		if cache.DebugTest {
			fmt.Fprintf(os.Stderr, "testcache: %s: test output expired due to go clean -testcache\n", a.Package.ImportPath)
		}
		return false
	}
	i := bytes.LastIndexByte(data[:len(data)-1], '\n') + 1
	if !bytes.HasPrefix(data[i:], []byte("ok  \t")) {
		if cache.DebugTest {
			fmt.Fprintf(os.Stderr, "testcache: %s: test output malformed\n", a.Package.ImportPath)
		}
		return false
	}
	j := bytes.IndexByte(data[i+len("ok  \t"):], '\t')
	if j < 0 {
		if cache.DebugTest {
			fmt.Fprintf(os.Stderr, "testcache: %s: test output malformed\n", a.Package.ImportPath)
		}
		return false
	}
	j += i + len("ok  \t") + 1

	// Committed to printing.
	c.buf = new(bytes.Buffer)
	c.buf.Write(data[:j])
	c.buf.WriteString("(cached)")
	for j < len(data) && ('0' <= data[j] && data[j] <= '9' || data[j] == '.' || data[j] == 's') {
		j++
	}
	c.buf.Write(data[j:])
	return true
}

var errBadTestInputs = errors.New("error parsing test inputs")
var testlogMagic = []byte("# test log\n") // known to testing/internal/testdeps/deps.go

// computeTestInputsID computes the "test inputs ID"
// (see comment in tryCacheWithID above) for the
// test log.
func computeTestInputsID(a *work.Action, testlog []byte) (cache.ActionID, error) {
	testlog = bytes.TrimPrefix(testlog, testlogMagic)
	h := cache.NewHash("testInputs")
	// The runtime always looks at GODEBUG, without telling us in the testlog.
	fmt.Fprintf(h, "env GODEBUG %x\n", hashGetenv("GODEBUG"))
	pwd := a.Package.Dir
	for _, line := range bytes.Split(testlog, []byte("\n")) {
		if len(line) == 0 {
			continue
		}
		s := string(line)
		op, name, found := strings.Cut(s, " ")
		if !found {
			if cache.DebugTest {
				fmt.Fprintf(os.Stderr, "testcache: %s: input list malformed (%q)\n", a.Package.ImportPath, line)
			}
			return cache.ActionID{}, errBadTestInputs
		}
		switch op {
		default:
			if cache.DebugTest {
				fmt.Fprintf(os.Stderr, "testcache: %s: input list malformed (%q)\n", a.Package.ImportPath, line)
			}
			return cache.ActionID{}, errBadTestInputs
		case "getenv":
			fmt.Fprintf(h, "env %s %x\n", name, hashGetenv(name))
		case "chdir":
			pwd = name // always absolute
			fmt.Fprintf(h, "chdir %s %x\n", name, hashStat(name))
		case "stat":
			if !filepath.IsAbs(name) {
				name = filepath.Join(pwd, name)
			}
			if a.Package.Root == "" || search.InDir(name, a.Package.Root) == "" {
				// Do not recheck files outside the module, GOPATH, or GOROOT root.
				break
			}
			fmt.Fprintf(h, "stat %s %x\n", name, hashStat(name))
		case "open":
			if !filepath.IsAbs(name) {
				name = filepath.Join(pwd, name)
			}
			if a.Package.Root == "" || search.InDir(name, a.Package.Root) == "" {
				// Do not recheck files outside the module, GOPATH, or GOROOT root.
				break
			}
			fh, err := hashOpen(name)
			if err != nil {
				if cache.DebugTest {
					fmt.Fprintf(os.Stderr, "testcache: %s: input file %s: %s\n", a.Package.ImportPath, name, err)
				}
				return cache.ActionID{}, err
			}
			fmt.Fprintf(h, "open %s %x\n", name, fh)
		}
	}
	sum := h.Sum()
	return sum, nil
}

func hashGetenv(name string) cache.ActionID {
	h := cache.NewHash("getenv")
	v, ok := os.LookupEnv(name)
	if !ok {
		h.Write([]byte{0})
	} else {
		h.Write([]byte{1})
		h.Write([]byte(v))
	}
	return h.Sum()
}

const modTimeCutoff = 2 * time.Second

var errFileTooNew = errors.New("file used as input is too new")

func hashOpen(name string) (cache.ActionID, error) {
	h := cache.NewHash("open")
	info, err := os.Stat(name)
	if err != nil {
		fmt.Fprintf(h, "err %v\n", err)
		return h.Sum(), nil
	}
	hashWriteStat(h, info)
	if info.IsDir() {
		files, err := os.ReadDir(name)
		if err != nil {
			fmt.Fprintf(h, "err %v\n", err)
		}
		for _, f := range files {
			fmt.Fprintf(h, "file %s ", f.Name())
			finfo, err := f.Info()
			if err != nil {
				fmt.Fprintf(h, "err %v\n", err)
			} else {
				hashWriteStat(h, finfo)
			}
		}
	} else if info.Mode().IsRegular() {
		// Because files might be very large, do not attempt
		// to hash the entirety of their content. Instead assume
		// the mtime and size recorded in hashWriteStat above
		// are good enough.
		//
		// To avoid problems for very recent files where a new
		// write might not change the mtime due to file system
		// mtime precision, reject caching if a file was read that
		// is less than modTimeCutoff old.
		if time.Since(info.ModTime()) < modTimeCutoff {
			return cache.ActionID{}, errFileTooNew
		}
	}
	return h.Sum(), nil
}

func hashStat(name string) cache.ActionID {
	h := cache.NewHash("stat")
	if info, err := os.Stat(name); err != nil {
		fmt.Fprintf(h, "err %v\n", err)
	} else {
		hashWriteStat(h, info)
	}
	if info, err := os.Lstat(name); err != nil {
		fmt.Fprintf(h, "err %v\n", err)
	} else {
		hashWriteStat(h, info)
	}
	return h.Sum()
}

func hashWriteStat(h io.Writer, info fs.FileInfo) {
	fmt.Fprintf(h, "stat %d %x %v %v\n", info.Size(), uint64(info.Mode()), info.ModTime(), info.IsDir())
}

// testAndInputKey returns the actual cache key for the pair (testID, testInputsID).
func testAndInputKey(testID, testInputsID cache.ActionID) cache.ActionID {
	return cache.Subkey(testID, fmt.Sprintf("inputs:%x", testInputsID))
}

func (c *runCache) saveOutput(a *work.Action) {
	if c.id1 == (cache.ActionID{}) && c.id2 == (cache.ActionID{}) {
		return
	}

	// See comment about two-level lookup in tryCacheWithID above.
	testlog, err := os.ReadFile(a.Objdir + "testlog.txt")
	if err != nil || !bytes.HasPrefix(testlog, testlogMagic) || testlog[len(testlog)-1] != '\n' {
		if cache.DebugTest {
			if err != nil {
				fmt.Fprintf(os.Stderr, "testcache: %s: reading testlog: %v\n", a.Package.ImportPath, err)
			} else {
				fmt.Fprintf(os.Stderr, "testcache: %s: reading testlog: malformed\n", a.Package.ImportPath)
			}
		}
		return
	}
	testInputsID, err := computeTestInputsID(a, testlog)
	if err != nil {
		return
	}
	if c.id1 != (cache.ActionID{}) {
		if cache.DebugTest {
			fmt.Fprintf(os.Stderr, "testcache: %s: save test ID %x => input ID %x => %x\n", a.Package.ImportPath, c.id1, testInputsID, testAndInputKey(c.id1, testInputsID))
		}
		cache.PutNoVerify(cache.Default(), c.id1, bytes.NewReader(testlog))
		cache.PutNoVerify(cache.Default(), testAndInputKey(c.id1, testInputsID), bytes.NewReader(a.TestOutput.Bytes()))
	}
	if c.id2 != (cache.ActionID{}) {
		if cache.DebugTest {
			fmt.Fprintf(os.Stderr, "testcache: %s: save test ID %x => input ID %x => %x\n", a.Package.ImportPath, c.id2, testInputsID, testAndInputKey(c.id2, testInputsID))
		}
		cache.PutNoVerify(cache.Default(), c.id2, bytes.NewReader(testlog))
		cache.PutNoVerify(cache.Default(), testAndInputKey(c.id2, testInputsID), bytes.NewReader(a.TestOutput.Bytes()))
	}
}

// coveragePercentage returns the coverage results (if enabled) for the
// test. It uncovers the data by scanning the output from the test run.
func coveragePercentage(out []byte) string {
	if !cfg.BuildCover {
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

// builderCleanTest is the action for cleaning up after a test.
func builderCleanTest(b *work.Builder, ctx context.Context, a *work.Action) error {
	if cfg.BuildWork {
		return nil
	}
	b.Shell(a).RemoveAll(a.Objdir)
	return nil
}

// builderPrintTest is the action for printing a test result.
func builderPrintTest(b *work.Builder, ctx context.Context, a *work.Action) error {
	clean := a.Deps[0]
	run := clean.Deps[0]
	if run.TestOutput != nil {
		os.Stdout.Write(run.TestOutput.Bytes())
		run.TestOutput = nil
	}
	return nil
}

// printExitStatus is the action for printing the final exit status.
// If we are running multiple test targets, print a final "FAIL"
// in case a failure in an early package has already scrolled
// off of the user's terminal.
// (See https://golang.org/issue/30507#issuecomment-470593235.)
//
// In JSON mode, we need to maintain valid JSON output and
// we assume that the test output is being parsed by a tool
// anyway, so the failure will not be missed and would be
// awkward to try to wedge into the JSON stream.
//
// In fuzz mode, we only allow a single package for now
// (see CL 350156 and https://golang.org/issue/46312),
// so there is no possibility of scrolling off and no need
// to print the final status.
func printExitStatus(b *work.Builder, ctx context.Context, a *work.Action) error {
	if !testJSON && testFuzz == "" && len(pkgArgs) != 0 {
		if base.GetExitStatus() != 0 {
			fmt.Println("FAIL")
			return nil
		}
	}
	return nil
}

// testBinaryName can be used to create name for test binary executable.
// Use last element of import path, not package name.
// They differ when package name is "main".
// But if the import path is "command-line-arguments",
// like it is during 'go run', use the package name.
func testBinaryName(p *load.Package) string {
	var elem string
	if p.ImportPath == "command-line-arguments" {
		elem = p.Name
	} else {
		elem = p.DefaultExecName()
	}

	return elem + ".test"
}
