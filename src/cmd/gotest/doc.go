// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

Gotest is an automated testing tool for Go packages.

Normally a Go package is compiled without its test files.  Gotest is a
tool that recompiles the package whose source is in the current
directory, along with any files whose names match the pattern
"[^.]*_test.go".  Functions in the test source named TestXXX (where
XXX is any alphanumeric string not starting with a lower case letter)
will be run when the binary is executed.  Gotest requires that the
package have a standard package Makefile, one that includes
go/src/Make.pkg.

The test functions are run in the order they appear in the source.
They should have the signature,

	func TestXXX(t *testing.T) { ... }

Benchmark functions can be written as well; they will be run only when
the -test.bench flag is provided.  Benchmarks should have the
signature,

	func BenchmarkXXX(b *testing.B) { ... }

Example functions may also be written. They are similar to test functions but,
instead of using *testing.T to report success or failure, their output to
os.Stdout and os.Stderr is compared against their doc comment.

	// The output of this example function.
	func ExampleXXX() {
		fmt.Println("The output of this example function.")
	}

The following naming conventions are used to declare examples for a function F, 
a type T and method M on type T:
	 func ExampleF() { ... }     and    func ExampleF_suffix() { ... } 
	 func ExampleT() { ... }     and    func ExampleT_suffix() { ... }
	 func ExampleT_M() { ... }   and    func ExampleT_M_suffix() { ... }

Multiple example functions may be provided by appending a distinct suffix
to the name.  The suffix must start with a lowercase letter.

Example functions without doc comments are compiled but not executed.

See the documentation of the testing package for more information.

By default, gotest needs no arguments.  It compiles all the .go files
in the directory, including tests, and runs the tests.  If file names
are given (with flag -file=test.go, one per extra test source file),
only those test files are added to the package.  (The non-test files
are always compiled.)

The package is built in a special subdirectory so it does not
interfere with the non-test installation.

Usage:
	gotest [-file a.go -file b.go ...] [-c] [-x] [args for test binary]

The flags specific to gotest are:
	-c         Compile the test binary but do not run it.
	-file a.go Use only the tests in the source file a.go.
	           Multiple -file flags may be provided.
	-x         Print each subcommand gotest executes.

Everything else on the command line is passed to the test binary.

The resulting test binary, called (for amd64) 6.out, has several flags.

Usage:
	6.out [-test.v] [-test.run pattern] [-test.bench pattern] \
		[-test.cpuprofile=cpu.out] \
		[-test.memprofile=mem.out] [-test.memprofilerate=1] \
		[-test.parallel=$GOMAXPROCS] \
		[-test.timeout=10] [-test.short] \
		[-test.benchtime=3] [-test.cpu=1,2,3,4]

The -test.v flag causes the tests to be logged as they run.  The
-test.run flag causes only those tests whose names match the regular
expression pattern to be run.  By default all tests are run silently.

If all specified tests pass, 6.out prints the word PASS and exits with
a 0 exit code.  If any tests fail, it prints error details, the word
FAIL, and exits with a non-zero code.  The -test.bench flag is
analogous to the -test.run flag, but applies to benchmarks.  No
benchmarks run by default.

The -test.cpuprofile flag causes the testing software to write a CPU
profile to the specified file before exiting.

The -test.memprofile flag causes the testing software to write a
memory profile to the specified file when all tests are complete.  The
-test.memprofilerate flag enables more precise (and expensive)
profiles by setting runtime.MemProfileRate; run
	godoc runtime MemProfileRate
for details.  The defaults are no memory profile and the standard
setting of MemProfileRate.  The memory profile records a sampling of
the memory in use at the end of the test.  To profile all memory
allocations, use -test.memprofilerate=1 to sample every byte and set
the environment variable GOGC=off to disable the garbage collector,
provided the test can run in the available memory without garbage
collection.

Use -test.run or -test.bench to limit profiling to a particular test
or benchmark.

The -test.parallel flag allows parallel execution of Test functions
that call test.Parallel.  The value of the flag is the maximum
number of tests to run simultaneously; by default, it is set to the
value of GOMAXPROCS.

The -test.short flag tells long-running tests to shorten their run
time.  It is off by default but set by all.bash so installations of
the Go tree can do a sanity check but not spend time running
exhaustive tests.

The -test.timeout flag sets a timeout for the test in seconds.  If the
test runs for longer than that, it will panic, dumping a stack trace
of all existing goroutines.

The -test.benchtime flag specifies the number of seconds to run each benchmark.
The default is one second.

The -test.cpu flag specifies a list of GOMAXPROCS values for which
the tests or benchmarks are executed.  The default is the current
value of GOMAXPROCS.

For convenience, each of these -test.X flags of the test binary is
also available as the flag -X in gotest itself.  Flags not listed here
are unaffected.  For instance, the command
	gotest -x -v -cpuprofile=prof.out -dir=testdata -update -file x_test.go
will compile the test binary using x_test.go and then run it as
	6.out -test.v -test.cpuprofile=prof.out -dir=testdata -update

*/
package documentation
