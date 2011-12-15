// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var cmdTest = &Command{
	Run:       runTest,
	UsageLine: "test [importpath...] [-file a.go -file b.go ...] [-c] [-x] [flags for test binary]",
	Short:     "test packages",
	Long: `
'Go test' automates testing the packages named by the import paths.
It prints a summary of the test results in the format:

	test archive/tar
	FAIL archive/zip
	test compress/gzip
	...

followed by detailed output for each failed package.

'Go test' recompiles each package along with any files with names matching
the file pattern "*_test.go".  These additional files can contain test functions,
benchmark functions, and example functions.  See 'go help testfunc' for more.

By default, gotest needs no arguments.  It compiles and tests the package
with source in the current directory, including tests, and runs the tests.
If file names are given (with flag -file=test.go, one per extra test source file),
only those test files are added to the package.  (The non-test files are always
compiled.)

The package is built in a temporary directory so it does not interfere with the
non-test installation.

See 'go help testflag' for details about flags
handled by 'go test' and the test binary.

See 'go help importpath' for more about import paths.

See also: go build, go compile, go vet.
	`,
}

var helpTestflag = &Command{
	UsageLine: "testflag",
	Short:     "description of testing flags",
	Long: `
The 'go test' command takes both flags that apply to 'go test' itself
and flags that apply to the resulting test binary.

The flags handled by 'go test' are:

	-c  Compile the test binary to test.out but do not run it.

	-file a.go
	    Use only the tests in the source file a.go.
	    Multiple -file flags may be provided.

	-x  Print each subcommand gotest executes.

The resulting test binary, called test.out, has its own flags:

	-test.v
	    Verbose output: log all tests as they are run.

	-test.run pattern
	    Run only those tests matching the regular expression.

	-test.bench pattern
	    Run benchmarks matching the regular expression.
	    By default, no benchmarks run.

	-test.cpuprofile cpu.out
	    Write a CPU profile to the specified file before exiting.

	-test.memprofile mem.out
	    Write a memory profile to the specified file when all tests
	    are complete.

	-test.memprofilerate n
	    Enable more precise (and expensive) memory profiles by setting
	    runtime.MemProfileRate.  See 'godoc runtime MemProfileRate'.
	    To profile all memory allocations, use -test.memprofilerate=1
	    and set the environment variable GOGC=off to disable the
	    garbage collector, provided the test can run in the available
	    memory without garbage collection.

	-test.parallel n
	    Allow parallel execution of test functions that call t.Parallel.
	    The value of this flag is the maximum number of tests to run
	    simultaneously; by default, it is set to the value of GOMAXPROCS.

	-test.short
	    Tell long-running tests to shorten their run time.
	    It is off by default but set during all.bash so that installing
	    the Go tree can run a sanity check but not spend time running
	    exhaustive tests.

	-test.timeout n
		If a test runs longer than n seconds, panic.

	-test.benchtime n
		Run enough iterations of each benchmark to take n seconds.
		The default is 1 second.

	-test.cpu 1,2,4
	    Specify a list of GOMAXPROCS values for which the tests or 
	    benchmarks should be executed.  The default is the current value
	    of GOMAXPROCS.

For convenience, each of these -test.X flags of the test binary is
also available as the flag -X in 'go test' itself.  Flags not listed
here are passed through unaltered.  For instance, the command

	go test -x -v -cpuprofile=prof.out -dir=testdata -update -file x_test.go

will compile the test binary using x_test.go and then run it as

	test.out -test.v -test.cpuprofile=prof.out -dir=testdata -update
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

An example function is similar to a test function but, instead of using *testing.T
to report success or failure, prints output to os.Stdout and os.Stderr.
That output is compared against the function's doc comment.
An example without a doc comment is compiled but not executed.

Godoc displays the body of ExampleXXX to demonstrate the use
of the function, constant, or variable XXX.  An example of a method M with
receiver type T or *T is named ExampleT_M.  There may be multiple examples
for a given function, constant, or variable, distinguished by a trailing _xxx,
where xxx is a suffix not beginning with an upper case letter.

Here is an example of an example:

	// The output of this example function.
	func ExamplePrintln() {
		Println("The output of this example function.")
	}

See the documentation of the testing package for more information.
		`,
}

func runTest(cmd *Command, args []string) {
	args = importPaths(args)
	_ = args
	panic("test not implemented")
}
