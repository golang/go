// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

Gotest is an automated testing tool for Go packages.

Normally a Go package is compiled without its test files.  Gotest
is a simple script that recompiles the package along with any files
named *_test.go.  Functions in the test sources named TestXXX
(where XXX is any alphanumeric string starting with an upper case
letter) will be run when the binary is executed.  Gotest requires
that the package have a standard package Makefile, one that
includes go/src/Make.pkg.

The test functions are run in the order they appear in the source.
They should have signature

	func TestXXX(t *testing.T) { ... }

Benchmark functions can be written as well; they will be run only
when the -test.bench flag is provided.  Benchmarks should have
signature

	func BenchmarkXXX(b *testing.B) { ... }

See the documentation of the testing package for more information.

By default, gotest needs no arguments.  It compiles all the .go files
in the directory, including tests, and runs the tests.  If file names
are given, only those test files are added to the package.
(The non-test files are always compiled.)

The package is built in a special subdirectory so it does not
interfere with the non-test installation.

Usage:
	gotest [pkg_test.go ...]

The resulting binary, called (for amd64) 6.out, has several flags.

Usage:
	6.out [-test.v] [-test.run pattern] [-test.bench pattern] \
		[-test.memprofile=prof.out] [-test.memprofilerate=1]

The -test.v flag causes the tests to be logged as they run.  The
-test.run flag causes only those tests whose names match the regular
expression pattern to be run. By default all tests are run silently.
If all the specified test pass, 6.out prints PASS and exits with a 0
exit code.  If any tests fail, it prints FAIL and exits with a
non-zero code.  The -test.bench flag is analogous to the -test.run
flag, but applies to benchmarks.  No benchmarks run by default.

The -test.memprofile flag causes the testing software to write a
memory profile to the specified file when all tests are complete.  Use
-test.run or -test.bench to limit the profile to a particular test or
benchmark.  The -test.memprofilerate flag enables more precise (and
expensive) profiles by setting runtime.MemProfileRate;
	godoc runtime MemProfileRate
for details.  The defaults are no memory profile and the standard
setting of MemProfileRate.  The memory profile records a sampling of
the memory in use at the end of the test.  To profile all memory
allocations, use -test.memprofilerate=1 to sample every byte and set
the environment variable GOGC=off to disable the garbage collector,
provided the test can run in the available memory without garbage
collection.

*/
package documentation
