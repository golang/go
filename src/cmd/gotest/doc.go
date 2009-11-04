// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

The gotest program is an automated testing tool for Go packages.

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

See the documentation of the testing package for more information.

By default, gotest needs no arguments.  It compiles all the .go files
in the directory, including tests, and runs the tests.  If file names
are given, only those test files are added to the package.
(The non-test files are always compiled.)

The package is built in a special subdirectory so it does not
interfere with the non-test installation.

Usage:
	gotest [pkg_test.go ...]

The resulting binary, called (for amd64) 6.out, has a couple of
arguments.

Usage:
	6.out [-v] [-match pattern]

The -v flag causes the tests to be logged as they run.  The --match
flag causes only those tests whose names match the regular expression
pattern to be run. By default all tests are run silently.  If all
the specified test pass, 6.out prints PASS and exits with a 0 exit
code.  If any tests fail, it prints FAIL and exits with a non-zero
code.

*/
package documentation
