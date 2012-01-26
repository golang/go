// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Go is a tool for managing Go source code.

Usage: go command [arguments]

The commands are:

    build       compile packages and dependencies
    doc         run godoc on package sources
    fix         run gofix on packages
    fmt         run gofmt on package sources
    get         download and install packages and dependencies
    install     compile and install packages and dependencies
    list        list packages
    run         compile and run Go program
    test        test packages
    version     print Go version
    vet         run govet on packages

Use "go help [command]" for more information about a command.

Additional help topics:

    gopath      GOPATH environment variable
    importpath  description of import paths
    remote      remote import path syntax
    testflag    description of testing flags
    testfunc    description of testing functions

Use "go help [topic]" for more information about that topic.


Compile packages and dependencies

Usage:

	go build [-a] [-n] [-o output] [-p n] [-v] [-x] [importpath... | gofiles...]

Build compiles the packages named by the import paths,
along with their dependencies, but it does not install the results.

If the arguments are a list of .go files, build treats them as a list
of source files specifying a single package.

When the command line specifies a single main package,
build writes the resulting executable to output (default a.out).
Otherwise build compiles the packages but discards the results,
serving only as a check that the packages can be built.

The -a flag forces rebuilding of packages that are already up-to-date.
The -n flag prints the commands but does not run them.
The -v flag prints the names of packages as they are compiled.
The -x flag prints the commands.

The -o flag specifies the output file name.
It is an error to use -o when the command line specifies multiple packages.

The -p flag specifies the number of builds that can be run in parallel.
The default is the number of CPUs available.

For more about import paths, see 'go help importpath'.

See also: go install, go get, go clean.


Run godoc on package sources

Usage:

	go doc [importpath...]

Doc runs the godoc command on the packages named by the
import paths.

For more about godoc, see 'godoc godoc'.
For more about import paths, see 'go help importpath'.

To run godoc with specific options, run godoc itself.

See also: go fix, go fmt, go vet.


Run gofix on packages

Usage:

	go fix [importpath...]

Fix runs the gofix command on the packages named by the import paths.

For more about gofix, see 'godoc gofix'.
For more about import paths, see 'go help importpath'.

To run gofix with specific options, run gofix itself.

See also: go fmt, go vet.


Run gofmt on package sources

Usage:

	go fmt [importpath...]

Fmt runs the command 'gofmt -l -w' on the packages named
by the import paths.  It prints the names of the files that are modified.

For more about gofmt, see 'godoc gofmt'.
For more about import paths, see 'go help importpath'.

To run gofmt with specific options, run gofmt itself.

See also: go doc, go fix, go vet.


Download and install packages and dependencies

Usage:

	go get [-a] [-d] [-fix] [-n] [-p n] [-u] [-v] [-x] [importpath...]

Get downloads and installs the packages named by the import paths,
along with their dependencies.

The -a, -n, -v, -x, and -p flags have the same meaning as in 'go build'
and 'go install'.  See 'go help install'.

The -d flag instructs get to stop after downloading the packages; that is,
it instructs get not to install the packages.

The -fix flag instructs get to run gofix on the downloaded packages
before resolving dependencies or building the code.

The -u flag instructs get to use the network to update the named packages
and their dependencies.  By default, get uses the network to check out 
missing packages but does not use it to look for updates to existing packages.

TODO: Explain versions better.

For more about import paths, see 'go help importpath'.

For more about how 'go get' finds source code to
download, see 'go help remote'.

See also: go build, go install, go clean.


Compile and install packages and dependencies

Usage:

	go install [-a] [-n] [-p n] [-v] [-x] [importpath...]

Install compiles and installs the packages named by the import paths,
along with their dependencies.

The -a flag forces reinstallation of packages that are already up-to-date.
The -n flag prints the commands but does not run them.
The -v flag prints the names of packages as they are compiled.
The -x flag prints the commands.

The -p flag specifies the number of builds that can be run in parallel.
The default is the number of CPUs available.

For more about import paths, see 'go help importpath'.

See also: go build, go get, go clean.


List packages

Usage:

	go list [-e] [-f format] [-json] [importpath...]

List lists the packages named by the import paths, one per line.

The default output shows the package import path:

    code.google.com/p/google-api-go-client/books/v1
    code.google.com/p/goauth2/oauth
    code.google.com/p/sqlite

The -f flag specifies an alternate format for the list,
using the syntax of package template.  The default output
is equivalent to -f '{{.ImportPath}}'.  The struct
being passed to the template is:

    type Package struct {
        Name       string // package name
        Doc        string // package documentation string
        ImportPath string // import path of package in dir
        Dir        string // directory containing package sources
        Version    string // version of installed package (TODO)
        Stale      bool   // would 'go install' do anything for this package?

        // Source files
        GoFiles      []string // .go source files (excluding CgoFiles, TestGoFiles, and XTestGoFiles)
        TestGoFiles  []string // _test.go source files internal to the package they are testing
        XTestGoFiles []string // _test.go source files external to the package they are testing
        CFiles       []string // .c source files
        HFiles       []string // .h source files
        SFiles       []string // .s source files
        CgoFiles     []string // .go sources files that import "C"

        // Dependency information
        Imports []string // import paths used by this package
        Deps    []string // all (recursively) imported dependencies

        // Error information
        Incomplete bool            // this package or a dependency has an error
        Error *PackageError        // error loading package
        DepsErrors []*PackageError // errors loading dependencies
    }

The -json flag causes the package data to be printed in JSON format
instead of using the template format.

The -e flag changes the handling of erroneous packages, those that
cannot be found or are malformed.  By default, the list command
prints an error to standard error for each erroneous package and
omits the packages from consideration during the usual printing.
With the -e flag, the list command never prints errors to standard
error and instead processes the erroneous packages with the usual
printing.  Erroneous packages will have a non-empty ImportPath and
a non-nil Error field; other information may or may not be missing
(zeroed).

For more about import paths, see 'go help importpath'.


Compile and run Go program

Usage:

	go run [-a] [-n] [-x] gofiles... [arguments...]

Run compiles and runs the main package comprising the named Go source files.

The -a flag forces reinstallation of packages that are already up-to-date.
The -n flag prints the commands but does not run them.
The -x flag prints the commands.

See also: go build.


Test packages

Usage:

	go test [-c] [-file a.go -file b.go ...] [-p n] [-x] [importpath...] [flags for test binary]

'Go test' automates testing the packages named by the import paths.
It prints a summary of the test results in the format:

	ok   archive/tar   0.011s
	FAIL archive/zip   0.022s
	ok   compress/gzip 0.033s
	...

followed by detailed output for each failed package.

'Go test' recompiles each package along with any files with names matching
the file pattern "*_test.go".  These additional files can contain test functions,
benchmark functions, and example functions.  See 'go help testfunc' for more.

By default, go test needs no arguments.  It compiles and tests the package
with source in the current directory, including tests, and runs the tests.
If file names are given (with flag -file=test.go, one per extra test source file),
only those test files are added to the package.  (The non-test files are always
compiled.)

The package is built in a temporary directory so it does not interfere with the
non-test installation.

See 'go help testflag' for details about flags handled by 'go test'
and the test binary.

See 'go help importpath' for more about import paths.

See also: go build, go vet.


Print Go version

Usage:

	go version

Version prints the Go version, as reported by runtime.Version.


Run govet on packages

Usage:

	go vet [importpath...]

Vet runs the govet command on the packages named by the import paths.

For more about govet, see 'godoc govet'.
For more about import paths, see 'go help importpath'.

To run govet with specific options, run govet itself.

See also: go fmt, go fix.


GOPATH environment variable

The GOPATH environment variable lists places to look for Go code.
On Unix, the value is a colon-separated string.
On Windows, the value is a semicolon-separated string.
On Plan 9, the value is a list.

GOPATH must be set to build and install packages outside the
standard Go tree.

Each directory listed in GOPATH must have a prescribed structure:

The src/ directory holds source code.  The path below 'src'
determines the import path or executable name.

The pkg/ directory holds installed package objects.
As in the Go tree, each target operating system and
architecture pair has its own subdirectory of pkg
(pkg/GOOS_GOARCH).

If DIR is a directory listed in the GOPATH, a package with
source in DIR/src/foo/bar can be imported as "foo/bar" and
has its compiled form installed to "DIR/pkg/GOOS_GOARCH/foo/bar.a".

The bin/ directory holds compiled commands.
Each command is named for its source directory, but only
the final element, not the entire path.  That is, the
command with source in DIR/src/foo/quux is installed into
DIR/bin/quux, not DIR/bin/foo/quux.  The foo/ is stripped
so that you can add DIR/bin to your PATH to get at the
installed commands.

Here's an example directory layout:

    GOPATH=/home/user/gocode

    /home/user/gocode/
        src/
            foo/
                bar/               (go code in package bar)
                    x.go
                quux/              (go code in package main)
                    y.go
        bin/
            quux                   (installed command)
        pkg/
            linux_amd64/
                foo/
                    bar.a          (installed package object)

Go searches each directory listed in GOPATH to find source code,
but new packages are always downloaded into the first directory 
in the list.


Description of import paths

Many commands apply to a set of packages named by import paths:

	go action [importpath...]

An import path that is a rooted path or that begins with
a . or .. element is interpreted as a file system path and
denotes the package in that directory.

Otherwise, the import path P denotes the package found in
the directory DIR/src/P for some DIR listed in the GOPATH
environment variable (see 'go help gopath'). 

If no import paths are given, the action applies to the
package in the current directory.

The special import path "all" expands to all package directories
found in all the GOPATH trees.  For example, 'go list all' 
lists all the packages on the local system.

The special import path "std" is like all but expands to just the
packages in the standard Go library.

An import path is a pattern if it includes one or more "..." wildcards,
each of which can match any string, including the empty string and
strings containing slashes.  Such a pattern expands to all package
directories found in the GOPATH trees with names matching the
patterns.  For example, encoding/... expands to all packages
in the encoding tree.

An import path can also name a package to be downloaded from
a remote repository.  Run 'go help remote' for details.

Every package in a program must have a unique import path.
By convention, this is arranged by starting each path with a
unique prefix that belongs to you.  For example, paths used
internally at Google all begin with 'google', and paths
denoting remote repositories begin with the path to the code,
such as 'code.google.com/p/project'.


Remote import path syntax

An import path (see 'go help importpath') denotes a package
stored in the local file system.  Certain import paths also
describe how to obtain the source code for the package using
a revision control system.

A few common code hosting sites have special syntax:

	BitBucket (Mercurial)

		import "bitbucket.org/user/project"
		import "bitbucket.org/user/project/sub/directory"

	GitHub (Git)

		import "github.com/user/project"
		import "github.com/user/project/sub/directory"

	Google Code Project Hosting (Git, Mercurial, Subversion)

		import "code.google.com/p/project"
		import "code.google.com/p/project/sub/directory"

		import "code.google.com/p/project.subrepository"
		import "code.google.com/p/project.subrepository/sub/directory"

	Launchpad (Bazaar)

		import "launchpad.net/project"
		import "launchpad.net/project/series"
		import "launchpad.net/project/series/sub/directory"

		import "launchpad.net/~user/project/branch"
		import "launchpad.net/~user/project/branch/sub/directory"

For code hosted on other servers, an import path of the form

	repository.vcs/path

specifies the given repository, with or without the .vcs suffix,
using the named version control system, and then the path inside
that repository.  The supported version control systems are:

	Bazaar      .bzr
	Git         .git
	Mercurial   .hg
	Subversion  .svn

For example,

	import "example.org/user/foo.hg"

denotes the root directory of the Mercurial repository at
example.org/user/foo or foo.hg, and

	import "example.org/repo.git/foo/bar"

denotes the foo/bar directory of the Git repository at
example.com/repo or repo.git.

When a version control system supports multiple protocols,
each is tried in turn when downloading.  For example, a Git
download tries git://, then https://, then http://.

New downloaded packages are written to the first directory
listed in the GOPATH environment variable (see 'go help gopath').

The go command attempts to download the version of the
package appropriate for the Go release being used.
Run 'go help install' for more.


Description of testing flags

The 'go test' command takes both flags that apply to 'go test' itself
and flags that apply to the resulting test binary.

The flags handled by 'go test' are:

	-c  Compile the test binary to test.out but do not run it.

	-file a.go
	    Use only the tests in the source file a.go.
	    Multiple -file flags may be provided.

	-p n
	    Compile and test up to n packages in parallel.
	    The default value is the number of CPUs available.

	-x  Print each subcommand go test executes.

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

	-test.timeout t
		If a test runs longer than t, panic.

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


Description of testing functions

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


*/
package documentation

// NOTE: cmdDoc is in fmt.go.
