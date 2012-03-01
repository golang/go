// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Go is a tool for managing Go source code.

Usage: go command [arguments]

The commands are:

    build       compile packages and dependencies
    clean       remove object files
    doc         run godoc on package sources
    fix         run go tool fix on packages
    fmt         run gofmt on package sources
    get         download and install packages and dependencies
    install     compile and install packages and dependencies
    list        list packages
    run         compile and run Go program
    test        test packages
    tool        run specified go tool
    version     print Go version
    vet         run go tool vet on packages

Use "go help [command]" for more information about a command.

Additional help topics:

    gopath      GOPATH environment variable
    packages    description of package lists
    remote      remote import path syntax
    testflag    description of testing flags
    testfunc    description of testing functions

Use "go help [topic]" for more information about that topic.


Compile packages and dependencies

Usage:

	go build [-o output] [build flags] [packages]

Build compiles the packages named by the import paths,
along with their dependencies, but it does not install the results.

If the arguments are a list of .go files, build treats them as a list
of source files specifying a single package.

When the command line specifies a single main package,
build writes the resulting executable to output (default a.out).
Otherwise build compiles the packages but discards the results,
serving only as a check that the packages can be built.

The -o flag specifies the output file name.

The build flags are shared by the build, install, run, and test commands:

	-a
		force rebuilding of packages that are already up-to-date.
	-n
		print the commands but does not run them.
	-p n
		the number of builds that can be run in parallel.
		The default is the number of CPUs available.
	-v
		print the names of packages as they are compiled.
	-work
		print the name of the temporary work directory and
		do not delete it when exiting.
	-x
		print the commands.

	-gccgoflags 'arg list'
		arguments to pass on each gccgo compiler/linker invocation
	-gcflags 'arg list'
		arguments to pass on each 5g, 6g, or 8g compiler invocation
	-ldflags 'flag list'
		arguments to pass on each 5l, 6l, or 8l linker invocation
	-tags 'tag list'
		a list of build tags to consider satisfied during the build.
		See the documentation for the go/build package for
		more information about build tags.

For more about specifying packages, see 'go help packages'.

See also: go install, go get, go clean.


Remove object files

Usage:

	go clean [-i] [-r] [-n] [-x] [packages]

Clean removes object files from package source directories.
The go command builds most objects in a temporary directory,
so go clean is mainly concerned with object files left by other
tools or by manual invocations of go build.

Specifically, clean removes the following files from each of the
source directories corresponding to the import paths:

	_obj/            old object directory, left from Makefiles
	_test/           old test directory, left from Makefiles
	_testmain.go     old gotest file, left from Makefiles
	test.out         old test log, left from Makefiles
	build.out        old test log, left from Makefiles
	*.[568ao]        object files, left from Makefiles

	DIR(.exe)        from go build
	DIR.test(.exe)   from go test -c
	MAINFILE(.exe)   from go build MAINFILE.go

In the list, DIR represents the final path element of the
directory, and MAINFILE is the base name of any Go source
file in the directory that is not included when building
the package.

The -i flag causes clean to remove the corresponding installed
archive or binary (what 'go install' would create).

The -n flag causes clean to print the remove commands it would execute,
but not run them.

The -r flag causes clean to be applied recursively to all the
dependencies of the packages named by the import paths.

The -x flag causes clean to print remove commands as it executes them.

For more about specifying packages, see 'go help packages'.


Run godoc on package sources

Usage:

	go doc [packages]

Doc runs the godoc command on the packages named by the
import paths.

For more about godoc, see 'godoc godoc'.
For more about specifying packages, see 'go help packages'.

To run godoc with specific options, run godoc itself.

See also: go fix, go fmt, go vet.


Run go tool fix on packages

Usage:

	go fix [packages]

Fix runs the Go fix command on the packages named by the import paths.

For more about fix, see 'godoc fix'.
For more about specifying packages, see 'go help packages'.

To run fix with specific options, run 'go tool fix'.

See also: go fmt, go vet.


Run gofmt on package sources

Usage:

	go fmt [packages]

Fmt runs the command 'gofmt -l -w' on the packages named
by the import paths.  It prints the names of the files that are modified.

For more about gofmt, see 'godoc gofmt'.
For more about specifying packages, see 'go help packages'.

To run gofmt with specific options, run gofmt itself.

See also: go doc, go fix, go vet.


Download and install packages and dependencies

Usage:

	go get [-a] [-d] [-fix] [-n] [-p n] [-u] [-v] [-x] [packages]

Get downloads and installs the packages named by the import paths,
along with their dependencies.

The -a, -n, -v, -x, and -p flags have the same meaning as in 'go build'
and 'go install'.  See 'go help install'.

The -d flag instructs get to stop after downloading the packages; that is,
it instructs get not to install the packages.

The -fix flag instructs get to run the fix tool on the downloaded packages
before resolving dependencies or building the code.

The -u flag instructs get to use the network to update the named packages
and their dependencies.  By default, get uses the network to check out
missing packages but does not use it to look for updates to existing packages.

TODO: Explain versions better.

For more about specifying packages, see 'go help packages'.

For more about how 'go get' finds source code to
download, see 'go help remote'.

See also: go build, go install, go clean.


Compile and install packages and dependencies

Usage:

	go install [build flags] [packages]

Install compiles and installs the packages named by the import paths,
along with their dependencies.

For more about the build flags, see 'go help build'.
For more about specifying packages, see 'go help packages'.

See also: go build, go get, go clean.


List packages

Usage:

	go list [-e] [-f format] [-json] [packages]

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

For more about specifying packages, see 'go help packages'.


Compile and run Go program

Usage:

	go run [build flags] gofiles... [arguments...]

Run compiles and runs the main package comprising the named Go source files.

For more about build flags, see 'go help build'.

See also: go build.


Test packages

Usage:

	go test [-c] [-i] [build flags] [packages] [flags for test binary]

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

The package is built in a temporary directory so it does not interfere with the
non-test installation.

In addition to the build flags, the flags handled by 'go test' itself are:

	-c  Compile the test binary to pkg.test but do not run it.

	-i
	    Install packages that are dependencies of the test.
	    Do not run the test.

The test binary also accepts flags that control execution of the test; these
flags are also accessible by 'go test'.  See 'go help testflag' for details.

For more about build flags, see 'go help build'.
For more about specifying packages, see 'go help packages'.

See also: go build, go vet.


Run specified go tool

Usage:

	go tool [-n] command [args...]

Tool runs the go tool command identified by the arguments.
With no arguments it prints the list of known tools.

The -n flag causes tool to print the command that would be
executed but not execute it.

For more about each tool command, see 'go tool command -h'.


Print Go version

Usage:

	go version

Version prints the Go version, as reported by runtime.Version.


Run go tool vet on packages

Usage:

	go vet [packages]

Vet runs the Go vet command on the packages named by the import paths.

For more about vet, see 'godoc vet'.
For more about specifying packages, see 'go help packages'.

To run the vet tool with specific options, run 'go tool vet'.

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


Description of package lists

Many commands apply to a set of packages:

	go action [packages]

Usually, [packages] is a list of import paths.

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
patterns.  For example, encoding/... expands to all package
in subdirectories of the encoding tree, while net... expands to
net and all its subdirectories.

An import path can also name a package to be downloaded from
a remote repository.  Run 'go help remote' for details.

Every package in a program must have a unique import path.
By convention, this is arranged by starting each path with a
unique prefix that belongs to you.  For example, paths used
internally at Google all begin with 'google', and paths
denoting remote repositories begin with the path to the code,
such as 'code.google.com/p/project'.

As a special case, if the package list is a list of .go files from a
single directory, the command is applied to a single synthesized
package made up of exactly those files, ignoring any build constraints
in those files and ignoring any other files in the directory.


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

The test binary, called pkg.test, where pkg is the name of the
directory containing the package sources, has its own flags:

	-test.v
	    Verbose output: log all tests as they are run.

	-test.run pattern
	    Run only those tests and examples matching the regular
	    expression.

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

	go test -x -v -cpuprofile=prof.out -dir=testdata -update

will compile the test binary and then run it as

	pkg.test -test.v -test.cpuprofile=prof.out -dir=testdata -update


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


*/
package documentation

// NOTE: cmdDoc is in fmt.go.
