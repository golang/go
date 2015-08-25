// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var helpC = &Command{
	UsageLine: "c",
	Short:     "calling between Go and C",
	Long: `
There are two different ways to call between Go and C/C++ code.

The first is the cgo tool, which is part of the Go distribution.  For
information on how to use it see the cgo documentation (go doc cmd/cgo).

The second is the SWIG program, which is a general tool for
interfacing between languages.  For information on SWIG see
http://swig.org/.  When running go build, any file with a .swig
extension will be passed to SWIG.  Any file with a .swigcxx extension
will be passed to SWIG with the -c++ option.

When either cgo or SWIG is used, go build will pass any .c, .m, .s,
or .S files to the C compiler, and any .cc, .cpp, .cxx files to the C++
compiler.  The CC or CXX environment variables may be set to determine
the C or C++ compiler, respectively, to use.
	`,
}

var helpPackages = &Command{
	UsageLine: "packages",
	Short:     "description of package lists",
	Long: `
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

There are four reserved names for paths that should not be used
for packages to be built with the go tool:

- "main" denotes the top-level package in a stand-alone executable.

- "all" expands to all package directories found in all the GOPATH
trees. For example, 'go list all' lists all the packages on the local
system.

- "std" is like all but expands to just the packages in the standard
Go library.

- "cmd" expands to the Go repository's commands and their
internal libraries.

An import path is a pattern if it includes one or more "..." wildcards,
each of which can match any string, including the empty string and
strings containing slashes.  Such a pattern expands to all package
directories found in the GOPATH trees with names matching the
patterns.  As a special case, x/... matches x as well as x's subdirectories.
For example, net/... expands to net and packages in its subdirectories.

An import path can also name a package to be downloaded from
a remote repository.  Run 'go help importpath' for details.

Every package in a program must have a unique import path.
By convention, this is arranged by starting each path with a
unique prefix that belongs to you.  For example, paths used
internally at Google all begin with 'google', and paths
denoting remote repositories begin with the path to the code,
such as 'github.com/user/repo'.

As a special case, if the package list is a list of .go files from a
single directory, the command is applied to a single synthesized
package made up of exactly those files, ignoring any build constraints
in those files and ignoring any other files in the directory.

Directory and file names that begin with "." or "_" are ignored
by the go tool, as are directories named "testdata".
	`,
}

var helpImportPath = &Command{
	UsageLine: "importpath",
	Short:     "import path syntax",
	Long: `

An import path (see 'go help packages') denotes a package
stored in the local file system.  In general, an import path denotes
either a standard package (such as "unicode/utf8") or a package
found in one of the work spaces (see 'go help gopath').

Relative import paths

An import path beginning with ./ or ../ is called a relative path.
The toolchain supports relative import paths as a shortcut in two ways.

First, a relative path can be used as a shorthand on the command line.
If you are working in the directory containing the code imported as
"unicode" and want to run the tests for "unicode/utf8", you can type
"go test ./utf8" instead of needing to specify the full path.
Similarly, in the reverse situation, "go test .." will test "unicode" from
the "unicode/utf8" directory. Relative patterns are also allowed, like
"go test ./..." to test all subdirectories. See 'go help packages' for details
on the pattern syntax.

Second, if you are compiling a Go program not in a work space,
you can use a relative path in an import statement in that program
to refer to nearby code also not in a work space.
This makes it easy to experiment with small multipackage programs
outside of the usual work spaces, but such programs cannot be
installed with "go install" (there is no work space in which to install them),
so they are rebuilt from scratch each time they are built.
To avoid ambiguity, Go programs cannot use relative import paths
within a work space.

Remote import paths

Certain import paths also
describe how to obtain the source code for the package using
a revision control system.

A few common code hosting sites have special syntax:

	Bitbucket (Git, Mercurial)

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

	IBM DevOps Services (Git)

		import "hub.jazz.net/git/user/project"
		import "hub.jazz.net/git/user/project/sub/directory"

For code hosted on other servers, import paths may either be qualified
with the version control type, or the go tool can dynamically fetch
the import path over https/http and discover where the code resides
from a <meta> tag in the HTML.

To declare the code location, an import path of the form

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
example.org/repo or repo.git.

When a version control system supports multiple protocols,
each is tried in turn when downloading.  For example, a Git
download tries https://, then git+ssh://.

If the import path is not a known code hosting site and also lacks a
version control qualifier, the go tool attempts to fetch the import
over https/http and looks for a <meta> tag in the document's HTML
<head>.

The meta tag has the form:

	<meta name="go-import" content="import-prefix vcs repo-root">

The import-prefix is the import path corresponding to the repository
root. It must be a prefix or an exact match of the package being
fetched with "go get". If it's not an exact match, another http
request is made at the prefix to verify the <meta> tags match.

The meta tag should appear as early in the file as possible.
In particular, it should appear before any raw JavaScript or CSS,
to avoid confusing the go command's restricted parser.

The vcs is one of "git", "hg", "svn", etc,

The repo-root is the root of the version control system
containing a scheme and not containing a .vcs qualifier.

For example,

	import "example.org/pkg/foo"

will result in the following requests:

	https://example.org/pkg/foo?go-get=1 (preferred)
	http://example.org/pkg/foo?go-get=1  (fallback, only with -insecure)

If that page contains the meta tag

	<meta name="go-import" content="example.org git https://code.org/r/p/exproj">

the go tool will verify that https://example.org/?go-get=1 contains the
same meta tag and then git clone https://code.org/r/p/exproj into
GOPATH/src/example.org.

New downloaded packages are written to the first directory
listed in the GOPATH environment variable (see 'go help gopath').

The go command attempts to download the version of the
package appropriate for the Go release being used.
Run 'go help get' for more.

Import path checking

When the custom import path feature described above redirects to a
known code hosting site, each of the resulting packages has two possible
import paths, using the custom domain or the known hosting site.

A package statement is said to have an "import comment" if it is immediately
followed (before the next newline) by a comment of one of these two forms:

	package math // import "path"
	package math /* import "path" */

The go command will refuse to install a package with an import comment
unless it is being referred to by that import path. In this way, import comments
let package authors make sure the custom import path is used and not a
direct path to the underlying code hosting site.

If the vendoring experiment is enabled (see 'go help gopath'),
then import path checking is disabled for code found within vendor trees.
This makes it possible to copy code into alternate locations in vendor trees
without needing to update import comments.

See https://golang.org/s/go14customimport for details.
	`,
}

var helpGopath = &Command{
	UsageLine: "gopath",
	Short:     "GOPATH environment variable",
	Long: `
The Go path is used to resolve import statements.
It is implemented by and documented in the go/build package.

The GOPATH environment variable lists places to look for Go code.
On Unix, the value is a colon-separated string.
On Windows, the value is a semicolon-separated string.
On Plan 9, the value is a list.

GOPATH must be set to get, build and install packages outside the
standard Go tree.

Each directory listed in GOPATH must have a prescribed structure:

The src directory holds source code.  The path below src
determines the import path or executable name.

The pkg directory holds installed package objects.
As in the Go tree, each target operating system and
architecture pair has its own subdirectory of pkg
(pkg/GOOS_GOARCH).

If DIR is a directory listed in the GOPATH, a package with
source in DIR/src/foo/bar can be imported as "foo/bar" and
has its compiled form installed to "DIR/pkg/GOOS_GOARCH/foo/bar.a".

The bin directory holds compiled commands.
Each command is named for its source directory, but only
the final element, not the entire path.  That is, the
command with source in DIR/src/foo/quux is installed into
DIR/bin/quux, not DIR/bin/foo/quux.  The "foo/" prefix is stripped
so that you can add DIR/bin to your PATH to get at the
installed commands.  If the GOBIN environment variable is
set, commands are installed to the directory it names instead
of DIR/bin.

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

See https://golang.org/doc/code.html for an example.

Internal Directories

Code in or below a directory named "internal" is importable only
by code in the directory tree rooted at the parent of "internal".
Here's an extended version of the directory layout above:

    /home/user/gocode/
        src/
            crash/
                bang/              (go code in package bang)
                    b.go
            foo/                   (go code in package foo)
                f.go
                bar/               (go code in package bar)
                    x.go
                internal/
                    baz/           (go code in package baz)
                        z.go
                quux/              (go code in package main)
                    y.go


The code in z.go is imported as "foo/internal/baz", but that
import statement can only appear in source files in the subtree
rooted at foo. The source files foo/f.go, foo/bar/x.go, and
foo/quux/y.go can all import "foo/internal/baz", but the source file
crash/bang/b.go cannot.

See https://golang.org/s/go14internal for details.

Vendor Directories

Go 1.5 includes experimental support for using local copies
of external dependencies to satisfy imports of those dependencies,
often referred to as vendoring. Setting the environment variable
GO15VENDOREXPERIMENT=1 enables that experimental support.

When the vendor experiment is enabled,
code below a directory named "vendor" is importable only
by code in the directory tree rooted at the parent of "vendor",
and only using an import path that omits the prefix up to and
including the vendor element.

Here's the example from the previous section,
but with the "internal" directory renamed to "vendor"
and a new foo/vendor/crash/bang directory added:

    /home/user/gocode/
        src/
            crash/
                bang/              (go code in package bang)
                    b.go
            foo/                   (go code in package foo)
                f.go
                bar/               (go code in package bar)
                    x.go
                vendor/
                    crash/
                        bang/      (go code in package bang)
                            b.go
                    baz/           (go code in package baz)
                        z.go
                quux/              (go code in package main)
                    y.go

The same visibility rules apply as for internal, but the code
in z.go is imported as "baz", not as "foo/vendor/baz".

Code in vendor directories deeper in the source tree shadows
code in higher directories. Within the subtree rooted at foo, an import
of "crash/bang" resolves to "foo/vendor/crash/bang", not the
top-level "crash/bang".

Code in vendor directories is not subject to import path
checking (see 'go help importpath').

When the vendor experiment is enabled, 'go get' checks out
submodules when checking out or updating a git repository
(see 'go help get').

The vendoring semantics are an experiment, and they may change
in future releases. Once settled, they will be on by default.

See https://golang.org/s/go15vendor for details.
	`,
}

var helpEnvironment = &Command{
	UsageLine: "environment",
	Short:     "environment variables",
	Long: `

The go command, and the tools it invokes, examine a few different
environment variables. For many of these, you can see the default
value of on your system by running 'go env NAME', where NAME is the
name of the variable.

General-purpose environment variables:

	GCCGO
		The gccgo command to run for 'go build -compiler=gccgo'.
	GOARCH
		The architecture, or processor, for which to compile code.
		Examples are amd64, 386, arm, ppc64.
	GOBIN
		The directory where 'go install' will install a command.
	GOOS
		The operating system for which to compile code.
		Examples are linux, darwin, windows, netbsd.
	GOPATH
		See 'go help gopath'.
	GORACE
		Options for the race detector.
		See https://golang.org/doc/articles/race_detector.html.
	GOROOT
		The root of the go tree.

Environment variables for use with cgo:

	CC
		The command to use to compile C code.
	CGO_ENABLED
		Whether the cgo command is supported.  Either 0 or 1.
	CGO_CFLAGS
		Flags that cgo will pass to the compiler when compiling
		C code.
	CGO_CPPFLAGS
		Flags that cgo will pass to the compiler when compiling
		C or C++ code.
	CGO_CXXFLAGS
		Flags that cgo will pass to the compiler when compiling
		C++ code.
	CGO_LDFLAGS
		Flags that cgo will pass to the compiler when linking.
	CXX
		The command to use to compile C++ code.

Architecture-specific environment variables:

	GOARM
		For GOARCH=arm, the ARM architecture for which to compile.
		Valid values are 5, 6, 7.
	GO386
		For GOARCH=386, the floating point instruction set.
		Valid values are 387, sse2.

Special-purpose environment variables:

	GOROOT_FINAL
		The root of the installed Go tree, when it is
		installed in a location other than where it is built.
		File names in stack traces are rewritten from GOROOT to
		GOROOT_FINAL.
	GO15VENDOREXPERIMENT
		Set to 1 to enable the Go 1.5 vendoring experiment.
	GO_EXTLINK_ENABLED
		Whether the linker should use external linking mode
		when using -linkmode=auto with code that uses cgo.
		Set to 0 to disable external linking mode, 1 to enable it.
	`,
}

var helpFileType = &Command{
	UsageLine: "filetype",
	Short:     "file types",
	Long: `
The go command examines the contents of a restricted set of files
in each directory. It identifies which files to examine based on
the extension of the file name. These extensions are:

	.go
		Go source files.
	.c, .h
		C source files.
		If the package uses cgo or SWIG, these will be compiled with the
		OS-native compiler (typically gcc); otherwise they will
		trigger an error.
	.cc, .cpp, .cxx, .hh, .hpp, .hxx
		C++ source files. Only useful with cgo or SWIG, and always
		compiled with the OS-native compiler.
	.m
		Objective-C source files. Only useful with cgo, and always
		compiled with the OS-native compiler.
	.s, .S
		Assembler source files.
		If the package uses cgo or SWIG, these will be assembled with the
		OS-native assembler (typically gcc (sic)); otherwise they
		will be assembled with the Go assembler.
	.swig, .swigcxx
		SWIG definition files.
	.syso
		System object files.

Files of each of these types except .syso may contain build
constraints, but the go command stops scanning for build constraints
at the first item in the file that is not a blank line or //-style
line comment.
	`,
}

var helpBuildmode = &Command{
	UsageLine: "buildmode",
	Short:     "description of build modes",
	Long: `
The 'go build' and 'go install' commands take a -buildmode argument which
indicates which kind of object file is to be built. Currently supported values
are:

	-buildmode=archive
		Build the listed non-main packages into .a files. Packages named
		main are ignored.

	-buildmode=c-archive
		Build the listed main package, plus all packages it imports,
		into a C archive file. The only callable symbols will be those
		functions exported using a cgo //export comment. Requires
		exactly one main package to be listed.

	-buildmode=c-shared
		Build the listed main packages, plus all packages that they
		import, into C shared libraries. The only callable symbols will
		be those functions exported using a cgo //export comment.
		Non-main packages are ignored.

	-buildmode=default
		Listed main packages are built into executables and listed
		non-main packages are built into .a files (the default
		behavior).

	-buildmode=shared
		Combine all the listed non-main packages into a single shared
		library that will be used when building with the -linkshared
		option. Packages named main are ignored.

	-buildmode=exe
		Build the listed main packages and everything they import into
		executables. Packages not named main are ignored.
`,
}
