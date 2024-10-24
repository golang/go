// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package help

import "cmd/go/internal/base"

var HelpC = &base.Command{
	UsageLine: "c",
	Short:     "calling between Go and C",
	Long: `
There are two different ways to call between Go and C/C++ code.

The first is the cgo tool, which is part of the Go distribution. For
information on how to use it see the cgo documentation (go doc cmd/cgo).

The second is the SWIG program, which is a general tool for
interfacing between languages. For information on SWIG see
http://swig.org/. When running go build, any file with a .swig
extension will be passed to SWIG. Any file with a .swigcxx extension
will be passed to SWIG with the -c++ option.

When either cgo or SWIG is used, go build will pass any .c, .m, .s, .S
or .sx files to the C compiler, and any .cc, .cpp, .cxx files to the C++
compiler. The CC or CXX environment variables may be set to determine
the C or C++ compiler, respectively, to use.
	`,
}

var HelpPackages = &base.Command{
	UsageLine: "packages",
	Short:     "package lists and patterns",
	Long: `
Many commands apply to a set of packages:

	go <action> [packages]

Usually, [packages] is a list of import paths.

An import path that is a rooted path or that begins with
a . or .. element is interpreted as a file system path and
denotes the package in that directory.

Otherwise, the import path P denotes the package found in
the directory DIR/src/P for some DIR listed in the GOPATH
environment variable (For more details see: 'go help gopath').

If no import paths are given, the action applies to the
package in the current directory.

There are four reserved names for paths that should not be used
for packages to be built with the go tool:

- "main" denotes the top-level package in a stand-alone executable.

- "all" expands to all packages in the main module (or workspace modules) and
their dependencies, including dependencies needed by tests of any of those. In
GOPATH mode, "all" expands to all packages found in all the GOPATH trees.

- "std" is like all but expands to just the packages in the standard
Go library.

- "cmd" expands to the Go repository's commands and their
internal libraries.

Package names match against fully-qualified import paths or patterns that
match against any number of import paths. For instance, "fmt" refers to the
standard library's package fmt, but "http" alone for package http would not
match the import path "net/http" from the standard library. Instead, the
complete import path "net/http" must be used.

Import paths beginning with "cmd/" only match source code in
the Go repository.

An import path is a pattern if it includes one or more "..." wildcards,
each of which can match any string, including the empty string and
strings containing slashes. Such a pattern expands to all package
directories found in the GOPATH trees with names matching the
patterns.

To make common patterns more convenient, there are two special cases.
First, /... at the end of the pattern can match an empty string,
so that net/... matches both net and packages in its subdirectories, like net/http.
Second, any slash-separated pattern element containing a wildcard never
participates in a match of the "vendor" element in the path of a vendored
package, so that ./... does not match packages in subdirectories of
./vendor or ./mycode/vendor, but ./vendor/... and ./mycode/vendor/... do.
Note, however, that a directory named vendor that itself contains code
is not a vendored package: cmd/vendor would be a command named vendor,
and the pattern cmd/... matches it.
See golang.org/s/go15vendor for more about vendoring.

An import path can also name a package to be downloaded from
a remote repository. Run 'go help importpath' for details.

Every package in a program must have a unique import path.
By convention, this is arranged by starting each path with a
unique prefix that belongs to you. For example, paths used
internally at Google all begin with 'google', and paths
denoting remote repositories begin with the path to the code,
such as 'github.com/user/repo'. Package patterns should include this prefix.
For instance, a package called 'http' residing under 'github.com/user/repo',
would be addressed with the fully-qualified pattern:
'github.com/user/repo/http'.

Packages in a program need not have unique package names,
but there are two reserved package names with special meaning.
The name main indicates a command, not a library.
Commands are built into binaries and cannot be imported.
The name documentation indicates documentation for
a non-Go program in the directory. Files in package documentation
are ignored by the go command.

As a special case, if the package list is a list of .go files from a
single directory, the command is applied to a single synthesized
package made up of exactly those files, ignoring any build constraints
in those files and ignoring any other files in the directory.

Directory and file names that begin with "." or "_" are ignored
by the go tool, as are directories named "testdata".
	`,
}

var HelpImportPath = &base.Command{
	UsageLine: "importpath",
	Short:     "import path syntax",
	Long: `

An import path (see 'go help packages') denotes a package stored in the local
file system. In general, an import path denotes either a standard package (such
as "unicode/utf8") or a package found in one of the work spaces (For more
details see: 'go help gopath').

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
that repository. The supported version control systems are:

	Bazaar      .bzr
	Fossil      .fossil
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
each is tried in turn when downloading. For example, a Git
download tries https://, then git+ssh://.

By default, downloads are restricted to known secure protocols
(e.g. https, ssh). To override this setting for Git downloads, the
GIT_ALLOW_PROTOCOL environment variable can be set (For more details see:
'go help environment').

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

The vcs is one of "bzr", "fossil", "git", "hg", "svn".

The repo-root is the root of the version control system
containing a scheme and not containing a .vcs qualifier.

For example,

	import "example.org/pkg/foo"

will result in the following requests:

	https://example.org/pkg/foo?go-get=1 (preferred)
	http://example.org/pkg/foo?go-get=1  (fallback, only with use of correctly set GOINSECURE)

If that page contains the meta tag

	<meta name="go-import" content="example.org git https://code.org/r/p/exproj">

the go tool will verify that https://example.org/?go-get=1 contains the
same meta tag and then git clone https://code.org/r/p/exproj into
GOPATH/src/example.org.

When using GOPATH, downloaded packages are written to the first directory
listed in the GOPATH environment variable.
(See 'go help gopath-get' and 'go help gopath'.)

When using modules, downloaded packages are stored in the module cache.
See https://golang.org/ref/mod#module-cache.

When using modules, an additional variant of the go-import meta tag is
recognized and is preferred over those listing version control systems.
That variant uses "mod" as the vcs in the content value, as in:

	<meta name="go-import" content="example.org mod https://code.org/moduleproxy">

This tag means to fetch modules with paths beginning with example.org
from the module proxy available at the URL https://code.org/moduleproxy.
See https://golang.org/ref/mod#goproxy-protocol for details about the
proxy protocol.

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

Import path checking is disabled for code found within vendor trees.
This makes it possible to copy code into alternate locations in vendor trees
without needing to update import comments.

Import path checking is also disabled when using modules.
Import path comments are obsoleted by the go.mod file's module statement.

See https://golang.org/s/go14customimport for details.
	`,
}

var HelpGopath = &base.Command{
	UsageLine: "gopath",
	Short:     "GOPATH environment variable",
	Long: `
The Go path is used to resolve import statements.
It is implemented by and documented in the go/build package.

The GOPATH environment variable lists places to look for Go code.
On Unix, the value is a colon-separated string.
On Windows, the value is a semicolon-separated string.
On Plan 9, the value is a list.

If the environment variable is unset, GOPATH defaults
to a subdirectory named "go" in the user's home directory
($HOME/go on Unix, %USERPROFILE%\go on Windows),
unless that directory holds a Go distribution.
Run "go env GOPATH" to see the current GOPATH.

See https://golang.org/wiki/SettingGOPATH to set a custom GOPATH.

Each directory listed in GOPATH must have a prescribed structure:

The src directory holds source code. The path below src
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
the final element, not the entire path. That is, the
command with source in DIR/src/foo/quux is installed into
DIR/bin/quux, not DIR/bin/foo/quux. The "foo/" prefix is stripped
so that you can add DIR/bin to your PATH to get at the
installed commands. If the GOBIN environment variable is
set, commands are installed to the directory it names instead
of DIR/bin. GOBIN must be an absolute path.

Here's an example directory layout:

    GOPATH=/home/user/go

    /home/user/go/
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

GOPATH and Modules

When using modules, GOPATH is no longer used for resolving imports.
However, it is still used to store downloaded source code (in GOPATH/pkg/mod)
and compiled commands (in GOPATH/bin).

Internal Directories

Code in or below a directory named "internal" is importable only
by code in the directory tree rooted at the parent of "internal".
Here's an extended version of the directory layout above:

    /home/user/go/
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

Go 1.6 includes support for using local copies of external dependencies
to satisfy imports of those dependencies, often referred to as vendoring.

Code below a directory named "vendor" is importable only
by code in the directory tree rooted at the parent of "vendor",
and only using an import path that omits the prefix up to and
including the vendor element.

Here's the example from the previous section,
but with the "internal" directory renamed to "vendor"
and a new foo/vendor/crash/bang directory added:

    /home/user/go/
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

When 'go get' checks out or updates a git repository, it now also
updates submodules.

Vendor directories do not affect the placement of new repositories
being checked out for the first time by 'go get': those are always
placed in the main GOPATH, never in a vendor subtree.

See https://golang.org/s/go15vendor for details.
	`,
}

var HelpEnvironment = &base.Command{
	UsageLine: "environment",
	Short:     "environment variables",
	Long: `

The go command and the tools it invokes consult environment variables
for configuration. If an environment variable is unset or empty, the go
command uses a sensible default setting. To see the effective setting of
the variable <NAME>, run 'go env <NAME>'. To change the default setting,
run 'go env -w <NAME>=<VALUE>'. Defaults changed using 'go env -w'
are recorded in a Go environment configuration file stored in the
per-user configuration directory, as reported by os.UserConfigDir.
The location of the configuration file can be changed by setting
the environment variable GOENV, and 'go env GOENV' prints the
effective location, but 'go env -w' cannot change the default location.
See 'go help env' for details.

General-purpose environment variables:

	GO111MODULE
		Controls whether the go command runs in module-aware mode or GOPATH mode.
		May be "off", "on", or "auto".
		See https://golang.org/ref/mod#mod-commands.
	GCCGO
		The gccgo command to run for 'go build -compiler=gccgo'.
	GOARCH
		The architecture, or processor, for which to compile code.
		Examples are amd64, 386, arm, ppc64.
	GOAUTH
		A semicolon-separated list of authentication commands for go-import and
		HTTPS module mirror interactions. Currently supports
		"off" (disables authentication) and
		"netrc" (uses credentials from NETRC or the .netrc file in your home directory).
		The default is netrc.
	GOBIN
		The directory where 'go install' will install a command.
	GOCACHE
		The directory where the go command will store cached
		information for reuse in future builds.
	GOMODCACHE
		The directory where the go command will store downloaded modules.
	GODEBUG
		Enable various debugging facilities. See https://go.dev/doc/godebug
		for details.
	GOENV
		The location of the Go environment configuration file.
		Cannot be set using 'go env -w'.
		Setting GOENV=off in the environment disables the use of the
		default configuration file.
	GOFLAGS
		A space-separated list of -flag=value settings to apply
		to go commands by default, when the given flag is known by
		the current command. Each entry must be a standalone flag.
		Because the entries are space-separated, flag values must
		not contain spaces. Flags listed on the command line
		are applied after this list and therefore override it.
	GOINSECURE
		Comma-separated list of glob patterns (in the syntax of Go's path.Match)
		of module path prefixes that should always be fetched in an insecure
		manner. Only applies to dependencies that are being fetched directly.
		GOINSECURE does not disable checksum database validation. GOPRIVATE or
		GONOSUMDB may be used to achieve that.
	GOOS
		The operating system for which to compile code.
		Examples are linux, darwin, windows, netbsd.
	GOPATH
		Controls where various files are stored. See: 'go help gopath'.
	GOPROXY
		URL of Go module proxy. See https://golang.org/ref/mod#environment-variables
		and https://golang.org/ref/mod#module-proxy for details.
	GOPRIVATE, GONOPROXY, GONOSUMDB
		Comma-separated list of glob patterns (in the syntax of Go's path.Match)
		of module path prefixes that should always be fetched directly
		or that should not be compared against the checksum database.
		See https://golang.org/ref/mod#private-modules.
	GOROOT
		The root of the go tree.
	GOSUMDB
		The name of checksum database to use and optionally its public key and
		URL. See https://golang.org/ref/mod#authenticating.
	GOTOOLCHAIN
		Controls which Go toolchain is used. See https://go.dev/doc/toolchain.
	GOTMPDIR
		The directory where the go command will write
		temporary source files, packages, and binaries.
	GOVCS
		Lists version control commands that may be used with matching servers.
		See 'go help vcs'.
	GOWORK
		In module aware mode, use the given go.work file as a workspace file.
		By default or when GOWORK is "auto", the go command searches for a
		file named go.work in the current directory and then containing directories
		until one is found. If a valid go.work file is found, the modules
		specified will collectively be used as the main modules. If GOWORK
		is "off", or a go.work file is not found in "auto" mode, workspace
		mode is disabled.

Environment variables for use with cgo:

	AR
		The command to use to manipulate library archives when
		building with the gccgo compiler.
		The default is 'ar'.
	CC
		The command to use to compile C code.
	CGO_ENABLED
		Whether the cgo command is supported. Either 0 or 1.
	CGO_CFLAGS
		Flags that cgo will pass to the compiler when compiling
		C code.
	CGO_CFLAGS_ALLOW
		A regular expression specifying additional flags to allow
		to appear in #cgo CFLAGS source code directives.
		Does not apply to the CGO_CFLAGS environment variable.
	CGO_CFLAGS_DISALLOW
		A regular expression specifying flags that must be disallowed
		from appearing in #cgo CFLAGS source code directives.
		Does not apply to the CGO_CFLAGS environment variable.
	CGO_CPPFLAGS, CGO_CPPFLAGS_ALLOW, CGO_CPPFLAGS_DISALLOW
		Like CGO_CFLAGS, CGO_CFLAGS_ALLOW, and CGO_CFLAGS_DISALLOW,
		but for the C preprocessor.
	CGO_CXXFLAGS, CGO_CXXFLAGS_ALLOW, CGO_CXXFLAGS_DISALLOW
		Like CGO_CFLAGS, CGO_CFLAGS_ALLOW, and CGO_CFLAGS_DISALLOW,
		but for the C++ compiler.
	CGO_FFLAGS, CGO_FFLAGS_ALLOW, CGO_FFLAGS_DISALLOW
		Like CGO_CFLAGS, CGO_CFLAGS_ALLOW, and CGO_CFLAGS_DISALLOW,
		but for the Fortran compiler.
	CGO_LDFLAGS, CGO_LDFLAGS_ALLOW, CGO_LDFLAGS_DISALLOW
		Like CGO_CFLAGS, CGO_CFLAGS_ALLOW, and CGO_CFLAGS_DISALLOW,
		but for the linker.
	CXX
		The command to use to compile C++ code.
	FC
		The command to use to compile Fortran code.
	PKG_CONFIG
		Path to pkg-config tool.

Architecture-specific environment variables:

	GOARM
		For GOARCH=arm, the ARM architecture for which to compile.
		Valid values are 5, 6, 7.
		The value can be followed by an option specifying how to implement floating point instructions.
		Valid options are ,softfloat (default for 5) and ,hardfloat (default for 6 and 7).
	GOARM64
		For GOARCH=arm64, the ARM64 architecture for which to compile.
		Valid values are v8.0 (default), v8.{1-9}, v9.{0-5}.
		The value can be followed by an option specifying extensions implemented by target hardware.
		Valid options are ,lse and ,crypto.
		Note that some extensions are enabled by default starting from a certain GOARM64 version;
		for example, lse is enabled by default starting from v8.1.
	GO386
		For GOARCH=386, how to implement floating point instructions.
		Valid values are sse2 (default), softfloat.
	GOAMD64
		For GOARCH=amd64, the microarchitecture level for which to compile.
		Valid values are v1 (default), v2, v3, v4.
		See https://golang.org/wiki/MinimumRequirements#amd64
	GOMIPS
		For GOARCH=mips{,le}, whether to use floating point instructions.
		Valid values are hardfloat (default), softfloat.
	GOMIPS64
		For GOARCH=mips64{,le}, whether to use floating point instructions.
		Valid values are hardfloat (default), softfloat.
	GOPPC64
		For GOARCH=ppc64{,le}, the target ISA (Instruction Set Architecture).
		Valid values are power8 (default), power9, power10.
	GORISCV64
		For GOARCH=riscv64, the RISC-V user-mode application profile for which
		to compile. Valid values are rva20u64 (default), rva22u64.
		See https://github.com/riscv/riscv-profiles/blob/main/src/profiles.adoc
	GOWASM
		For GOARCH=wasm, comma-separated list of experimental WebAssembly features to use.
		Valid values are satconv, signext.

Environment variables for use with code coverage:

	GOCOVERDIR
		Directory into which to write code coverage data files
		generated by running a "go build -cover" binary.
		Requires that GOEXPERIMENT=coverageredesign is enabled.

Special-purpose environment variables:

	GCCGOTOOLDIR
		If set, where to find gccgo tools, such as cgo.
		The default is based on how gccgo was configured.
	GOEXPERIMENT
		Comma-separated list of toolchain experiments to enable or disable.
		The list of available experiments may change arbitrarily over time.
		See src/internal/goexperiment/flags.go for currently valid values.
		Warning: This variable is provided for the development and testing
		of the Go toolchain itself. Use beyond that purpose is unsupported.
	GO_EXTLINK_ENABLED
		Whether the linker should use external linking mode
		when using -linkmode=auto with code that uses cgo.
		Set to 0 to disable external linking mode, 1 to enable it.
	GIT_ALLOW_PROTOCOL
		Defined by Git. A colon-separated list of schemes that are allowed
		to be used with git fetch/clone. If set, any scheme not explicitly
		mentioned will be considered insecure by 'go get'.
		Because the variable is defined by Git, the default value cannot
		be set using 'go env -w'.

Additional information available from 'go env' but not read from the environment:

	GOEXE
		The executable file name suffix (".exe" on Windows, "" on other systems).
	GOGCCFLAGS
		A space-separated list of arguments supplied to the CC command.
	GOHOSTARCH
		The architecture (GOARCH) of the Go toolchain binaries.
	GOHOSTOS
		The operating system (GOOS) of the Go toolchain binaries.
	GOMOD
		The absolute path to the go.mod of the main module.
		If module-aware mode is enabled, but there is no go.mod, GOMOD will be
		os.DevNull ("/dev/null" on Unix-like systems, "NUL" on Windows).
		If module-aware mode is disabled, GOMOD will be the empty string.
	GOTELEMETRY
		The current Go telemetry mode ("off", "local", or "on").
		See "go help telemetry" for more information.
	GOTELEMETRYDIR
		The directory Go telemetry data is written is written to.
	GOTOOLDIR
		The directory where the go tools (compile, cover, doc, etc...) are installed.
	GOVERSION
		The version of the installed Go tree, as reported by runtime.Version.
	`,
}

var HelpFileType = &base.Command{
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
	.s, .S, .sx
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
line comment. See the go/build package documentation for
more details.
	`,
}

var HelpBuildmode = &base.Command{
	UsageLine: "buildmode",
	Short:     "build modes",
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
		Build the listed main package, plus all packages it imports,
		into a C shared library. The only callable symbols will
		be those functions exported using a cgo //export comment.
		Requires exactly one main package to be listed.

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

	-buildmode=pie
		Build the listed main packages and everything they import into
		position independent executables (PIE). Packages not named
		main are ignored.

	-buildmode=plugin
		Build the listed main packages, plus all packages that they
		import, into a Go plugin. Packages not named main are ignored.

On AIX, when linking a C program that uses a Go archive built with
-buildmode=c-archive, you must pass -Wl,-bnoobjreorder to the C compiler.
`,
}

var HelpCache = &base.Command{
	UsageLine: "cache",
	Short:     "build and test caching",
	Long: `
The go command caches build outputs for reuse in future builds.
The default location for cache data is a subdirectory named go-build
in the standard user cache directory for the current operating system.
Setting the GOCACHE environment variable overrides this default,
and running 'go env GOCACHE' prints the current cache directory.

The go command periodically deletes cached data that has not been
used recently. Running 'go clean -cache' deletes all cached data.

The build cache correctly accounts for changes to Go source files,
compilers, compiler options, and so on: cleaning the cache explicitly
should not be necessary in typical use. However, the build cache
does not detect changes to C libraries imported with cgo.
If you have made changes to the C libraries on your system, you
will need to clean the cache explicitly or else use the -a build flag
(see 'go help build') to force rebuilding of packages that
depend on the updated C libraries.

The go command also caches successful package test results.
See 'go help test' for details. Running 'go clean -testcache' removes
all cached test results (but not cached build results).

The go command also caches values used in fuzzing with 'go test -fuzz',
specifically, values that expanded code coverage when passed to a
fuzz function. These values are not used for regular building and
testing, but they're stored in a subdirectory of the build cache.
Running 'go clean -fuzzcache' removes all cached fuzzing values.
This may make fuzzing less effective, temporarily.

The GODEBUG environment variable can enable printing of debugging
information about the state of the cache:

GODEBUG=gocacheverify=1 causes the go command to bypass the
use of any cache entries and instead rebuild everything and check
that the results match existing cache entries.

GODEBUG=gocachehash=1 causes the go command to print the inputs
for all of the content hashes it uses to construct cache lookup keys.
The output is voluminous but can be useful for debugging the cache.

GODEBUG=gocachetest=1 causes the go command to print details of its
decisions about whether to reuse a cached test result.
`,
}

var HelpBuildConstraint = &base.Command{
	UsageLine: "buildconstraint",
	Short:     "build constraints",
	Long: `
A build constraint, also known as a build tag, is a condition under which a
file should be included in the package. Build constraints are given by a
line comment that begins

	//go:build

Build constraints can also be used to downgrade the language version
used to compile a file.

Constraints may appear in any kind of source file (not just Go), but
they must appear near the top of the file, preceded
only by blank lines and other comments. These rules mean that in Go
files a build constraint must appear before the package clause.

To distinguish build constraints from package documentation,
a build constraint should be followed by a blank line.

A build constraint comment is evaluated as an expression containing
build tags combined by ||, &&, and ! operators and parentheses.
Operators have the same meaning as in Go.

For example, the following build constraint constrains a file to
build when the "linux" and "386" constraints are satisfied, or when
"darwin" is satisfied and "cgo" is not:

	//go:build (linux && 386) || (darwin && !cgo)

It is an error for a file to have more than one //go:build line.

During a particular build, the following build tags are satisfied:

	- the target operating system, as spelled by runtime.GOOS, set with the
	  GOOS environment variable.
	- the target architecture, as spelled by runtime.GOARCH, set with the
	  GOARCH environment variable.
	- any architecture features, in the form GOARCH.feature
	  (for example, "amd64.v2"), as detailed below.
	- "unix", if GOOS is a Unix or Unix-like system.
	- the compiler being used, either "gc" or "gccgo"
	- "cgo", if the cgo command is supported (see CGO_ENABLED in
	  'go help environment').
	- a term for each Go major release, through the current version:
	  "go1.1" from Go version 1.1 onward, "go1.12" from Go 1.12, and so on.
	- any additional tags given by the -tags flag (see 'go help build').

There are no separate build tags for beta or minor releases.

If a file's name, after stripping the extension and a possible _test suffix,
matches any of the following patterns:
	*_GOOS
	*_GOARCH
	*_GOOS_GOARCH
(example: source_windows_amd64.go) where GOOS and GOARCH represent
any known operating system and architecture values respectively, then
the file is considered to have an implicit build constraint requiring
those terms (in addition to any explicit constraints in the file).

Using GOOS=android matches build tags and files as for GOOS=linux
in addition to android tags and files.

Using GOOS=illumos matches build tags and files as for GOOS=solaris
in addition to illumos tags and files.

Using GOOS=ios matches build tags and files as for GOOS=darwin
in addition to ios tags and files.

The defined architecture feature build tags are:

	- For GOARCH=386, GO386=387 and GO386=sse2
	  set the 386.387 and 386.sse2 build tags, respectively.
	- For GOARCH=amd64, GOAMD64=v1, v2, and v3
	  correspond to the amd64.v1, amd64.v2, and amd64.v3 feature build tags.
	- For GOARCH=arm, GOARM=5, 6, and 7
	  correspond to the arm.5, arm.6, and arm.7 feature build tags.
	- For GOARCH=arm64, GOARM64=v8.{0-9} and v9.{0-5}
	  correspond to the arm64.v8.{0-9} and arm64.v9.{0-5} feature build tags.
	- For GOARCH=mips or mipsle,
	  GOMIPS=hardfloat and softfloat
	  correspond to the mips.hardfloat and mips.softfloat
	  (or mipsle.hardfloat and mipsle.softfloat) feature build tags.
	- For GOARCH=mips64 or mips64le,
	  GOMIPS64=hardfloat and softfloat
	  correspond to the mips64.hardfloat and mips64.softfloat
	  (or mips64le.hardfloat and mips64le.softfloat) feature build tags.
	- For GOARCH=ppc64 or ppc64le,
	  GOPPC64=power8, power9, and power10 correspond to the
	  ppc64.power8, ppc64.power9, and ppc64.power10
	  (or ppc64le.power8, ppc64le.power9, and ppc64le.power10)
	  feature build tags.
	- For GOARCH=riscv64,
	  GORISCV64=rva20u64 and rva22u64 correspond to the riscv64.rva20u64
	  and riscv64.rva22u64 build tags.
	- For GOARCH=wasm, GOWASM=satconv and signext
	  correspond to the wasm.satconv and wasm.signext feature build tags.

For GOARCH=amd64, arm, ppc64, ppc64le, and riscv64, a particular feature level
sets the feature build tags for all previous levels as well.
For example, GOAMD64=v2 sets the amd64.v1 and amd64.v2 feature flags.
This ensures that code making use of v2 features continues to compile
when, say, GOAMD64=v4 is introduced.
Code handling the absence of a particular feature level
should use a negation:

	//go:build !amd64.v2

To keep a file from being considered for any build:

	//go:build ignore

(Any other unsatisfied word will work as well, but "ignore" is conventional.)

To build a file only when using cgo, and only on Linux and OS X:

	//go:build cgo && (linux || darwin)

Such a file is usually paired with another file implementing the
default functionality for other systems, which in this case would
carry the constraint:

	//go:build !(cgo && (linux || darwin))

Naming a file dns_windows.go will cause it to be included only when
building the package for Windows; similarly, math_386.s will be included
only when building the package for 32-bit x86.

Go versions 1.16 and earlier used a different syntax for build constraints,
with a "// +build" prefix. The gofmt command will add an equivalent //go:build
constraint when encountering the older syntax.

In modules with a Go version of 1.21 or later, if a file's build constraint
has a term for a Go major release, the language version used when compiling
the file will be the minimum version implied by the build constraint.
`,
}
